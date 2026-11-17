"""A device's tool restriction narrows what a session on it may use, and nothing else.

Hub keeps the restriction with the device and sends it with every provision
and refresh (工具总览与开关设计 §6). The Provider takes the withheld tools out of
the two session tables, so the plan, the device and the worker's capability
transport all lose them at once; it leaves what the device declared and what
the Owner's own app may invoke as they were (I8), and carries the restriction
beside the plan for the tools no device runs.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from eidolon_sdk.biz.capabilities import installed_properties, installed_registry
from eidolon_sdk.biz.capabilities.device_settings import (
    AUDIO_VOLUME,
    AUDIO_VOLUME_ADJUST,
    AUDIO_VOLUME_SET,
)
from eidolon_sdk.biz.contracts import SESSION_INTENT_USER_INITIATED
from eidolon_sdk.biz.presentation import DeviceOutputPolicy, OutputSelection, SessionOutputPlan
from eidolon_sdk.biz.tools import DeviceToolRestriction
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

from eidolon.channel_provider.contracts import (
    ContractError,
    CurrentRequest,
    InvalidTransition,
    ProvisionRequest,
    request_fingerprint,
)
from eidolon.channel_provider.spec import derive_spec

from .helpers import current_payload, encoded, provision_payload
from .test_livekit_adapter import _adapter
from .test_service import _service

DECLARED = ("motion.nod", "motion.stop", AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST)
RESTRICTION = {"revision": 1, "denied": [AUDIO_VOLUME_SET, "get_weather", "motion.nod"]}


def _payload(*, restriction: dict | None = RESTRICTION, motion: bool = True,
             operation: str = "channel.provision-device", operation_id: str = "enrollment-1",
             device: str = "device-1") -> dict[str, Any]:
    """A head that nods and a speaker whose volume its Owner can set."""

    payload = provision_payload(
        device_id=named_device_instance_id(device),
        output_policy=DeviceOutputPolicy(
            revision=3, allowed=OutputSelection(speech=True, motion=motion),
        ).model_dump(mode="json"),
    )
    payload.update(operation=operation, operation_id=operation_id)
    if restriction is not None:
        payload["tool_restriction"] = restriction
    manifest = payload["device"]["manifest"]
    manifest["properties"] += [
        {"name": "output.motion", "observable": False, "writable": False,
         "schema": {"type": "boolean", "const": True}},
        {"name": "motion.profile", "observable": False, "writable": False,
         "schema": {"type": "string", "const": "stackchan.head.v1"}},
        installed_properties()[AUDIO_VOLUME],
    ]
    registry = installed_registry()
    manifest["actions"] = [registry.get(name).action() for name in DECLARED]
    return payload


def _request(**kwargs) -> ProvisionRequest:
    return ProvisionRequest.parse(encoded(_payload(**kwargs)))


def _spec(request: ProvisionRequest):
    return derive_spec(
        request.device, device_instance_id=request.device_ref.device_instance_id,
        agent_name="eidolon", tool_restriction=request.tool_restriction,
    )


# -- the request ------------------------------------------------------------------


def _fingerprint_before_restrictions(request: ProvisionRequest) -> str:
    """``content_fingerprint`` exactly as it was written before restrictions existed."""

    device = request.device
    return request_fingerprint({
        "operation": request.operation,
        "operation_id": request.operation_id,
        "device_ref": request.device_ref.model_dump(mode="json"),
        "device": {
            "owner_id": str(device.owner_id), "display_name": device.display_name,
            "device_kind": device.manifest_id, "manifest": device.manifest,
            "manifest_revision": device.manifest_revision,
            **({"output_policy": device.output_policy.model_dump(mode="json", exclude_unset=True)}
               if device.output_policy is not None else {}),
        },
    })


def test_an_unrestricted_request_keeps_the_fingerprint_it_always_had():
    """Rows stored before the deploy must still match their replays."""

    for payload in (provision_payload(), _payload(restriction=None)):
        request = ProvisionRequest.parse(encoded(payload))
        assert request.tool_restriction is None
        assert request.fingerprint == _fingerprint_before_restrictions(request)
    restricted = _request()
    assert restricted.tool_restriction == DeviceToolRestriction.model_validate(RESTRICTION)
    assert restricted.fingerprint != _request(restriction=None).fingerprint


@pytest.mark.parametrize("restriction", [
    {"revision": 0, "denied": []},
    {"revision": 1, "denied": ["Not A Key"]},
    {"revision": 1, "denied": [], "extra": True},
    ["get_weather"],
])
def test_a_malformed_restriction_is_refused(restriction):
    with pytest.raises(ContractError, match="tool_restriction"):
        _request(restriction=restriction)


# -- what a session may use -------------------------------------------------------


def test_the_restriction_leaves_the_declaration_and_the_owners_controls_alone():
    restricted, unrestricted = _spec(_request()), _spec(_request(restriction=None))
    assert unrestricted.motion_capabilities == ("motion.nod", "motion.stop")
    assert unrestricted.device_capabilities == (AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST)
    # Only the two session tables lose what was withheld.
    assert restricted.motion_capabilities == ("motion.stop",)
    assert restricted.device_capabilities == (AUDIO_VOLUME_ADJUST,)
    # I8: keeping a Companion from the volume leaves the Owner's slider.
    for spec in (restricted, unrestricted):
        assert spec.declared_capabilities == DECLARED
        assert spec.owner_capabilities == (AUDIO_VOLUME_SET,)
        assert spec.observable_properties == (AUDIO_VOLUME,)
        assert spec.motion_profile == "stackchan.head.v1"


async def test_the_plan_the_capability_transport_and_the_worker_lose_the_withheld_names():
    from unittest.mock import MagicMock

    from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm
    from eidolon.livekit.agent.session.capability_rpc import SessionCapabilityRPC

    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    handle = grant.handle
    assert handle["tool_restriction"] == RESTRICTION
    template = handle["output_template"]
    assert "tool_restriction" not in template
    plan = SessionOutputPlan(session_id="session-1", **template)  # still the strict plan
    assert plan.motion_capabilities == ("motion.stop",)
    assert plan.device_capabilities == (AUDIO_VOLUME_ADJUST,)
    assert SessionCapabilityRPC(SimpleNamespace(), plan).names == ("motion.stop", AUDIO_VOLUME_ADJUST)
    assert handle["owner_control"]["capabilities"] == [AUDIO_VOLUME_SET]

    # The worker, given that plan and the dispatch's restriction, offers and
    # routes neither the withheld device contracts nor its own withheld tools.
    async def music(call):
        raise AssertionError("a withheld tool was routed")

    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1", device_token=lambda: "x", conversation_id="c",
        output_plan=plan, presentation_room=MagicMock(),
        device_tool_deny=frozenset(handle["tool_restriction"]["denied"]) | {"music.play"},
    )
    llm.configure_channel_tools(music, None, frozenset({"music.play", "music.stop"}))
    assert llm._channel_tool_capabilities == frozenset(
        {"motion.stop", AUDIO_VOLUME_ADJUST, "music.stop"})
    unrestricted = await adapter.open(_spec(_request(restriction=None)), issued_at_ms=1_000)
    assert "tool_restriction" not in unrestricted.handle
    await adapter.shutdown()


# -- the conversation it reaches ------------------------------------------------


def _with_restriction(denied: list[str], revision: int) -> ProvisionRequest:
    return _request(restriction={"revision": revision, "denied": denied}, motion=False)


async def test_a_restriction_on_a_tool_no_device_runs_still_ends_the_conversation():
    """The plan does not move, so the restriction itself is what is compared (§6.5)."""

    adapter, client = _adapter()
    before = await adapter.open(_spec(_with_restriction(["get_weather"], 1)), issued_at_ms=1_000)
    after = await adapter.open(
        _spec(_with_restriction(["get_time", "get_weather"], 2)), issued_at_ms=2_000)
    assert before.handle["output_template"] == after.handle["output_template"]
    room = before.handle["room"]

    await adapter.open_session(before.handle, "session-1", session_intent=SESSION_INTENT_USER_INITIATED)
    metadata = json.loads(client.agent_dispatch.created[-1][2])
    assert metadata["tool_restriction"] == {"revision": 1, "denied": ["get_weather"]}
    # Unchanged: the running conversation stays.
    await adapter._reconcile_output_dispatches(before.handle)
    assert client.agent_dispatch.deleted == []
    # A retry of the same conversation cannot carry another restriction.
    with pytest.raises(InvalidTransition, match="cannot change"):
        await adapter.open_session(after.handle, "session-1",
                                   session_intent=SESSION_INTENT_USER_INITIATED)
    # Changed: it ends, and the next one starts under the new restriction.
    await adapter._reconcile_output_dispatches(after.handle)
    assert len(client.agent_dispatch.deleted) == 1 and client.agent_dispatch.deleted[0][0] == room
    await adapter.open_session(after.handle, "session-2", session_intent=SESSION_INTENT_USER_INITIATED)
    metadata = json.loads(client.agent_dispatch.created[-1][2])
    assert metadata["tool_restriction"] == {"revision": 2, "denied": ["get_time", "get_weather"]}
    await adapter.shutdown()


async def test_an_unrestricted_device_dispatches_as_before():
    adapter, client = _adapter()
    grant = await adapter.open(_spec(_request(restriction=None, motion=False)), issued_at_ms=1_000)
    await adapter.open_session(grant.handle, "session-1", session_intent=SESSION_INTENT_USER_INITIATED)
    assert "tool_restriction" not in json.loads(client.agent_dispatch.created[-1][2])
    await adapter._reconcile_output_dispatches(grant.handle)
    assert client.agent_dispatch.deleted == []
    await adapter.shutdown()


async def test_a_directed_session_carries_the_restriction_of_the_device_it_hears():
    adapter, client = _adapter()
    source = await adapter.open(_spec(_request(motion=False)), issued_at_ms=1_000)
    target = await adapter.open(
        _spec(_request(restriction=None, motion=False, device="device-2")), issued_at_ms=1_000)
    await adapter.open_directed_session(
        source.handle, target.handle, source_session_id="source-1",
        target_session_id="target-1", target_companion_id="companion-1")
    metadata = json.loads(client.agent_dispatch.created[-1][2])
    assert metadata["presentation_endpoint"] is not None
    assert metadata["tool_restriction"] == RESTRICTION
    await adapter.shutdown()


# -- the ledger -------------------------------------------------------------------


def _refresh(restriction: dict | None, operation_id: str) -> ProvisionRequest:
    return _request(restriction=restriction, operation="channel.refresh-device",
                    operation_id=operation_id)


async def test_the_grant_echoes_exactly_the_restriction_it_was_sent(tmp_path):
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    response = json.loads(await service.provision(_request()))
    assert response["tool_restriction"] == RESTRICTION
    current = json.loads(await service.current(CurrentRequest.parse(encoded(current_payload(
        device_id=named_device_instance_id("device-1"))))))
    assert current["binding"]["tool_restriction"] == RESTRICTION
    assert backend.opened[-1].tool_restriction == DeviceToolRestriction.model_validate(RESTRICTION)


async def test_an_unrestricted_grant_echoes_nothing(tmp_path):
    clock = [1_700_000_000_000]
    service, _store, _backend = _service(tmp_path, clock)
    response = json.loads(await service.provision(_request(restriction=None)))
    assert "tool_restriction" not in response
    current = json.loads(await service.current(CurrentRequest.parse(encoded(current_payload(
        device_id=named_device_instance_id("device-1"))))))
    assert "tool_restriction" not in current["binding"]


async def test_a_newer_restriction_alone_refreshes_the_grant(tmp_path):
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(_request())
    newer = {"revision": 2, "denied": ["get_weather"]}
    response = json.loads(await service.provision(_refresh(newer, "refresh-1")))
    assert response["tool_restriction"] == newer
    assert backend.opened[-1].device_capabilities == (AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST)


@pytest.mark.parametrize("restriction", [
    None,                                                   # withdrawn
    {"revision": 1, "denied": ["get_weather"]},             # earlier revision
    {"revision": 2, "denied": ["get_time"]},                # same revision, other content
])
async def test_a_restriction_cannot_regress_or_reuse_a_revision(tmp_path, restriction):
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(_request(restriction={"revision": 2, "denied": ["get_weather"]}))
    opened = len(backend.opened)
    with pytest.raises(InvalidTransition, match="tool restriction"):
        await service.provision(_refresh(restriction, "refresh-1"))
    assert len(backend.opened) == opened  # refused before anything was minted


async def test_the_ledger_refuses_a_regression_the_service_did_not_see(tmp_path):
    """The store checks every row of the generation, not only the current one."""

    from eidolon.channel_provider.store import REFRESH

    clock = [1_700_000_000_000]
    service, store, _ = _service(tmp_path, clock)
    await service.provision(_request(restriction={"revision": 2, "denied": ["get_weather"]}))
    current = store.current_channel(_request().device_ref)
    older = replace(
        current, operation_id="refresh-older", operation_kind=REFRESH,
        request_fingerprint="refresh-older",
        response_json=json.dumps({**json.loads(current.response_json),
                                  "tool_restriction": {"revision": 1, "denied": []}}),
    )
    with pytest.raises(InvalidTransition, match="tool restriction"):
        store.create_provision(older, now_ms=clock[0])
