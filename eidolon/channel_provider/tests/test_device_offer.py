"""What a device's channel offers is read from the grant, never recomputed.

The Owner's app shows which tools a Companion can use on a device, and why not
(工具总览与开关设计 §5.4). Two facts answer it side by side: what the Manifest
declared, and what the Owner's output policy lets a session use of that. Both
are recorded on the channel's handle when the Provider grants it, so "the page
says it works" and "the session was given it" are one fact.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from eidolon_sdk.biz.capabilities import installed_registry
from eidolon_sdk.biz.capabilities.device_settings import AUDIO_VOLUME_SET
from eidolon_sdk.biz.presentation import DeviceOutputPolicy, OutputSelection, SessionOutputPlan
from eidolon_sdk.biz.tools import DeviceToolOffer
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

from eidolon.channel_provider.access import Role
from eidolon.channel_provider.contracts import (
    CurrentRequest,
    Forbidden,
    ProvisionRequest,
    UnknownChannel,
)
from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.spec import derive_spec

from .helpers import ROLE_CREDENTIALS, FakeAdapter, bearer, current_payload, encoded, provision_payload
from .test_livekit_adapter import _adapter
from .test_service import _service

HEAD = "stackchan.head.v1"
DECLARED = ("motion.nod", "motion.stop", AUDIO_VOLUME_SET)


def _request(*, motion: bool | None, operation_id: str = "enrollment-1") -> ProvisionRequest:
    """A head that nods and a speaker; ``motion=None`` is an Owner who has not decided."""

    overrides: dict[str, Any] = {}
    if motion is not None:
        overrides["output_policy"] = DeviceOutputPolicy(
            revision=3, allowed=OutputSelection(speech=True, motion=motion),
        ).model_dump(mode="json")
    payload = provision_payload(**overrides)
    payload["operation_id"] = operation_id
    manifest = payload["device"]["manifest"]
    manifest["properties"] += [
        {"name": "output.motion", "observable": False, "writable": False,
         "schema": {"type": "boolean", "const": True}},
        {"name": "motion.profile", "observable": False, "writable": False,
         "schema": {"type": "string", "const": HEAD}},
    ]
    registry = installed_registry()
    manifest["actions"] = [registry.get(name).action() for name in DECLARED]
    return ProvisionRequest.parse(encoded(payload))


def _spec(request: ProvisionRequest):
    return derive_spec(
        request.device, device_instance_id=request.device_ref.device_instance_id,
        agent_name="eidolon",
    )


# -- what the spec keeps -----------------------------------------------------


@pytest.mark.parametrize("motion", [False, True])
def test_the_declared_head_survives_an_owner_who_did_not_select_motion(motion):
    spec = _spec(_request(motion=motion))
    assert spec.declared_capabilities == DECLARED
    assert spec.declared_motion_profile == HEAD
    # A session still gets the profile and the gestures only with motion selected.
    assert spec.motion_profile == (HEAD if motion else None)
    assert spec.motion_capabilities == (("motion.nod", "motion.stop") if motion else ())
    assert spec.device_capabilities == (AUDIO_VOLUME_SET,)


# -- what the grant records ---------------------------------------------------


async def test_the_handle_records_the_declaration_outside_what_devices_are_sent():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request(motion=False)), issued_at_ms=1_000)
    assert grant.handle["declared"] == {"capabilities": list(DECLARED), "motion_profile": HEAD}
    template = grant.handle["output_template"]
    assert set(template) == {
        "policy_revision", "inputs", "outputs", "expression_profile", "motion_profile",
        "device_capabilities",
    }
    # Devices parse this strictly; it still is exactly a session's plan.
    SessionOutputPlan(session_id="session-1", **template)

    undecided = await adapter.open(_spec(_request(motion=None)), issued_at_ms=1_000)
    assert undecided.handle["declared"] == grant.handle["declared"]
    assert "output_template" not in undecided.handle


async def test_a_handle_minted_before_the_declaration_was_kept_is_not_current():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request(motion=False)), issued_at_ms=1_000)
    assert adapter.binding_current(grant.handle)
    older = {key: value for key, value in grant.handle.items() if key != "declared"}
    assert not adapter.binding_current(older)


# -- the Owner-scoped read ----------------------------------------------------


class _GrantingAdapter(FakeAdapter):
    """The fake transport, holding the handle the LiveKit adapter mints for the spec.

    ``legacy`` plays a Provider deployed before handles recorded the declaration.
    """

    def __init__(self):
        super().__init__(name="livekit")
        self.livekit, _ = _adapter()
        self.legacy = False

    async def open(self, spec, *, issued_at_ms, observed_host_address=""):
        grant = await super().open(spec, issued_at_ms=issued_at_ms,
                                   observed_host_address=observed_host_address)
        minted = await self.livekit.open(spec, issued_at_ms=issued_at_ms)
        handle = {**minted.handle, **grant.handle}
        if self.legacy:
            handle.pop("declared")
        return replace(grant, handle=handle)

    def binding_current(self, handle: dict[str, Any]) -> bool:
        return self.livekit.binding_current(handle)


async def _provisioned(tmp_path, *, motion: bool | None, backend=None):
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock, backend or _GrantingAdapter())
    request = _request(motion=motion)
    await service.provision(request)
    return service, backend, request


async def _client(service) -> TestClient:
    client = TestClient(TestServer(create_app(service=service, credentials=ROLE_CREDENTIALS)))
    await client.start_server()
    return client


def _offer_path(device: str) -> str:
    return f"/v1/device-channels/{device}/offer"


async def test_the_owner_reads_what_the_grant_offers(tmp_path):
    service, _backend, request = await _provisioned(tmp_path, motion=False)
    device = request.device_ref.device_instance_id
    client = await _client(service)
    try:
        response = await client.get(
            _offer_path(device), params={"owner_id": "owner_1"}, headers=bearer(Role.MANAGEMENT),
        )
        assert response.status == 200
        text = await response.text()
    finally:
        await client.close()
    assert json.loads(text) == {
        "operation": "channel.device-offer",
        "device_id": device,
        "declared_capabilities": list(DECLARED),
        "declared_motion_profile": HEAD,
        "policy_revision": 3,
        "outputs": {"speech": True, "dialogue_text": False, "expression": False,
                    "audio_cue": False, "motion": False},
        "session_capabilities": [AUDIO_VOLUME_SET],
    }
    assert DeviceToolOffer.model_validate_json(text).device_id == device


async def test_selected_motion_reaches_the_session_capabilities(tmp_path):
    service, _backend, request = await _provisioned(tmp_path, motion=True)
    offer = DeviceToolOffer.model_validate_json(await service.device_offer(
        owner_id="owner_1", device_id=request.device_ref.device_instance_id))
    assert offer.session_capabilities == ("motion.nod", "motion.stop", AUDIO_VOLUME_SET)
    assert offer.outputs["motion"] is True


async def test_a_device_with_no_output_policy_offers_a_session_nothing(tmp_path):
    service, _backend, request = await _provisioned(tmp_path, motion=None)
    device = request.device_ref.device_instance_id
    answer = json.loads(await service.device_offer(owner_id="owner_1", device_id=device))
    assert answer == {
        "operation": "channel.device-offer",
        "device_id": device,
        "declared_capabilities": list(DECLARED),
        "declared_motion_profile": HEAD,
        "policy_revision": None,
        "outputs": None,
        "session_capabilities": [],
    }


async def test_the_offer_is_the_owners_alone_and_admins_to_read(tmp_path):
    service, _backend, request = await _provisioned(tmp_path, motion=False)
    device = request.device_ref.device_instance_id
    with pytest.raises(Forbidden):
        await service.device_offer(owner_id="owner_2", device_id=device)
    client = await _client(service)
    try:
        management = bearer(Role.MANAGEMENT)
        other = await client.get(
            _offer_path(device), params={"owner_id": "owner_2"}, headers=management)
        assert other.status == 403
        assert (await other.json())["code"] == "FORBIDDEN"
        unknown = await client.get(
            _offer_path(named_device_instance_id("unknown")),
            params={"owner_id": "owner_1"}, headers=management)
        assert unknown.status == 404
        assert (await unknown.json())["code"] == "NOT_FOUND"
        missing_owner = await client.get(_offer_path(device), headers=management)
        assert missing_owner.status == 422
        for role in (Role.HUB, Role.WORKER):
            refused = await client.get(
                _offer_path(device), params={"owner_id": "owner_1"}, headers=bearer(role))
            assert refused.status == 403, role
        unauthenticated = await client.get(_offer_path(device), params={"owner_id": "owner_1"})
        assert unauthenticated.status == 401
    finally:
        await client.close()


async def test_a_transport_that_records_no_declaration_is_not_ready(tmp_path):
    service, _backend, request = await _provisioned(tmp_path, motion=False, backend=FakeAdapter(
        name="livekit"))
    device = request.device_ref.device_instance_id
    with pytest.raises(UnknownChannel, match="declared"):
        await service.device_offer(owner_id="owner_1", device_id=device)
    client = await _client(service)
    try:
        response = await client.get(
            _offer_path(device), params={"owner_id": "owner_1"}, headers=bearer(Role.MANAGEMENT))
        assert response.status == 404
        assert (await response.json())["code"] == "NOT_FOUND"
    finally:
        await client.close()


async def test_a_grant_from_before_the_declaration_is_refreshed_once_then_offers(tmp_path):
    """After deploy every older binding is renewed through the refresh Hub already runs."""

    backend = _GrantingAdapter()
    backend.legacy = True
    service, _backend, request = await _provisioned(tmp_path, motion=False, backend=backend)
    device = request.device_ref.device_instance_id
    backend.legacy = False  # the new Provider is deployed

    current = CurrentRequest.parse(encoded(current_payload()))
    assert json.loads(await service.current(current))["refresh_required"] is True
    with pytest.raises(UnknownChannel):
        await service.device_offer(owner_id="owner_1", device_id=device)

    await service.provision(replace(_request(motion=False, operation_id="refresh-1"),
                                    operation="channel.refresh-device"))
    assert "refresh_required" not in json.loads(await service.current(current))
    offer = DeviceToolOffer.model_validate_json(
        await service.device_offer(owner_id="owner_1", device_id=device))
    assert offer.declared_capabilities == DECLARED
