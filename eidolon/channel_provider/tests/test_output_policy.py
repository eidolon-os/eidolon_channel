import json
from dataclasses import replace
import pytest
from eidolon_sdk.biz.contracts import SESSION_INTENT_USER_INITIATED
from eidolon_sdk.biz.presentation import DeviceOutputPolicy, OutputSelection, FACE_PROFILE
from eidolon.channel_provider.contracts import (
    ProvisionRequest,
    ContractError,
    InvalidTransition,
    ChannelNotServable,
)
from eidolon.channel_provider.spec import derive_spec, MediaFlow
from .helpers import provision_payload, encoded
from .test_service import _service
from .test_livekit_adapter import _adapter, FakeDispatch


def request(*, revision=1, speech=False):
    payload = provision_payload(
        output_policy=DeviceOutputPolicy(
            revision=revision, allowed=OutputSelection(expression=True, speech=speech)
        ).model_dump(mode="json")
    )
    payload["device"]["manifest"]["properties"] = [
        {
            "name": "expression.profile",
            "observable": False,
            "writable": False,
            "schema": {"type": "string", "const": FACE_PROFILE},
        },
        {
            "name": "output.dialogue_text",
            "observable": False,
            "writable": False,
            "schema": {"type": "boolean", "const": True},
        },
    ]
    return ProvisionRequest.parse(encoded(payload))


def spec(req):
    return derive_spec(
        req.device, device_instance_id=req.device_ref.device_instance_id, agent_name="eidolon"
    )


def test_silent_negotiation_preserves_microphone_and_turn_mode():
    selected = spec(request())
    assert selected.audio == MediaFlow.PUBLISH
    assert selected.selected_outputs == OutputSelection(expression=True)
    assert selected.serving.interaction_mode == "half_duplex"
    assert spec(request(speech=True)).audio == MediaFlow.DUPLEX


def test_new_face_without_owner_policy_cannot_fall_back_to_speech():
    req = request()
    with pytest.raises(ContractError, match="OUTPUT_POLICY_REQUIRED"):
        spec(replace(req, device=replace(req.device, output_policy=None)))


async def test_policy_refresh_without_manifest_change_is_durable_and_rejects_rollback(tmp_path):
    service, store, backend = _service(tmp_path, [1700000000000])
    old = request(speech=True)
    await service.provision(old)
    changed = replace(
        request(revision=2), operation="channel.refresh-device", operation_id="policy-2"
    )
    result = json.loads(await service.provision(changed))
    assert result["output_policy"]["revision"] == 2
    assert store.active_device(old.device_ref).output_policy.allowed.speech is False
    before = len(backend.opened)
    stale = replace(old, operation="channel.refresh-device", operation_id="stale-policy")
    with pytest.raises(InvalidTransition, match="output policy"):
        await service.provision(stale)
    assert len(backend.opened) == before  # no network effects before rejection


async def test_dispatch_binds_output_plan_and_changed_policy_withdraws_legacy_voice():
    adapter, client = _adapter()
    grant = await adapter.open(spec(request()), issued_at_ms=1700000000000)
    room = grant.handle["room"]
    client.agent_dispatch._room(room).append(
        FakeDispatch("old-voice", "eidolon", metadata='{"conversation_id":"old"}')
    )
    await adapter._reconcile_output_dispatches(grant.handle)
    assert (room, "old-voice") in client.agent_dispatch.deleted
    await adapter.open_session(
        grant.handle, "session-1", session_intent=SESSION_INTENT_USER_INITIATED
    )
    metadata = json.loads(client.agent_dispatch.created[-1][2])
    assert metadata["output_plan"]["session_id"] == "session-1"
    assert metadata["output_plan"]["outputs"] == OutputSelection(expression=True).model_dump()
    # An unchanged policy does not restart a good session.
    before = len(client.agent_dispatch.deleted)
    await adapter._reconcile_output_dispatches(grant.handle)
    assert len(client.agent_dispatch.deleted) == before
    await adapter.shutdown()


async def test_all_outputs_denied_still_replaces_policy_but_cannot_start_a_session():
    adapter, client = _adapter()
    req = request()
    req = replace(
        req,
        device=replace(
            req.device, output_policy=DeviceOutputPolicy(revision=2, allowed=OutputSelection())
        ),
    )
    grant = await adapter.open(spec(req), issued_at_ms=1700000000000)
    with pytest.raises(ChannelNotServable, match="NO_RESPONSE_OUTPUT"):
        await adapter.open_session(
            grant.handle, "session-1", session_intent=SESSION_INTENT_USER_INITIATED
        )
    await adapter.shutdown()
