"""Device settings travel the installed-contract path end to end.

Admission is the registry's, the session carries what the Owner's selection
allows, the Owner's own app reaches only what is marked ``owner_control``, and
what a device reports about its values is read against the Host's declarations.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from eidolon_sdk.biz.capabilities import installed_properties, installed_registry
from eidolon_sdk.biz.capabilities.device_settings import (
    AUDIO_VOLUME,
    AUDIO_VOLUME_ADJUST,
    AUDIO_VOLUME_SET,
    DISPLAY_BRIGHTNESS,
    DISPLAY_BRIGHTNESS_ADJUST,
    DISPLAY_BRIGHTNESS_SET,
)
from eidolon_sdk.biz.contracts import EVENT_TOPIC
from eidolon_sdk.biz.presentation import FACE_PROFILE, DeviceOutputPolicy, OutputSelection
from eidolon_sdk.biz.presentation.capability_rpc import (
    INVOKE,
    STATUS,
    CapabilityInvocation,
    ExecutionScope,
    ExecutionSnapshot,
)
from eidolon_sdk.biz.presentation.device_properties import PROPERTIES_READ, PropertyValues

from eidolon.channel_provider.adapters.livekit import adapter as adapter_mod
from eidolon.channel_provider.contracts import (
    BackendUnavailable,
    CapabilityNotOffered,
    ContractError,
    Forbidden,
    InvalidTransition,
    ProvisionRequest,
    UnknownChannel,
)
from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.ports import ChannelGrant
from eidolon.channel_provider.spec import derive_spec

from .helpers import FakeAdapter, encoded, provision_payload
from .test_livekit_adapter import _adapter
from .test_service import _service

SETTINGS = (AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST, DISPLAY_BRIGHTNESS_SET, DISPLAY_BRIGHTNESS_ADJUST)


def _manifest(*, actions=None, properties=None) -> dict[str, Any]:
    registry = installed_registry()
    installed = installed_properties()
    return {
        "schema_version": 1,
        "title": "Eidolon Box",
        "properties": [
            {"name": "expression.profile", "observable": False, "writable": False,
             "schema": {"type": "string", "const": FACE_PROFILE}},
            *(properties if properties is not None
              else [installed[AUDIO_VOLUME], installed[DISPLAY_BRIGHTNESS]]),
        ],
        "actions": actions if actions is not None else [
            registry.get(name).action() for name in SETTINGS
        ],
        "events": [],
        "media": [{"kind": "audio", "direction": "bidirectional", "codecs": ["audio/opus"]}],
    }


def _request(*, speech=True, expression=False, policy=True, manifest=None) -> ProvisionRequest:
    overrides: dict[str, Any] = {"manifest": manifest or _manifest()}
    if policy:
        overrides["output_policy"] = DeviceOutputPolicy(
            revision=7, allowed=OutputSelection(speech=speech, expression=expression),
        ).model_dump(mode="json")
    return ProvisionRequest.parse(encoded(provision_payload(**overrides)))


def _spec(request: ProvisionRequest):
    return derive_spec(
        request.device, device_instance_id=request.device_ref.device_instance_id,
        agent_name="eidolon",
    )


# -- admission ---------------------------------------------------------------


def test_a_session_carries_only_the_settings_whose_output_the_owner_selected():
    speech_only = _spec(_request(speech=True, expression=False))
    assert speech_only.device_capabilities == (AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST)
    both = _spec(_request(speech=True, expression=True))
    assert set(both.device_capabilities) == set(SETTINGS)


def test_the_owners_app_reaches_the_setters_whatever_the_session_selection():
    selected = _spec(_request(speech=True, expression=False))
    assert selected.owner_capabilities == (AUDIO_VOLUME_SET, DISPLAY_BRIGHTNESS_SET)
    assert selected.observable_properties == (AUDIO_VOLUME, DISPLAY_BRIGHTNESS)


def test_a_device_cannot_widen_an_installed_contract_or_property():
    registry = installed_registry()
    widened = registry.get(AUDIO_VOLUME_SET).action()
    widened["input_schema"] = {"type": "object"}
    installed = installed_properties()
    selected = _spec(_request(manifest=_manifest(
        actions=[widened, registry.get(DISPLAY_BRIGHTNESS_SET).action()],
        properties=[{**installed[AUDIO_VOLUME], "schema": {"type": "integer"}}],
    ), expression=True))
    assert AUDIO_VOLUME_SET not in selected.device_capabilities
    assert AUDIO_VOLUME_SET not in selected.owner_capabilities
    assert selected.owner_capabilities == (DISPLAY_BRIGHTNESS_SET,)
    assert selected.observable_properties == ()


def test_a_channel_with_no_owner_policy_offers_the_owner_nothing():
    manifest = _manifest()
    manifest["properties"] = manifest["properties"][1:]  # no expression: no policy demanded
    selected = _spec(_request(policy=False, manifest=manifest))
    assert selected.owner_capabilities == ()
    assert selected.device_capabilities == ()


async def test_the_grant_handle_carries_what_the_owner_may_invoke_and_read():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    assert grant.handle["owner_control"] == {
        "policy_revision": 7, "capabilities": [AUDIO_VOLUME_SET, DISPLAY_BRIGHTNESS_SET],
    }
    assert grant.handle["observable_properties"] == [AUDIO_VOLUME, DISPLAY_BRIGHTNESS]
    assert grant.handle["output_template"]["device_capabilities"] == [
        AUDIO_VOLUME_SET, AUDIO_VOLUME_ADJUST,
    ]


# -- standing invocation and property projection on the adapter -------------


class _Participant:
    def __init__(self, answers):
        self.answers = answers
        self.calls: list[tuple[str, str, str]] = []

    async def perform_rpc(self, *, destination_identity, method, payload, response_timeout=None):
        self.calls.append((destination_identity, method, payload))
        answer = self.answers[method]
        if isinstance(answer, BaseException):
            raise answer
        return answer(payload) if callable(answer) else answer


def _listening(adapter, handle, participant, *, present=True):
    connection = SimpleNamespace(
        local_participant=participant,
        remote_participants={handle["device"]: object()} if present else {},
    )
    watch = adapter_mod._Listening(device=handle["device"], sink=None, connection=connection)
    watch.input_handle = handle
    adapter._listeners[handle["room"]] = watch
    return watch


def _snapshot(payload: str, state: str, **extra) -> str:
    execution_id = json.loads(payload)["execution_id"]
    return ExecutionSnapshot(
        execution_id=execution_id, state=state, sequence=1 if state != "accepted" else 0,
        **extra,
    ).model_dump_json()


async def test_a_standing_invocation_is_bound_to_the_room_and_waits_for_the_outcome():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    participant = _Participant({
        INVOKE: lambda p: _snapshot(p, "accepted"),
        STATUS: lambda p: _snapshot(p, "completed", result={"volume": 30},
                                    completion_basis="configuration_applied"),
    })
    _listening(adapter, grant.handle, participant)

    snapshot = await adapter.invoke_standing(
        grant.handle, capability=AUDIO_VOLUME_SET, arguments={"volume": 30},
        execution_id="owner:abc", timeout_s=5,
    )

    assert (snapshot.state, snapshot.result) == ("completed", {"volume": 30})
    (device, method, payload), (_, status_method, status_payload) = participant.calls
    assert (device, method, status_method) == (grant.handle["device"], INVOKE, STATUS)
    invocation = CapabilityInvocation.model_validate_json(payload)
    assert invocation.session_id == grant.handle["room"]
    assert invocation.policy_revision == 7
    assert (invocation.capability, invocation.arguments) == (AUDIO_VOLUME_SET, {"volume": 30})
    assert ExecutionScope.model_validate_json(status_payload).execution_id == "owner:abc"
    adapter._listeners.clear()


async def test_a_standing_invocation_refuses_what_the_channel_does_not_offer():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    participant = _Participant({})
    _listening(adapter, grant.handle, participant)
    with pytest.raises(InvalidTransition, match="no such capability"):
        await adapter.invoke_standing(
            grant.handle, capability=AUDIO_VOLUME_ADJUST, arguments={"change": 10},
            execution_id="owner:abc", timeout_s=5,
        )
    assert participant.calls == []
    adapter._listeners.clear()


async def test_a_standing_invocation_to_an_absent_device_says_so_rather_than_waiting():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    _listening(adapter, grant.handle, _Participant({}), present=False)
    with pytest.raises(BackendUnavailable, match="not on its channel"):
        await adapter.invoke_standing(
            grant.handle, capability=AUDIO_VOLUME_SET, arguments={"volume": 30},
            execution_id="owner:abc", timeout_s=5,
        )
    adapter._listeners.clear()


def _event_packet(device: str, name: str, value: Any, *, event_id="evt-1"):
    body = {
        "schema_v": 1, "kind": "event", "event_id": event_id, "flow_id": "flow-1",
        "causation_id": "", "type": "property.changed",
        "source": {"device_id": device, "component": "settings"},
        "occurred_at_ms": 1_800_000_000_000, "expires_at_ms": 1_800_000_001_000,
        "payload": {"name": name, "value": value},
    }
    return SimpleNamespace(topic=EVENT_TOPIC, participant=SimpleNamespace(identity=device),
                           data=json.dumps(body).encode())


async def test_properties_come_from_what_the_device_reported_and_a_gap_is_asked_once():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    handle = grant.handle
    participant = _Participant({
        PROPERTIES_READ: PropertyValues(
            values={AUDIO_VOLUME: 40, DISPLAY_BRIGHTNESS: 70}).model_dump_json(),
    })
    watch = _listening(adapter, handle, participant)
    device = handle["device"]

    # A reported change is kept; an undeclared or out-of-range one is not.
    adapter._property_change(_event_packet(device, AUDIO_VOLUME, 55), room=handle["room"], watch=watch)
    adapter._property_change(_event_packet(device, AUDIO_VOLUME, 101), room=handle["room"], watch=watch)
    adapter._property_change(_event_packet(device, "battery.level", 3), room=handle["room"], watch=watch)

    # Brightness was never reported, so the device is asked; the reported
    # volume is newer than anything the read could say and stays.
    assert await adapter.device_properties(handle) == {AUDIO_VOLUME: 55, DISPLAY_BRIGHTNESS: 70}
    assert [call[1] for call in participant.calls] == [PROPERTIES_READ]

    # Complete now: no second read.
    assert await adapter.device_properties(handle) == {AUDIO_VOLUME: 55, DISPLAY_BRIGHTNESS: 70}
    assert len(participant.calls) == 1
    adapter._listeners.clear()


async def test_an_unreachable_device_reports_nothing_rather_than_a_default():
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    _listening(adapter, grant.handle, _Participant({PROPERTIES_READ: RuntimeError("gone")}))
    assert await adapter.device_properties(grant.handle) == {}
    adapter._listeners.clear()


async def test_what_a_device_said_is_forgotten_when_it_leaves(monkeypatch):
    from .test_livekit_adapter import FakeRoom, _unused_sink

    adapter, _ = _adapter()
    grant = await adapter.open(_spec(_request()), issued_at_ms=1_000)
    FakeRoom.instances.clear()
    monkeypatch.setattr("eidolon.channel_provider.adapters.livekit.adapter.rtc.Room", FakeRoom)
    await adapter.accept_requests(grant.handle, sink=_unused_sink)
    room = FakeRoom.instances[-1]
    device = grant.handle["device"]

    room.handlers["data_received"](_event_packet(device, AUDIO_VOLUME, 33))
    assert adapter._device_properties[grant.handle["room"]] == {AUDIO_VOLUME: 33}
    room.handlers["participant_disconnected"](SimpleNamespace(identity=device))
    assert grant.handle["room"] not in adapter._device_properties
    await adapter.stop_accepting(grant.handle)


# -- the Owner-scoped service surface ----------------------------------------


class _SettingsAdapter(FakeAdapter):
    """A transport whose handle says what the real one says about Owner control."""

    def __init__(self, **kwargs):
        super().__init__(name="livekit", **kwargs)
        self.invoked: list[dict[str, Any]] = []
        self.snapshot_state = "completed"

    async def open(self, spec, *, issued_at_ms, observed_host_address=""):
        grant = await super().open(spec, issued_at_ms=issued_at_ms,
                                   observed_host_address=observed_host_address)
        handle = dict(grant.handle)
        handle["observable_properties"] = list(spec.observable_properties)
        if spec.owner_capabilities:
            handle["owner_control"] = {"policy_revision": spec.output_policy.revision,
                                       "capabilities": list(spec.owner_capabilities)}
        return ChannelGrant(binding_format=grant.binding_format, payload=grant.payload,
                            expires_at_ms=grant.expires_at_ms, handle=handle)

    async def invoke_standing(self, handle, *, capability, arguments, execution_id, timeout_s):
        self.invoked.append({"capability": capability, "arguments": arguments,
                             "execution_id": execution_id, "timeout_s": timeout_s})
        return ExecutionSnapshot(execution_id=execution_id, state=self.snapshot_state,
                                 sequence=1, result={"volume": arguments.get("volume", 0)})

    async def device_properties(self, handle):
        return {AUDIO_VOLUME: 25}


async def _provisioned(tmp_path):
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock, _SettingsAdapter())
    request = _request()
    await service.provision(request)
    return service, backend, request.device_ref.device_instance_id


async def test_the_owner_invokes_a_setter_and_hears_the_devices_outcome(tmp_path):
    service, backend, device = await _provisioned(tmp_path)
    answer = json.loads(await service.invoke_owner_capability(
        owner_id="owner_1", device_id=device, capability=AUDIO_VOLUME_SET,
        arguments={"volume": 30}, request_id="slider-1",
    ))
    assert answer["operation"] == "channel.device-capability"
    assert answer["execution"]["state"] == "completed"
    assert answer["execution"]["result"] == {"volume": 30}
    # The request id names the execution, so a retry is the same execution.
    await service.invoke_owner_capability(
        owner_id="owner_1", device_id=device, capability=AUDIO_VOLUME_SET,
        arguments={"volume": 30}, request_id="slider-1",
    )
    first, second = backend.invoked
    assert first["execution_id"] == second["execution_id"]
    assert first["execution_id"].startswith("owner:")
    assert first["timeout_s"] == installed_registry().get(AUDIO_VOLUME_SET).timeout_s


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        ({"owner_id": "owner_2"}, Forbidden),
        ({"device_id": "unknown"}, UnknownChannel),
        ({"capability": AUDIO_VOLUME_ADJUST, "arguments": {"change": 5}}, CapabilityNotOffered),
        ({"capability": "camera.capture", "arguments": {}}, CapabilityNotOffered),
        ({"arguments": {"volume": 101}}, ContractError),
    ],
)
async def test_the_owner_surface_refuses_what_it_does_not_offer(tmp_path, changes, refusal):
    service, backend, device = await _provisioned(tmp_path)
    call = {"owner_id": "owner_1", "device_id": device, "capability": AUDIO_VOLUME_SET,
            "arguments": {"volume": 30}, "request_id": "r-1", **changes}
    with pytest.raises(refusal):
        await service.invoke_owner_capability(**call)
    assert backend.invoked == []


async def test_properties_are_read_for_the_owner_only(tmp_path):
    service, _backend, device = await _provisioned(tmp_path)
    answer = json.loads(await service.device_properties(owner_id="owner_1", device_id=device))
    assert answer == {"operation": "channel.device-properties", "device_id": device,
                      "values": {AUDIO_VOLUME: 25}}
    with pytest.raises(Forbidden):
        await service.device_properties(owner_id="owner_2", device_id=device)


async def test_the_http_routes_carry_the_owner_scope(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    service, _backend, device = await _provisioned(tmp_path)
    token = "provider-token-" + "x" * 32
    client = TestClient(TestServer(create_app(service=service, bearer_token=token)))
    await client.start_server()
    auth = {"Authorization": f"Bearer {token}"}

    response = await client.post(
        f"/v1/device-channels/{device}/capabilities/{AUDIO_VOLUME_SET}",
        json={"owner_id": "owner_1", "request_id": "slider-2", "arguments": {"volume": 12}},
        headers=auth,
    )
    assert response.status == 200
    assert (await response.json())["execution"]["result"] == {"volume": 12}

    response = await client.get(
        f"/v1/device-channels/{device}/properties", params={"owner_id": "owner_1"}, headers=auth
    )
    assert (await response.json())["values"] == {AUDIO_VOLUME: 25}

    refused = await client.post(
        f"/v1/device-channels/{device}/capabilities/{AUDIO_VOLUME_SET}",
        json={"owner_id": "owner_1", "request_id": "slider-3", "arguments": {"volume": 300}},
        headers=auth,
    )
    assert refused.status == 422
    unauthenticated = await client.get(
        f"/v1/device-channels/{device}/properties", params={"owner_id": "owner_1"}
    )
    assert unauthenticated.status == 401
    missing_owner = await client.get(f"/v1/device-channels/{device}/properties", headers=auth)
    assert missing_owner.status == 422
    await client.close()
