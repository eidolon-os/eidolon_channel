from __future__ import annotations

import json
from uuid import uuid4

import pytest

from eidolon.channel_provider.device_capabilities import (
    DeviceCommandRequest,
    capabilities_from_manifest,
)
from eidolon.channel_provider.contracts import (
    CapabilityNotDeclared,
    CurrentRequest,
    BackendUnavailable,
    IdempotencyConflict,
    Forbidden,
    InvalidTransition,
    ProvisionRequest,
    UnknownChannel,
)
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.ports import ControlReceipt
from eidolon.channel_provider.service import (
    DEVICE_COMMAND_RETRY_WINDOW,
    ChannelProviderService,
)
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import FakeAdapter, provision_payload


def _manifest(*actions: dict) -> dict:
    return {
        "schema_version": 1,
        "title": "Eidolon Box",
        "properties": [],
        "actions": list(actions),
        "events": [],
        "media": [
            {"kind": "audio", "direction": "bidirectional", "codecs": ["audio/opus"]}
        ],
    }


def _volume_action() -> dict:
    return {
        "name": "volume.set",
        "version": 1,
        "input_schema": {
            "type": "object",
            "properties": {"volume": {"type": "number", "minimum": 0, "maximum": 100}},
            "required": ["volume"],
            "additionalProperties": False,
        },
        "output_schema": {"type": "object", "properties": {"volume": {"type": "number"}}},
        "idempotent": True,
    }


def test_manifest_actions_map_to_agent_capabilities() -> None:
    capabilities = capabilities_from_manifest(_manifest(_volume_action()))

    assert [(item.name, item.version) for item in capabilities] == [("volume.set", 1)]
    assert capabilities[0].description == "volume.set"
    assert capabilities[0].input_schema["required"] == ["volume"]


def test_manifest_without_actions_declares_nothing() -> None:
    assert capabilities_from_manifest({"schema_version": 1}) == ()


def test_malformed_action_is_refused_rather_than_dropped() -> None:
    with pytest.raises(ValueError):
        capabilities_from_manifest(_manifest({"version": 1}))


@pytest.fixture
def service(tmp_path):
    adapter = FakeAdapter(name="livekit")
    service = ChannelProviderService(
        store=ChannelProviderStore(tmp_path / "provider.sqlite3"),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    service.initialize()
    return service, adapter


@pytest.mark.asyncio
async def test_directory_lists_only_online_devices_with_declared_capabilities(service) -> None:
    service, adapter = service
    adapter.on_channel = True
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))

    directory = json.loads(await service.capability_directory())

    assert directory["operation"] == "channel.capabilities"
    assert len(directory["devices"]) == 1
    device = directory["devices"][0]
    assert device["capabilities"][0]["name"] == "volume.set"


@pytest.mark.asyncio
async def test_command_refuses_capability_the_device_did_not_declare(service) -> None:
    service, _adapter = service
    await service.provision(ProvisionRequest.parse(json.dumps(provision_payload()).encode()))

    with pytest.raises(CapabilityNotDeclared, match="no declaration for volume.set.v1"):
        await service.deliver_device_command(
            DeviceCommandRequest(
                owner_id="owner_1",
                companion_id="companion_1",
                idempotency_key="key-1",
                device_id=service._store.active_provisions()[0].device_id,
                op="volume.set",
                capability_version=1,
                payload={"volume": 10},
                ttl_ms=5_000,
            )
        )


@pytest.mark.asyncio
async def test_command_refuses_out_of_range_payload_before_delivery(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id

    with pytest.raises(InvalidTransition):
        await service.deliver_device_command(
            DeviceCommandRequest(
                owner_id="owner_1",
                companion_id="companion_1",
                idempotency_key="key-1",
                device_id=device_id,
                op="volume.set",
                capability_version=1,
                payload={"volume": 101},
                ttl_ms=5_000,
            )
        )
    assert adapter.delivered == []


@pytest.mark.asyncio
async def test_command_refuses_a_different_owner(service) -> None:
    service, _adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id

    with pytest.raises(Forbidden):
        await service.deliver_device_command(
            DeviceCommandRequest(
                owner_id="someone_else",
                companion_id="companion_1",
                idempotency_key="key-owner",
                device_id=device_id,
                op="volume.set",
                capability_version=1,
                payload={"volume": 10},
                ttl_ms=5_000,
            )
        )


@pytest.mark.asyncio
async def test_command_to_unknown_device_is_refused(service) -> None:
    service, _adapter = service

    with pytest.raises(UnknownChannel):
        await service.deliver_device_command(
            DeviceCommandRequest(
                owner_id="owner_1",
                companion_id="companion_1",
                idempotency_key="key-absent",
                device_id="absent",
                op="volume.set",
                capability_version=1,
                payload={"volume": 10},
                ttl_ms=5_000,
            )
        )


@pytest.mark.asyncio
async def test_command_delivers_a_bounded_envelope_to_the_device(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id

    result = await service.deliver_device_command(
        DeviceCommandRequest(
            owner_id="owner_1",
            companion_id="companion_1",
            idempotency_key=f"key-{uuid4().hex}",
            device_id=device_id,
            op="volume.set",
            capability_version=1,
            payload={"volume": 42},
            ttl_ms=5_000,
        )
    )

    assert result.status == "succeeded"
    assert result.op == "volume.set"
    handle, command, wait_for_terminal = adapter.delivered[0]
    assert wait_for_terminal is True
    assert command["op"] == "volume.set"
    assert command["capability_version"] == 1
    assert command["payload"] == {"volume": 42}
    assert command["dst"] == {"type": "device", "id": device_id}
    assert command["src"]["type"] == "channel"


@pytest.mark.asyncio
async def test_command_keeps_the_result_from_a_device_receipt(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id

    async def deliver(*_args, **_kwargs):
        return ControlReceipt(status="succeeded", result={"volume": 42})

    adapter.deliver_control = deliver
    result = await service.deliver_device_command(
        DeviceCommandRequest(
            owner_id="owner_1",
            companion_id="companion_1",
            idempotency_key=f"key-{uuid4().hex}",
            device_id=device_id,
            op="volume.set",
            capability_version=1,
            payload={"volume": 42},
            ttl_ms=5_000,
        )
    )

    assert result.status == "succeeded"
    assert result.result == {"volume": 42}


@pytest.mark.asyncio
async def test_unreachable_device_reports_failure_not_success(service) -> None:
    service, adapter = service
    adapter.delivery_error = BackendUnavailable("control receipt timed out")
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id

    result = await service.deliver_device_command(
        DeviceCommandRequest(
            owner_id="owner_1",
            companion_id="companion_1",
            idempotency_key=f"key-{uuid4().hex}",
            device_id=device_id,
            op="volume.set",
            capability_version=1,
            payload={"volume": 10},
            ttl_ms=5_000,
        )
    )

    assert result.status == "failed"
    assert result.message == "control receipt timed out"


@pytest.mark.asyncio
async def test_a_replayed_idempotency_key_does_not_command_the_device_twice(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id
    request = DeviceCommandRequest(
        owner_id="owner_1",
        companion_id="companion_1",
        idempotency_key="retry-me",
        device_id=device_id,
        op="volume.set",
        capability_version=1,
        payload={"volume": 30},
        ttl_ms=5_000,
    )

    first = await service.deliver_device_command(request)
    second = await service.deliver_device_command(request)

    assert first == second
    assert len(adapter.delivered) == 1


@pytest.mark.asyncio
async def test_reusing_an_idempotency_key_with_other_content_is_refused(service) -> None:
    service, _adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id
    base = dict(
        owner_id="owner_1",
        companion_id="companion_1",
        idempotency_key="same-key",
        device_id=device_id,
        op="volume.set",
        capability_version=1,
        ttl_ms=5_000,
    )

    await service.deliver_device_command(DeviceCommandRequest(payload={"volume": 30}, **base))
    with pytest.raises(IdempotencyConflict):
        await service.deliver_device_command(
            DeviceCommandRequest(payload={"volume": 90}, **base)
        )


@pytest.mark.asyncio
async def test_command_is_refused_while_the_device_serves_another_companion(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id
    service._device_companions[device_id] = "companion_actual"

    with pytest.raises(Forbidden, match="different Companion"):
        await service.deliver_device_command(
            DeviceCommandRequest(
                owner_id="owner_1",
                companion_id="companion_asked",
                idempotency_key="key-companion",
                device_id=device_id,
                op="volume.set",
                capability_version=1,
                payload={"volume": 10},
                ttl_ms=5_000,
            )
        )
    assert adapter.delivered == []


@pytest.mark.asyncio
async def test_the_idempotency_window_is_bounded(service) -> None:
    service, adapter = service
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = service._store.active_provisions()[0].device_id
    base = dict(
        owner_id="owner_1",
        companion_id="companion_1",
        device_id=device_id,
        op="volume.set",
        capability_version=1,
        ttl_ms=5_000,
    )

    for index in range(DEVICE_COMMAND_RETRY_WINDOW + 5):
        await service.deliver_device_command(
            DeviceCommandRequest(
                idempotency_key=f"key-{index}", payload={"volume": index % 100}, **base
            )
        )

    assert len(service._device_commands) == DEVICE_COMMAND_RETRY_WINDOW
    assert len(adapter.delivered) == DEVICE_COMMAND_RETRY_WINDOW + 5


def test_a_general_json_schema_is_reduced_to_the_agent_vocabulary() -> None:
    action = {
        "idempotent": True,
        "name": "expression.play",
        "version": 1,
        "input_schema": {
            "$defs": {
                "Plan": {
                    "title": "Plan",
                    "type": "object",
                    "properties": {"gesture": {"type": "string", "title": "Gesture"}},
                }
            },
            "additionalProperties": False,
            "properties": {
                "session_id": {"type": "string", "maxLength": 96, "title": "Session Id"},
                "plan": {"$ref": "#/$defs/Plan"},
            },
            "required": ["session_id", "plan"],
            "title": "PlayExpression",
            "type": "object",
        },
        "output_schema": {"title": "R", "type": "object", "properties": {}},
    }

    capabilities = capabilities_from_manifest({"actions": [action]})

    schema = capabilities[0].input_schema
    assert "title" not in schema and "$defs" not in schema
    assert schema["properties"]["session_id"] == {"type": "string", "maxLength": 96}
    assert schema["properties"]["plan"] == {"type": "object"}
    assert schema["required"] == ["session_id", "plan"]
    assert capabilities[0].result_schema == {"type": "object", "properties": {}}


def test_a_schema_with_an_unresolvable_required_is_still_readable() -> None:
    action = {
        "name": "volume.set",
        "version": 1,
        "input_schema": {
            "type": "object",
            "title": "Volume",
            "properties": {"volume": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["volume", "gone"],
        },
        "output_schema": {},
    }

    capabilities = capabilities_from_manifest({"actions": [action]})

    assert capabilities[0].input_schema["required"] == ["volume"]


async def test_a_restarted_provider_asks_to_be_told_the_declaration_again(
    tmp_path,
) -> None:
    """A binding this process cannot describe is not one it should keep silently.

    The declarations live in memory and are filled only when a binding is
    provisioned or refreshed, so a restarted Provider holds a live binding whose
    capabilities it has forgotten. Reporting ``refresh_required`` is how it gets
    the manifest again — otherwise every device stops answering to its own
    capabilities until the grant happens to lapse.
    """
    adapter = FakeAdapter(name="livekit")
    adapter.on_channel = True
    path = tmp_path / "provider.sqlite3"
    first = ChannelProviderService(
        store=ChannelProviderStore(path),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    first.initialize()
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await first.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_ref = first._store.active_provisions()[0].device_ref

    current = json.loads(
        await first.current(CurrentRequest.parse(json.dumps({
            "operation": "channel.current-device",
            "device_ref": device_ref.model_dump(mode="json"),
        }).encode()))
    )
    assert "refresh_required" not in current, "a Provider that knows the declaration asks for nothing"

    restarted = ChannelProviderService(
        store=ChannelProviderStore(path),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    restarted.initialize()

    current = json.loads(
        await restarted.current(CurrentRequest.parse(json.dumps({
            "operation": "channel.current-device",
            "device_ref": device_ref.model_dump(mode="json"),
        }).encode()))
    )

    assert current.get("refresh_required") is True


async def test_a_restart_rebuilds_the_declarations_it_lost(tmp_path) -> None:
    """The refresh a restart asks for must be the one the ledger accepts.

    Declarations are derived from the Manifest and live in memory, so a
    restarted Provider holds the binding and describes none of it. ``current``
    therefore asks for a refresh — but the ledger only accepts one that is
    expired, carries a changed Manifest, or reports stale runtime inputs, and a
    fresh credential is none of those. Refusing it finally leaves the device
    waiting on a request nobody will make again; refusing it retryably leaves it
    retrying forever.

    So the one precondition this process lost is a reason the ledger accepts,
    and the refresh rebuilds exactly what was missing.
    """
    adapter = FakeAdapter(name="livekit")
    adapter.on_channel = True
    path = tmp_path / "provider.sqlite3"
    first = ChannelProviderService(
        store=ChannelProviderStore(path),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    first.initialize()
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await first.provision(ProvisionRequest.parse(json.dumps(payload).encode()))
    device_id = first._store.active_provisions()[0].device_id

    restarted = ChannelProviderService(
        store=ChannelProviderStore(path),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    restarted.initialize()
    assert device_id not in restarted._device_capabilities, "a restart must have lost them"

    refresh_payload = dict(payload)
    refresh_payload["operation"] = "channel.refresh-device"
    refresh_payload["operation_id"] = "refresh-after-restart"
    await restarted.provision(ProvisionRequest.parse(json.dumps(refresh_payload).encode()))

    assert device_id in restarted._device_capabilities
    declared = [item.name for item in restarted._device_capabilities[device_id]]
    assert declared == ["volume.set"]


async def test_a_refresh_of_an_unexpired_binding_still_needs_a_reason(tmp_path) -> None:
    """The new reason does not make every refresh acceptable.

    A Provider that still holds the declarations has the only reason available,
    and it is not one: without expiry, a changed Manifest or stale runtime
    inputs, the ledger keeps refusing — so a caller cannot rotate credentials on
    a whim.
    """
    adapter = FakeAdapter(name="livekit")
    adapter.on_channel = True
    service = ChannelProviderService(
        store=ChannelProviderStore(tmp_path / "provider.sqlite3"),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
    )
    service.initialize()
    payload = provision_payload()
    payload["device"]["manifest"] = _manifest(_volume_action())
    await service.provision(ProvisionRequest.parse(json.dumps(payload).encode()))

    refresh_payload = dict(payload)
    refresh_payload["operation"] = "channel.refresh-device"
    refresh_payload["operation_id"] = "refresh-with-declarations-held"

    with pytest.raises(InvalidTransition, match="credential refresh requires expiry"):
        await service.provision(ProvisionRequest.parse(json.dumps(refresh_payload).encode()))
