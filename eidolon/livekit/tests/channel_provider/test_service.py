from __future__ import annotations

import base64
import json

import pytest

from eidolon.livekit.channel_provider.contracts import (
    IdempotencyConflict,
    ProvisionRequest,
    RevokeRequest,
    canonical_json,
)
from eidolon.livekit.channel_provider.livekit_backend import LiveKitBinding
from eidolon.livekit.channel_provider.service import ChannelProviderService
from eidolon.livekit.channel_provider.store import ChannelProviderStore

from .helpers import encoded, livekit_config, provision_payload, revoke_payload


class FakeBackend:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = ttl_seconds
        self.ensure_calls: list[tuple[str, str]] = []
        self.revoke_calls: list[tuple[str, str]] = []
        self.binding_calls = 0
        self.closed = False

    async def ensure_rooms(self, active_room: str, control_room: str) -> None:
        self.ensure_calls.append((active_room, control_room))

    async def revoke_rooms(self, active_room: str, control_room: str) -> None:
        self.revoke_calls.append((active_room, control_room))

    def build_binding(
        self,
        *,
        active_room: str,
        control_room: str,
        device_id: str,
        owner_id: str,
        issued_at_ms: int,
    ) -> LiveKitBinding:
        self.binding_calls += 1
        payload = canonical_json(
            {
                "schema_version": 1,
                "active": {
                    "server_url": "wss://livekit.example.test",
                    "token": f"active-token-{self.binding_calls}",
                    "identity": device_id,
                    "room_name": active_room,
                },
                "control": {
                    "server_url": "wss://livekit.example.test",
                    "token": f"control-token-{self.binding_calls}",
                    "identity": device_id,
                    "room_name": control_room,
                },
                "audio": {"sample_rate": 16000, "channels": 1},
                "test_owner": owner_id,
            }
        ).encode()
        return LiveKitBinding(
            payload=payload,
            expires_at_ms=issued_at_ms + self.ttl_seconds * 1000,
        )

    async def close(self) -> None:
        self.closed = True


def _service(tmp_path, clock: list[int], backend: FakeBackend | None = None):
    config = livekit_config()
    resolved_backend = backend or FakeBackend(config.grant_ttl_seconds)
    store = ChannelProviderStore(tmp_path / "provider.sqlite3")
    service = ChannelProviderService(
        store=store,
        backend=resolved_backend,
        livekit=config,
        now_ms=lambda: clock[0],
    )
    service.initialize()
    return service, store, resolved_backend


def _binding(response: str) -> dict:
    channel = json.loads(response)["channels"][0]
    return json.loads(base64.b64decode(channel["opaque_binding"], validate=True))


async def test_provision_is_exactly_idempotent_across_restart(tmp_path) -> None:
    clock = [1_700_000_000_000]
    request = ProvisionRequest.parse(encoded(provision_payload()))
    service, _store, backend = _service(tmp_path, clock)

    first = await service.provision(request)
    second = await service.provision(request)
    restarted, _store2, restarted_backend = _service(tmp_path, clock)
    third = await restarted.provision(request)

    assert first == second == third
    assert backend.binding_calls == 1
    assert len(backend.ensure_calls) == 1
    assert restarted_backend.binding_calls == 0
    response = json.loads(first)
    assert response["operation"] == "channel.provisioned-device"
    assert response["operation_id"] == "enrollment-1"
    assert response["device_id"] == "device-1"
    assert response["channels"][0]["binding_format"] == (
        "application/vnd.eidolon.livekit-device+json;v=1"
    )
    binding = _binding(first)
    assert binding["active"]["room_name"].endswith("-voice")
    assert binding["control"]["room_name"].endswith("-control")


async def test_provision_refreshes_only_near_expiry_with_stable_resources(tmp_path) -> None:
    clock = [1_700_000_000_000]
    request = ProvisionRequest.parse(encoded(provision_payload()))
    service, _store, backend = _service(tmp_path, clock)
    first = await service.provision(request)

    clock[0] += (1800 - 119) * 1000
    refreshed = await service.provision(request)

    assert refreshed != first
    assert backend.binding_calls == 2
    assert len(backend.ensure_calls) == 2
    assert _binding(refreshed)["active"]["room_name"] == _binding(first)["active"]["room_name"]
    assert json.loads(refreshed)["channels"][0]["channel_id"] == (
        json.loads(first)["channels"][0]["channel_id"]
    )


async def test_provision_rejects_operation_or_device_authority_reuse(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, _backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    changed = provision_payload(owner_id="owner-2")
    with pytest.raises(IdempotencyConflict):
        await service.provision(ProvisionRequest.parse(encoded(changed)))

    second_operation = provision_payload()
    second_operation["operation_id"] = "enrollment-2"
    with pytest.raises(IdempotencyConflict):
        await service.provision(ProvisionRequest.parse(encoded(second_operation)))


async def test_revoke_deletes_rooms_scrubs_binding_and_is_idempotent(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, store, backend = _service(tmp_path, clock)
    provision = ProvisionRequest.parse(encoded(provision_payload()))
    await service.provision(provision)
    request = RevokeRequest.parse(encoded(revoke_payload()))

    first = await service.revoke(request)
    second = await service.revoke(request)

    assert first == second
    assert json.loads(first) == {
        "device_id": "device-1",
        "operation": "channel.revoked-device",
        "operation_id": "revoke-1",
    }
    assert len(backend.revoke_calls) == 1
    stored = store.provision("enrollment-1")
    assert stored is not None
    assert stored.status == "revoked"
    assert stored.response_json == ""
    assert stored.expires_at_ms == 0
    with pytest.raises(IdempotencyConflict):
        await service.provision(provision)


async def test_revoke_unknown_device_is_desired_state_success(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    request = RevokeRequest.parse(encoded(revoke_payload(device_id="unknown")))

    response = await service.revoke(request)

    assert json.loads(response)["device_id"] == "unknown"
    assert backend.revoke_calls == []
