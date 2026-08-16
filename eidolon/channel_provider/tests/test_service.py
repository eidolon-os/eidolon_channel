from __future__ import annotations

import base64
import json

import pytest

from eidolon.channel_provider.contracts import (
    IdempotencyConflict,
    ProvisionRequest,
    RevokeRequest,
)
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import FakeAdapter, encoded, livekit_config, provision_payload, revoke_payload


def _service(tmp_path, clock: list[int], backend: FakeAdapter | None = None):
    config = livekit_config()
    resolved_backend = backend or FakeAdapter(name='livekit', ttl_seconds=config.grant_ttl_seconds)
    store = ChannelProviderStore(tmp_path / "provider.sqlite3")
    service = ChannelProviderService(
        store=store,
        registry=AdapterRegistry([resolved_backend], preference=('livekit',)),
        agent_name='eidolon',
        refresh_before_expiry_seconds=config.refresh_before_expiry_seconds,
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
    assert len(backend.opened) == 1
    assert len(restarted_backend.opened) == 0
    response = json.loads(first)
    assert response["operation"] == "channel.provisioned-device"
    assert response["operation_id"] == "enrollment-1"
    assert response["device_id"] == "device-1"
    assert response["channels"][0]["binding_format"] == (
        "application/vnd.eidolon.livekit-session+json;v=2"
    )
    binding = _binding(first)
    # One device, one channel: the binding names a single session, not a pair.
    assert set(binding) == {"device", "resource"}


async def test_provision_refreshes_only_near_expiry_with_stable_resources(tmp_path) -> None:
    clock = [1_700_000_000_000]
    request = ProvisionRequest.parse(encoded(provision_payload()))
    service, _store, backend = _service(tmp_path, clock)
    first = await service.provision(request)

    clock[0] += (1800 - 119) * 1000
    refreshed = await service.provision(request)

    assert refreshed != first
    assert len(backend.opened) == 2
    assert len(backend.opened) == 2
    assert _binding(refreshed)["resource"] == _binding(first)["resource"]
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
    assert len(backend.closed) == 1
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
    assert backend.closed == []
