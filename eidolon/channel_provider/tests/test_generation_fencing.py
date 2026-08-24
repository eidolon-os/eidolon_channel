from __future__ import annotations

import asyncio
import json

import pytest
from eidolon_sdk.device_foundation.v1 import DeviceRef

from eidolon.channel_provider.contracts import (
    IdempotencyConflict,
    InvalidTransition,
    ProvisionRequest,
    RevokeRequest,
    StaleGeneration,
)
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.store import PROVISION, ChannelProviderStore

from .helpers import FakeAdapter, encoded, provision_payload, revoke_payload


def _service(path, clock: list[int], adapter: FakeAdapter | None = None):
    backend = adapter or FakeAdapter(name="livekit", ttl_seconds=60)
    store = ChannelProviderStore(path)
    service = ChannelProviderService(
        store=store,
        registry=AdapterRegistry([backend], preference=("livekit",)),
        agent_name="eidolon",
        now_ms=lambda: clock[0],
    )
    service.initialize()
    return service, store, backend


def _request(*, operation_id="operation-1", operation="channel.provision-device", **ref):
    payload = provision_payload(device_ref=ref)
    payload["operation_id"] = operation_id
    payload["operation"] = operation
    return ProvisionRequest.parse(encoded(payload))


def test_contract_consumes_the_canonical_sdk_device_ref() -> None:
    request = _request(owner_domain_generation=3, claim_generation=5, trust_epoch=7)

    assert type(request.device_ref) is DeviceRef
    assert tuple(DeviceRef.model_fields) == (
        "device_instance_id",
        "owner_domain_id",
        "owner_domain_generation",
        "claim_generation",
        "trust_epoch",
    )


async def test_new_generation_fences_old_active_and_old_request_is_stale(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, store, backend = _service(tmp_path / "provider.sqlite3", clock)
    old = _request(operation_id="same-id")
    await service.provision(old)

    new = _request(operation_id="same-id", trust_epoch=2)
    await service.provision(new)

    fenced = store.operation(old.device_ref, PROVISION, old.operation_id)
    assert fenced is not None
    assert (fenced.status, fenced.terminal_reason) == ("fenced", "generation_advanced")
    assert fenced.handle_json == ""
    assert store.active_device(new.device_ref) is not None
    assert len(backend.closed) == 1
    with pytest.raises(StaleGeneration):
        await service.provision(old)


async def test_expired_old_generation_is_explicitly_fenced_by_new_generation(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, store, _backend = _service(tmp_path / "provider.sqlite3", clock)
    old = _request()
    await service.provision(old)
    clock[0] += 60_001
    assert store.expire_credentials(clock[0]) == 1

    await service.provision(_request(operation_id="new", claim_generation=2))

    fenced = store.operation(old.device_ref, PROVISION, old.operation_id)
    assert fenced is not None
    assert (fenced.status, fenced.terminal_reason) == ("fenced", "generation_advanced")


@pytest.mark.parametrize(
    ("old_ref", "new_ref"),
    [
        (
            {"owner_domain_generation": 1, "claim_generation": 1, "trust_epoch": 4},
            {"owner_domain_generation": 1, "claim_generation": 2, "trust_epoch": 1},
        ),
        (
            {"owner_domain_generation": 2, "claim_generation": 7, "trust_epoch": 4},
            {"owner_domain_generation": 3, "claim_generation": 1, "trust_epoch": 1},
        ),
    ],
)
async def test_higher_claim_or_owner_domain_generation_can_reset_inner_epoch(
    tmp_path, old_ref, new_ref
) -> None:
    clock = [1_700_000_000_000]
    service, store, _backend = _service(tmp_path / "provider.sqlite3", clock)
    old = _request(operation_id="old", **old_ref)
    new = _request(operation_id="new", **new_ref)

    await service.provision(old)
    await service.provision(new)

    assert store.active_device(new.device_ref) is not None
    fenced = store.operation(old.device_ref, PROVISION, old.operation_id)
    assert fenced is not None and fenced.status == "fenced"


async def test_restart_preserves_generation_high_watermark(tmp_path) -> None:
    clock = [1_700_000_000_000]
    path = tmp_path / "provider.sqlite3"
    service, _store, _backend = _service(path, clock)
    old = _request(operation_id="old")
    new = _request(operation_id="new", trust_epoch=2)
    await service.provision(old)
    await service.provision(new)

    restarted, store, _fresh_backend = _service(path, clock)
    with pytest.raises(StaleGeneration):
        await restarted.provision(old)
    assert store.active_device(new.device_ref) is not None


async def test_same_scope_id_different_payload_is_stable_conflict(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path / "provider.sqlite3", clock)
    request = _request()
    first = await service.provision(request)
    changed = provision_payload(owner_id="owner_2")
    changed["operation_id"] = request.operation_id

    with pytest.raises(IdempotencyConflict):
        await service.provision(ProvisionRequest.parse(encoded(changed)))

    assert json.loads(await service.provision(request)) == json.loads(first)
    assert len(backend.opened) == 1


async def test_refresh_is_invalid_before_expiry_and_explicit_after_expiry(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, store, _backend = _service(tmp_path / "provider.sqlite3", clock)
    provision = _request()
    await service.provision(provision)
    refresh = _request(operation_id="refresh-1", operation="channel.refresh-device")

    with pytest.raises(InvalidTransition):
        await service.provision(refresh)

    clock[0] += 60_001
    await service.provision(refresh)
    old = store.operation(provision.device_ref, PROVISION, provision.operation_id)
    assert old is not None
    assert (old.status, old.terminal_reason) == ("fenced", "credential_refreshed")


async def test_stale_revocation_cannot_reverse_affect_new_generation(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, store, backend = _service(tmp_path / "provider.sqlite3", clock)
    old = _request()
    await service.provision(old)
    new = _request(operation_id="new", trust_epoch=2)
    await service.provision(new)
    closed_before = list(backend.closed)

    stale_revoke = RevokeRequest.parse(encoded(revoke_payload()))
    with pytest.raises(StaleGeneration):
        await service.revoke(stale_revoke)

    assert store.active_device(new.device_ref) is not None
    assert backend.closed == closed_before


async def test_two_processes_racing_same_generation_commit_one_operation(tmp_path) -> None:
    clock = [1_700_000_000_000]
    path = tmp_path / "provider.sqlite3"
    both_opening = asyncio.Event()
    arrivals = [0]

    class RacingAdapter(FakeAdapter):
        async def open(self, spec, *, issued_at_ms):
            arrivals[0] += 1
            if arrivals[0] == 2:
                both_opening.set()
            await both_opening.wait()
            return await super().open(spec, issued_at_ms=issued_at_ms)

    first, store, first_backend = _service(path, clock, RacingAdapter(name="livekit"))
    second, _store2, second_backend = _service(path, clock, RacingAdapter(name="livekit"))
    request = _request()

    left, right = await asyncio.gather(first.provision(request), second.provision(request))

    assert left == right
    assert len(store.active_provisions()) == 1
    assert len(first_backend.opened) + len(second_backend.opened) == 2
    assert len(first_backend.closed) + len(second_backend.closed) == 1
