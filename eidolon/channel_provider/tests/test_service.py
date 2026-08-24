from __future__ import annotations

import base64
import json

import pytest

from eidolon.channel_provider.contracts import (
    CLOSE_SESSION,
    OPEN_SESSION,
    IdempotencyConflict,
    ProvisionRequest,
    RevokeRequest,
    SessionRequest,
    UnknownChannel,
)
from eidolon.channel_provider.ports import ServingRequest
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import (
    FakeAdapter,
    encoded,
    livekit_config,
    provision_payload,
    revoke_payload,
    session_payload,
)


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


async def test_characterize_expired_active_operation_blocks_a_new_lifecycle(tmp_path) -> None:
    """Record the deployed generation-blind lock before replacing its contract.

    The legacy request cannot name a DeviceRef generation.  Once its credential
    expires, the row remains active: replaying the same operation refreshes it,
    while a new lifecycle operation for the same device collides with the
    owner/device unique lock.
    """
    clock = [1_700_000_000_000]
    service, store, backend = _service(tmp_path, clock)
    legacy = ProvisionRequest.parse(encoded(provision_payload()))

    first = await service.provision(legacy)
    stored = store.provision(legacy.operation_id)
    assert stored is not None
    clock[0] = stored.expires_at_ms + 1
    assert stored.status == "active"
    assert stored.expires_at_ms < clock[0]

    replayed = await service.provision(legacy)
    assert replayed != first
    assert len(backend.opened) == 2

    new_lifecycle = provision_payload()
    new_lifecycle["operation_id"] = "enrollment-new-generation"
    with pytest.raises(
        IdempotencyConflict,
        match="device already has an active Channel Provider operation",
    ):
        await service.provision(ProvisionRequest.parse(encoded(new_lifecycle)))


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


async def test_a_session_runs_on_the_adapter_that_opened_the_channel(tmp_path) -> None:
    """A session is served by whoever is carrying the device, never re-selected."""
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    opened = await service.open_session(
        SessionRequest.parse(
            encoded(session_payload(operation=OPEN_SESSION)), expected=OPEN_SESSION
        )
    )
    closed = await service.close_session(
        SessionRequest.parse(
            encoded(session_payload(operation=CLOSE_SESSION)), expected=CLOSE_SESSION
        )
    )

    handle = {"resource": "livekit:device-1"}
    assert backend.sessions_opened == [handle]
    assert backend.sessions_closed == [handle]
    assert json.loads(opened) == {
        "operation": "channel.opened-session",
        "device_id": "device-1",
        "channel_id": json.loads(opened)["channel_id"],
        "serving": True,
    }
    assert json.loads(closed)["serving"] is False
    # A conversation beginning or ending never disturbs the channel itself.
    assert backend.closed == []


async def test_a_device_with_no_channel_cannot_hold_a_session(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)

    with pytest.raises(UnknownChannel):
        await service.open_session(
            SessionRequest.parse(
                encoded(session_payload(operation=OPEN_SESSION)), expected=OPEN_SESSION
            )
        )

    assert backend.sessions_opened == []


async def test_a_provisioned_channel_is_listened_to(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)

    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    assert list(backend.watched) == ["livekit:device-1"]


async def test_a_device_can_start_and_stop_its_own_conversation(tmp_path) -> None:
    """The device asking over its channel converges the same state a call does."""
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    await backend.device_asks("device-1", ServingRequest.START)
    await backend.device_asks("device-1", ServingRequest.STOP)

    handle = {"resource": "livekit:device-1"}
    assert backend.sessions_opened == [handle]
    assert backend.sessions_closed == [handle]


async def test_a_restart_resumes_listening_to_every_open_channel(tmp_path) -> None:
    """A device's right to be heard outlives the process that was hearing it."""
    clock = [1_700_000_000_000]
    service, _store, _backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    await service.provision(
        ProvisionRequest.parse(
            encoded(
                provision_payload(device_id="device-2")
                | {"operation_id": "enrollment-2"}
            )
        )
    )

    restarted, _store2, fresh_backend = _service(tmp_path, clock)
    await restarted.start()

    assert sorted(fresh_backend.watched) == ["livekit:device-1", "livekit:device-2"]
    await fresh_backend.device_asks("device-2", ServingRequest.START)
    assert fresh_backend.sessions_opened == [{"resource": "livekit:device-2"}]


async def test_a_channel_that_cannot_be_watched_is_still_provisioned(tmp_path) -> None:
    """A device that cannot ask must still get the channel it was promised."""
    clock = [1_700_000_000_000]
    backend = FakeAdapter(name="livekit")

    async def _refuse(handle, *, sink):
        raise ConnectionError("transport refused to carry requests")

    backend.accept_requests = _refuse
    service, _store, _ = _service(tmp_path, clock, backend)

    response = await service.provision(
        ProvisionRequest.parse(encoded(provision_payload()))
    )

    assert json.loads(response)["operation"] == "channel.provisioned-device"
    assert len(backend.opened) == 1


async def test_revocation_stops_listening_to_the_channel(tmp_path) -> None:
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))

    await service.revoke(RevokeRequest.parse(encoded(revoke_payload())))

    assert backend.watched == {}
    assert backend.stopped == [{"resource": "livekit:device-1"}]


async def test_a_revoked_device_cannot_hold_a_session(tmp_path) -> None:
    """Revocation must actually cut the device off, sessions included."""
    clock = [1_700_000_000_000]
    service, _store, backend = _service(tmp_path, clock)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    await service.revoke(RevokeRequest.parse(encoded(revoke_payload())))

    with pytest.raises(UnknownChannel):
        await service.open_session(
            SessionRequest.parse(
                encoded(session_payload(operation=OPEN_SESSION)), expected=OPEN_SESSION
            )
        )

    assert backend.sessions_opened == []
