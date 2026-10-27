from __future__ import annotations

import asyncio
import sqlite3

import pytest

from eidolon.channel_provider.contracts import (
    BackendUnavailable,
    CLOSE_SESSION,
    OPEN_SESSION,
    InvalidTransition,
    IdempotencyConflict,
    ProvisionRequest,
    SessionRequest,
)
from eidolon.channel_provider.store import ChannelProviderStore
from .helpers import FakeAdapter, encoded, provision_payload, session_payload
from .test_service import _service


class PersonalAdapter(FakeAdapter):
    def __init__(self):
        super().__init__(name="livekit", ttl_seconds=1800)
        self.observers = {}
        self.events = []
        self.cleanup_error = False
        self.open_error = False

    async def resume_prepared_channels(self, handles):
        self.resumed_handles = handles

    def observe_session_end(self, handle, session_id, callback):
        self.observers[session_id] = callback
        return lambda: self.observers.pop(session_id, None)

    async def open_session(self, handle, conversation_id, *, personal_memory=None, **kwargs):
        self.events.append(("open", conversation_id))
        if self.open_error:
            raise BackendUnavailable("unknown dispatch result")
        await super().open_session(handle, conversation_id, **kwargs)

    async def close_session(self, handle, conversation_id):
        self.events.append(("stop", conversation_id))
        return []  # Worker has already removed its dispatch.

    async def end_prepared_session(self, handle, session_id):
        self.events.append(("clear", session_id))
        if self.cleanup_error:
            raise BackendUnavailable("body cleanup unconfirmed")


def request(session="private-a", *, person="a", closing=False):
    payload = session_payload(operation=CLOSE_SESSION if closing else OPEN_SESSION)
    payload["conversation_id"] = session
    if not closing and person is not None:
        payload["personal_memory"] = {"person_id": person, "person_revision":1, "identity_epoch": 0}
    return SessionRequest.parse(
        encoded(payload), expected=CLOSE_SESSION if closing else OPEN_SESSION
    )


async def ready(tmp_path):
    adapter = PersonalAdapter()
    service, store, _ = _service(tmp_path, [1_700_000_000_000], adapter)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    return service, store, adapter


async def test_close_cleans_body_even_after_worker_deleted_dispatch(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    await service.close_session(request(closing=True))
    assert adapter.events == [("open", "private-a"), ("stop", "private-a"), ("clear", "private-a")]
    assert store.personal_session("private-a")["state"] == "closed"
    await service.close_session(request(closing=True))
    assert len(adapter.events) == 3
    with pytest.raises(InvalidTransition):
        await service.open_session(request())
    await service.open_session(request("private-b", person="b"))


async def test_failed_close_blocks_default_device_and_private_open_until_retry(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    adapter.cleanup_error = True
    with pytest.raises(BackendUnavailable):
        await service.close_session(request(closing=True))
    assert store.personal_session("private-a")["state"] == "closing"
    for next_request in (request("default", person=None), request("private-b", person="b")):
        with pytest.raises(InvalidTransition):
            await service.open_session(next_request)
    ref = request().device_ref
    with pytest.raises(InvalidTransition):
        await service._converge_serving(
            ref,
            serving=True,
            conversation_id="device-new",
            session_intent="user_initiated",
            device_requested=True,
        )
    with pytest.raises(InvalidTransition):
        service._personal_sessions.require_available(ref.device_instance_id)
    adapter.cleanup_error = False
    await service.close_session(request(closing=True))
    await service.open_session(request("default", person=None))


async def test_recovery_retains_reservation_until_cleanup_ack(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    # Simulate a process restart after a lost worker/end event.
    recovered, reopened, _ = _service(tmp_path, [1_700_000_000_000], adapter)
    adapter.cleanup_error = True
    await recovered.start()
    assert reopened.personal_session("private-a")["state"] == "closing"
    with pytest.raises(InvalidTransition):
        await recovered.open_session(request("default", person=None))
    adapter.cleanup_error = False
    await recovered.close_session(request(closing=True))
    assert reopened.personal_session("private-a")["state"] == "closed"
    await recovered.shutdown()


async def test_trusted_end_event_uses_same_durable_close_path(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    callback = adapter.observers["private-a"]
    callback()
    callback("worker-disconnected")
    await asyncio.gather(*service._personal_sessions._tasks.values())
    assert store.personal_session("private-a")["state"] == "closed"
    assert adapter.events.count(("clear", "private-a")) == 1


async def test_unknown_open_outcome_preserves_cleanup_obligation(tmp_path):
    service, store, adapter = await ready(tmp_path)
    adapter.open_error = adapter.cleanup_error = True
    with pytest.raises(BackendUnavailable):
        await service.open_session(request())
    await asyncio.gather(*service._personal_sessions._tasks.values())
    assert store.personal_session("private-a")["state"] == "closing"
    with pytest.raises(InvalidTransition):
        await service.open_session(request("private-b", person="b"))


async def test_private_binding_is_immutable_and_closed_ids_cannot_be_reclassified(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    with pytest.raises(IdempotencyConflict):
        await service.open_session(request(person="b"))
    assert store.personal_session("private-a")["binding"]["personal_memory"]["person_id"] == "a"
    await service.close_session(request(closing=True))
    with pytest.raises(InvalidTransition):
        await service.open_session(request(person=None))


def test_v5_database_upgrades_with_personal_reservation_constraints(tmp_path):
    path = tmp_path / "store.sqlite3"
    store = ChannelProviderStore(path)
    store.initialize()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE personal_sessions")
        conn.execute("PRAGMA user_version=5")
    store.initialize()
    store.claim_personal_session("s1", "device", {"person": "a"})
    with pytest.raises(InvalidTransition):
        store.claim_personal_session("s2", "device", {"person": "b"})
    with pytest.raises(IdempotencyConflict):
        store.claim_personal_session("s1", "device", {"person": "b"})
    assert ChannelProviderStore(path).pending_personal_sessions()[0]["session_id"] == "s1"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 6


async def test_http_personal_lifecycle_errors_and_retry(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from eidolon.channel_provider.http import create_app
    from .helpers import device_ref

    adapter = PersonalAdapter()
    service, store, _ = _service(tmp_path, [1_700_000_000_000], adapter)
    client = TestClient(TestServer(create_app(service=service, bearer_token="s" * 32)))
    headers = {"Authorization": "Bearer " + "s" * 32}
    opening = session_payload(operation=OPEN_SESSION)
    opening.update(
        conversation_id="http-private", personal_memory={"person_id": "p", "person_revision":1, "identity_epoch": 0}
    )
    closing = session_payload(operation=CLOSE_SESSION)
    closing["conversation_id"] = "http-private"
    try:
        await client.start_server()
        unauthorized = await client.post("/v1/device-channels/sessions/open", json=opening)
        assert unauthorized.status == 401
        assert store.personal_session("http-private") is None
        provisioned = await client.post(
            "/v1/device-channels/provision", json=provision_payload(), headers=headers
        )
        assert provisioned.status == 200
        bad = {**opening, "personal_memory": {"person_id": "p", "person_revision":1, "identity_epoch": True}}
        assert (
            await client.post("/v1/device-channels/sessions/open", json=bad, headers=headers)
        ).status == 422
        opened = await client.post(
            "/v1/device-channels/sessions/open", json=opening, headers=headers
        )
        assert opened.status == 200
        assert (await opened.json())["conversation_id"] == "http-private"
        adapter.cleanup_error = True
        failed = await client.post(
            "/v1/device-channels/sessions/close", json=closing, headers=headers
        )
        assert failed.status == 503
        assert (await failed.json())["retryable"] is True
        pending = session_payload(operation=OPEN_SESSION)
        pending["conversation_id"] = "default-next"
        assert (
            await client.post("/v1/device-channels/sessions/open", json=pending, headers=headers)
        ).status == 409
        crossed = {**closing, "device_ref": device_ref(trust_epoch=2)}
        assert (
            await client.post("/v1/device-channels/sessions/close", json=crossed, headers=headers)
        ).status == 409
        assert store.personal_session("http-private")["state"] == "closing"
        adapter.cleanup_error = False
        for _ in range(2):
            closed = await client.post(
                "/v1/device-channels/sessions/close", json=closing, headers=headers
            )
            assert closed.status == 200
            assert (await closed.json())["serving"] is False
        assert store.personal_session("http-private")["state"] == "closed"
        assert (
            await client.post("/v1/device-channels/sessions/open", json=pending, headers=headers)
        ).status == 200
    finally:
        adapter.cleanup_error = False
        await client.close()


async def test_concurrent_new_identity_waits_for_cleanup_ack(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    entered, release = asyncio.Event(), asyncio.Event()
    original = adapter.end_prepared_session

    async def delayed(handle, session_id):
        entered.set()
        await release.wait()
        await original(handle, session_id)

    adapter.end_prepared_session = delayed
    closing = asyncio.create_task(service.close_session(request(closing=True)))
    await entered.wait()
    opening = asyncio.create_task(service.open_session(request("private-b", person="b")))
    await asyncio.sleep(0)
    assert not opening.done()
    assert store.personal_session("private-a")["state"] == "closing"
    release.set()
    await asyncio.gather(closing, opening)
    assert adapter.events.index(("clear", "private-a")) < adapter.events.index(
        ("open", "private-b")
    )


async def test_failed_observer_registration_does_not_orphan_open_claim(tmp_path):
    service, store, adapter = await ready(tmp_path)

    def fail(*args):
        raise InvalidTransition("observer unavailable")

    adapter.observe_session_end = fail
    with pytest.raises(InvalidTransition):
        await service.open_session(request())
    await asyncio.gather(*service._personal_sessions._tasks.values())
    assert store.personal_session("private-a")["state"] == "closed"
    assert ("open", "private-a") not in adapter.events


async def test_credential_refresh_preserves_same_personal_claim(tmp_path):
    from dataclasses import replace

    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    provision = store.active_device(request().device_ref)
    refreshed = replace(
        provision, operation_id="refreshed", operation_kind="channel.refresh-device"
    )
    record = service._personal_sessions.claim(
        refreshed,
        "private-a",
        personal_memory={"person_id": "a", "person_revision":1, "identity_epoch": 0},
        target_companion_id=None,
        session_intent="user_initiated",
    )
    assert record["binding"]["operation_id"] == provision.operation_id
    await service.close_session(request(closing=True))
