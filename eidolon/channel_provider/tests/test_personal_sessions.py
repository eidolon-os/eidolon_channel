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
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7


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


async def test_authority_loss_closes_idle_personal_session_and_retries_body_cleanup(tmp_path):
    from unittest.mock import AsyncMock
    service, store, adapter = await ready(tmp_path)
    await service.open_session(request())
    resolver = AsyncMock(return_value="private-realm")
    await service._personal_sessions.reconcile_authority(resolver)
    assert store.personal_session("private-a")["state"] == "open"
    resolver.side_effect = LookupError("person revision changed")
    adapter.cleanup_error = True
    await service._personal_sessions.reconcile_authority(resolver)
    await asyncio.gather(*tuple(service._personal_sessions._tasks.values()))
    assert store.personal_session("private-a")["state"] == "closing"
    with pytest.raises(InvalidTransition):
        await service.open_session(request("unknown", person=None))
    adapter.cleanup_error = False
    await service._personal_sessions.reconcile_authority(resolver)
    await asyncio.gather(*tuple(service._personal_sessions._tasks.values()))
    assert store.personal_session("private-a")["state"] == "closed"
    # Producers went on the first attempt; the retry is only about the Body.
    assert adapter.events.count(("stop", "private-a")) == 1
    assert adapter.events[-2:] == [("clear", "private-a"), ("clear", "private-a")]
    assert resolver.await_count == 2  # closing retry does not need Data availability


async def test_default_and_device_sessions_hold_companion_until_acknowledged_close(tmp_path):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    service, store, adapter = await ready(tmp_path)
    service._companion_authority = AsyncMock(return_value='partner')
    ordinary = request('ordinary', person=None)
    await service.open_session(ordinary)
    assert store.personal_session('ordinary')['binding']['target_companion_id'] == 'partner'
    # Replayed open uses the pinned target and does not change its identity.
    await service.open_session(ordinary)
    assert service._companion_authority.call_args.kwargs['target_companion_id'] == 'partner'
    with pytest.raises(InvalidTransition, match='ordinary'):
        service._require_scene_companions_available('owner_1', ['partner'])
    adapter.cleanup_error = True
    with pytest.raises(BackendUnavailable):
        await service.close_session(request('ordinary', person=None, closing=True))
    with pytest.raises(InvalidTransition, match='ordinary'):
        service._require_scene_companions_available('owner_1', ['partner'])
    adapter.cleanup_error = False
    await service.close_session(request('ordinary', person=None, closing=True))
    service._require_scene_companions_available('owner_1', ['partner'])
    # Scene reservations also stop ordinary/device starts before dispatch.
    service._team_conversations[('owner_1', 'team')] = SimpleNamespace(
        closure_complete=False, selection=SimpleNamespace(members=[SimpleNamespace(companion_id='partner')]))
    for device_requested in (False, True):
        with pytest.raises(InvalidTransition, match='team'):
            await service._converge_serving(ordinary.device_ref, serving=True,
                conversation_id='next', session_intent='user_initiated', device_requested=device_requested)
    assert store.personal_session('next') is None
    service._team_conversations.clear()
    for session in ('device-first', 'device-restarted'):
        await service._converge_serving(ordinary.device_ref, serving=True,
            conversation_id=session, session_intent='user_initiated', device_requested=True)
    assert store.personal_session('device-first')['state'] == 'closed'
    assert store.personal_session('device-restarted')['state'] == 'open'
    # Restart recovers by cleaning; it does not start another dialogue.
    await service._personal_sessions.recover()
    assert store.personal_session('device-restarted')['state'] == 'closed'
    assert [e for e in adapter.events if e[0] == 'open'] == [
        ('open', 'ordinary'), ('open', 'ordinary'), ('open', 'device-first'), ('open', 'device-restarted')]


async def _device_session(service, adapter, session='device-a'):
    from unittest.mock import AsyncMock
    service._companion_authority = AsyncMock(return_value='partner')
    ref = request().device_ref
    await service._converge_serving(ref, serving=True, conversation_id=session,
        session_intent='user_initiated', device_requested=True)
    return ref


async def test_the_body_saying_it_ended_is_its_own_cleanup_receipt(tmp_path):
    """A phone drops its channel as soon as it has said so; nobody is left to answer room.leave."""
    service, store, adapter = await ready(tmp_path)
    ref = await _device_session(service, adapter)
    adapter.cleanup_error = True  # Asking the Body again would go unanswered.
    await service._converge_serving(ref, serving=False, conversation_id='device-a',
        session_intent=None, device_requested=True)
    assert store.personal_session('device-a')['state'] == 'closed'
    assert ('stop', 'device-a') in adapter.events
    assert ('clear', 'device-a') not in adapter.events
    service._require_scene_companions_available('owner_1', ['partner'])


async def test_a_stop_the_body_did_not_ask_for_still_waits_for_its_receipt(tmp_path):
    service, store, adapter = await ready(tmp_path)
    await _device_session(service, adapter)
    adapter.cleanup_error = True
    with pytest.raises(BackendUnavailable):
        await service.close_session(request('device-a', person=None, closing=True))
    assert store.personal_session('device-a')['state'] == 'closing'
    with pytest.raises(InvalidTransition, match='device-a'):
        service._require_scene_companions_available('owner_1', ['partner'])


@pytest.mark.parametrize('lapsed', [False, True])
async def test_revoking_a_device_releases_what_it_held(tmp_path, lapsed):
    """A removed Body can never answer, so its reservation must not outlive it."""
    from eidolon.channel_provider.contracts import RevokeRequest
    from .helpers import revoke_payload
    adapter = PersonalAdapter()
    clock = [1_700_000_000_000]
    service, store, _ = _service(tmp_path, clock, adapter)
    await service.provision(ProvisionRequest.parse(encoded(provision_payload())))
    await _device_session(service, adapter)
    adapter.cleanup_error = True
    service._personal_sessions.request_close('device-a')
    await asyncio.gather(*tuple(service._personal_sessions._tasks.values()))
    assert store.personal_session('device-a')['state'] == 'closing'
    if lapsed:
        clock[0] += 3_600_000  # The credential lapsed; the room did not.
    await service.revoke(RevokeRequest.parse(encoded(revoke_payload())))
    assert len(adapter.closed) == 1  # The room goes either way.
    assert store.personal_session('device-a')['state'] == 'closed'
    service._require_scene_companions_available('owner_1', ['partner'])


async def test_a_body_that_stays_away_is_reported_once(tmp_path, caplog):
    service, store, adapter = await ready(tmp_path)
    await _device_session(service, adapter)
    adapter.cleanup_error = True
    for _ in range(3):
        service._personal_sessions.request_close('device-a')
        await asyncio.gather(*tuple(service._personal_sessions._tasks.values()))
    pending = [r for r in caplog.records if 'personal cleanup pending' in r.getMessage()]
    assert len(pending) == 1
    assert not pending[0].exc_info  # An absent Body is not a crash.
    # Nor does waiting for it keep asking LiveKit to withdraw the same producers.
    assert adapter.events.count(('stop', 'device-a')) == 1
    adapter.cleanup_error = False
    service._personal_sessions.request_close('device-a')
    await asyncio.gather(*tuple(service._personal_sessions._tasks.values()))
    assert store.personal_session('device-a')['state'] == 'closed'


async def test_companion_admission_requires_current_body_and_owned_target():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from .helpers import device_ref
    from eidolon.channel_provider.companion_admission import CompanionAdmission
    from eidolon.channel_provider.contracts import Forbidden
    from eidolon_sdk.device_foundation.v1 import DeviceRef
    ref = DeviceRef.model_validate(device_ref())
    bodies = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(
        owner_id='owner', device_ref=ref, answering_companion_id='partner')))
    runtime = SimpleNamespace(resolve_companion=AsyncMock(return_value=SimpleNamespace(
        owner_id='owner', companion_id='partner', device_id=ref.device_instance_id)))
    admission = CompanionAdmission(bodies, runtime)
    assert await admission.resolve(owner_id='owner', device_ref=ref, target_companion_id=None) == 'partner'
    runtime.resolve_companion.return_value.owner_id = 'other'
    with pytest.raises(Forbidden):
        await admission.resolve(owner_id='owner', device_ref=ref, target_companion_id=None)
    bodies.resolve.return_value.device_ref = ref.model_copy(update={'trust_epoch': ref.trust_epoch + 1})
    with pytest.raises(Forbidden):
        await admission.resolve(owner_id='owner', device_ref=ref, target_companion_id=None)


def test_store_companion_reservation_is_owner_scoped_and_retained_while_closing(tmp_path):
    store = ChannelProviderStore(tmp_path / 'reservations.db')
    store.initialize()
    binding = {'owner_id': 'owner', 'target_companion_id': 'partner'}
    store.claim_personal_session('one', 'device-one', binding)
    store.claim_personal_session('other-owner', 'device-other', binding | {'owner_id': 'other'})
    with pytest.raises(InvalidTransition):
        store.claim_personal_session('two', 'device-two', binding)
    store.transition_personal_session('one', expected='open', state='closing')
    with pytest.raises(InvalidTransition):
        store.claim_personal_session('two', 'device-two', binding)
    store.transition_personal_session('one', expected='closing', state='closed')
    store.claim_personal_session('two', 'device-two', binding)
