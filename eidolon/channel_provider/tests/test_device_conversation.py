import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.control.device_conversation import DeviceConversationSelection
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import BackendUnavailable, ProvisionRequest, Forbidden, InvalidTransition
from eidolon.channel_provider.ports import ServingAction, ServingRequest
from .helpers import FakeAdapter, provision_payload, encoded
from .test_service import _service


class Adapter(FakeAdapter):
    def __init__(self):
        super().__init__(name="livekit")
        self.launched = asyncio.Event()
        self.sent = []
        self.end_prepared_session = AsyncMock()
        self.quiesce_prepared_sessions = AsyncMock()
        self.prepared_endpoints_present = AsyncMock(return_value=True)
        self.resume_prepared_channels = AsyncMock()
        self.require_idle = AsyncMock()
        self.defer_requests = False
        self.observers = {}
        self.directed = []

    async def open(self, spec, **kwargs):
        grant = await super().open(spec, **kwargs)
        return replace(grant, handle=grant.handle | {"device": spec.device_id, "room": spec.device_id})

    async def deliver_control(self, handle, command, *, wait_for_terminal):
        assert wait_for_terminal
        self.sent.append((handle, command))
        if self.defer_requests:
            await asyncio.Event().wait()
        await self.watched[handle["resource"]](ServingRequest(ServingAction.START,
            f"native-{len(self.sent)}", command["id"]))
        await self.launched.wait()
        return "succeeded"

    async def open_directed_session(self, source, target, **kwargs):
        self.directed.append((source, target, kwargs))
        self.launched.set()

    def observe_session_end(self, handle, session_id, callback):
        self.observers[handle["device"]] = callback
        return lambda: self.observers.pop(handle["device"], None)


async def prepare(tmp_path):
    backend = Adapter()
    service, store, _ = _service(tmp_path, [1700000000000], backend)
    requests = []
    for name in ("input", "speaker"):
        request = ProvisionRequest.parse(encoded(provision_payload(
            device_id=named_device_instance_id(name))))
        await service.provision(request)
        requests.append(request)
    selection = DeviceConversationSelection(session_id="dialogue", input_device=requests[0].device_ref,
        output_device=requests[1].device_ref, target_companion_id="selected")
    return service, backend, selection


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


async def test_prepare_uses_native_ids_and_one_worker_then_close_releases_both(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    result = await service.open_device_conversation(selection, authenticated_owner_id="owner_1")
    assert result["state"] == "preparing"
    visit = service._device_conversations[("owner_1", "dialogue")]
    await wait_for(lambda: visit.state == "ready")
    assert len(backend.directed) == 1
    assert backend.directed[0][2] == dict(source_session_id="native-1",
        target_session_id="native-2", target_companion_id="selected")
    retry = await service.open_device_conversation(selection, authenticated_owner_id="owner_1")
    assert retry["state"] == "ready"
    assert len(backend.sent) == 2
    result = await service.device_conversation("dialogue", authenticated_owner_id="owner_1", close=True)
    assert result["state"] == "closed"
    assert not service._transport_scopes
    assert backend.end_prepared_session.await_count == 2
    assert not backend.observers
    assert not backend.sessions_opened
    await service.shutdown()


async def test_cancelled_preparation_cannot_turn_late_request_into_single_chat(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    backend.defer_requests = True
    await service.open_device_conversation(selection, authenticated_owner_id="owner_1")
    await wait_for(lambda: len(backend.sent) == 2)
    await service.device_conversation("dialogue", authenticated_owner_id="owner_1", close=True)
    for handle, command in backend.sent:
        with pytest.raises(InvalidTransition, match="no longer active"):
            await backend.watched[handle["resource"]](ServingRequest(
                ServingAction.START, "late-session", command["id"]))
    assert not backend.directed
    assert not backend.sessions_opened
    await service.shutdown()


async def test_wrong_owner_cannot_prepare_or_observe_conversation(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    with pytest.raises(Forbidden):
        await service.open_device_conversation(selection, authenticated_owner_id="different")
    assert not backend.sent
    assert not service._transport_scopes
    await service.shutdown()


async def test_worker_end_releases_both_reservations(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id="owner_1")
    visit = service._device_conversations[("owner_1", "dialogue")]
    await wait_for(lambda: visit.state == "ready")
    backend.observers[selection.input_device.device_instance_id]()
    await visit.task
    assert visit.state == "closed"
    assert not service._transport_scopes
    await service.shutdown()


async def test_http_preparing_status_close_and_auth(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from eidolon.channel_provider.http import create_app
    service, backend, selection = await prepare(tmp_path)
    token = 'directed-test-token-at-least-32-bytes'
    path = '/v1/device-conversations'
    body = dict(owner_id='owner_1', selection=selection.model_dump(mode='json'))
    async with TestClient(TestServer(create_app(service=service, bearer_token=token))) as client:
        assert (await client.post(f'{path}/open', json=body)).status == 401
        assert not backend.sent
        headers = dict(Authorization=f'Bearer {token}')
        response = await client.post(f'{path}/open', json=body, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['state'] == 'preparing'
        visit = service._device_conversations[('owner_1', selection.session_id)]
        await wait_for(lambda: visit.state == 'ready')
        query = dict(owner_id='owner_1', session_id=selection.session_id)
        response = await client.post(f'{path}/status', json=query, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['state'] == 'ready'
        response = await client.post(f'{path}/close', json=query, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['state'] == 'closed'


async def test_closing_old_visit_never_cancels_new_visit_on_same_devices(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    backend.require_idle.side_effect = InvalidTransition('device busy')
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    failed = service._device_conversations[('owner_1', selection.session_id)]
    await failed.task
    assert failed.state == 'failed'
    backend.require_idle.side_effect = None
    other = selection.model_copy(update={'session_id': 'new-dialogue'})
    await service.open_device_conversation(other, authenticated_owner_id='owner_1')
    active = service._device_conversations[('owner_1', other.session_id)]
    await wait_for(lambda: active.state == 'ready')
    result = await service.device_conversation(selection.session_id,
        authenticated_owner_id='owner_1', close=True)
    assert result['state'] == 'closed'
    assert active.state == 'ready'
    assert not active.task.done()
    assert len(service._transport_scopes) == 2
    await service.shutdown()


async def test_partial_cleanup_failure_keeps_all_reservations_until_retry(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    backend.end_prepared_session.side_effect = [RuntimeError('endpoint unavailable'), None]
    backend.observers[selection.input_device.device_instance_id]()
    with pytest.raises(BackendUnavailable, match="cleanup unconfirmed"):
        await visit.task
    assert backend.end_prepared_session.await_count == 2
    assert len(service._transport_scopes) == 2
    status = await service.device_conversation(selection.session_id, authenticated_owner_id="owner_1")
    assert status["state"] == "failed"
    assert "cleanup unconfirmed" in status["error"]
    backend.end_prepared_session.side_effect = None
    result = await service.device_conversation(selection.session_id,
        authenticated_owner_id='owner_1', close=True)
    assert result['state'] == 'closed'
    assert result['error'] == ''
    assert not service._transport_scopes
    await service.shutdown()


async def test_restart_recovers_only_unconfirmed_close_without_reopening_generation(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    pending = selection.input_device.device_instance_id
    async def close_endpoint(handle, *args, **kwargs):
        if handle['device'] == pending:
            raise BackendUnavailable('ACK lost after physical execution')
    backend.end_prepared_session.side_effect = close_endpoint
    backend.observers[pending]()
    with pytest.raises(BackendUnavailable):
        await visit.task
    assert visit._ended_devices == {selection.output_device.device_instance_id}
    recovered_backend = Adapter()
    recovered, _, _ = _service(tmp_path, [1700000000000], recovered_backend)
    await recovered.start()
    restored = recovered._device_conversations[('owner_1', selection.session_id)]
    # Reservation exists before the recovery task gets any opportunity to run.
    assert len(recovered._transport_scopes) == 2
    await restored.task
    assert restored.state == 'closed'
    assert not recovered._transport_scopes
    assert not recovered_backend.sent and not recovered_backend.directed
    recovered_backend.quiesce_prepared_sessions.assert_not_awaited()
    recovered_backend.resume_prepared_channels.assert_awaited_once()
    assert recovered_backend.end_prepared_session.await_count == 1
    assert recovered_backend.end_prepared_session.call_args.args[0]['device'] == pending
    await recovered.shutdown()
    backend.end_prepared_session.side_effect = None
    await service.shutdown()


async def test_closed_scene_survives_restart_without_claiming_new_scene_devices(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    await service.device_conversation(selection.session_id, authenticated_owner_id='owner_1', close=True)
    recovered_backend = Adapter()
    recovered, _, _ = _service(tmp_path, [1700000000000], recovered_backend)
    await recovered.start()
    assert not recovered._transport_scopes
    status = await recovered.device_conversation(selection.session_id, authenticated_owner_id='owner_1')
    assert status['state'] == 'closed'
    recovered_backend.end_prepared_session.assert_not_awaited()
    await recovered.shutdown()
    await service.shutdown()


async def test_status_cannot_keep_ready_after_transport_was_lost(tmp_path):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    backend.prepared_endpoints_present.return_value = False
    status = await service.device_conversation(selection.session_id, authenticated_owner_id='owner_1')
    assert status['state'] == 'closing'
    await visit.task
    assert visit.state == 'failed' and visit.closure_complete
    assert visit.error == 'ENDPOINT_REACHABILITY_LOST'
    backend.quiesce_prepared_sessions.assert_awaited_once()
    assert not service._transport_scopes
    await service.shutdown()


@pytest.mark.parametrize('closing', [False, True])
@pytest.mark.parametrize('renewal', ['expired', 'runtime_changed'])
async def test_standing_channel_renewal_never_waits_for_scene_cleanup(tmp_path, closing, renewal):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    if closing:
        backend.end_prepared_session.side_effect = BackendUnavailable('device must reconnect for ACK')
        visit.stopped.set()
        with pytest.raises(BackendUnavailable):
            await visit.task
    original_scopes = dict(service._transport_scopes)
    if renewal == 'expired':
        service._now_ms = lambda: 1700000000000 + backend.ttl_seconds * 1000 + 1
    else:
        backend.binding_current = lambda handle: False
    payload = provision_payload(device_id=selection.input_device.device_instance_id)
    payload.update(operation='channel.refresh-device', operation_id='renew-to-reconnect')
    # A device cannot ACK anything until this credential response reaches it.
    response = await asyncio.wait_for(service.provision(ProvisionRequest.parse(encoded(payload))), 1)
    assert 'channel.provisioned-device' in response
    assert service._transport_scopes == original_scopes
    if not closing:
        assert visit.state == 'ready' and not visit.task.done()
        backend.quiesce_prepared_sessions.assert_not_awaited()
    backend.end_prepared_session.side_effect = None
    await service.device_conversation(selection.session_id, authenticated_owner_id='owner_1', close=True)
    assert visit.closure_complete and not service._transport_scopes
    await service.shutdown()


@pytest.mark.parametrize('change', ['manifest', 'resource'])
async def test_real_configuration_replacement_still_requires_confirmed_scene_close(tmp_path, change):
    service, backend, selection = await prepare(tmp_path)
    await service.open_device_conversation(selection, authenticated_owner_id='owner_1')
    visit = service._device_conversations[('owner_1', selection.session_id)]
    await wait_for(lambda: visit.state == 'ready')
    backend.end_prepared_session.side_effect = BackendUnavailable('unconfirmed')
    backend.binding_current = lambda handle: False
    payload = provision_payload(device_id=selection.input_device.device_instance_id)
    payload.update(operation='channel.refresh-device', operation_id='replace')
    if change == 'manifest':
        payload['device']['manifest_revision'] = 'new-manifest'
    else:
        original = backend.open
        async def different_resource(*args, **kwargs):
            grant = await original(*args, **kwargs)
            return replace(grant, handle=grant.handle | {'resource': 'new-resource'})
        backend.open = different_resource
    with pytest.raises(BackendUnavailable):
        await service.provision(ProvisionRequest.parse(encoded(payload)))
    assert len(service._transport_scopes) == 2
    assert not visit.closure_complete
    backend.end_prepared_session.side_effect = None
    await service.shutdown()
