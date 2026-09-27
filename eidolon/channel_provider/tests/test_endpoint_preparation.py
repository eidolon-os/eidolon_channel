import asyncio
from unittest.mock import AsyncMock

import pytest
from eidolon.channel_provider.contracts import BackendUnavailable, InvalidTransition
from eidolon.channel_provider.endpoint_preparation import EndpointPreparation
from eidolon.channel_provider.ports import ServingAction, ServingRequest


class Adapter:
    def __init__(self):
        self.launched = asyncio.Event()
        self.sent = []
        self.require_idle = AsyncMock()
        self.end_prepared_session = AsyncMock()
        self.quiesce_prepared_sessions = AsyncMock()
        self.prepared_endpoints_present = AsyncMock(return_value=True)
        self.removed = []

    async def deliver_control(self, handle, command, **kwargs):
        self.sent.append(command)
        self.visit.request(handle['device'], ServingRequest(
            ServingAction.START, 'native-' + handle['device'], command['id']))
        await self.launched.wait()
        return 'succeeded'

    def observe_session_end(self, handle, session_id, callback):
        return lambda: self.removed.append(handle['device'])


async def ready(visit):
    async with asyncio.timeout(2):
        while visit.state != 'ready':
            await asyncio.sleep(0)


async def test_three_endpoints_activate_once_only_after_all_native_sessions():
    adapter = Adapter()
    async def activate(sessions):
        assert sessions == {key: 'native-' + key for key in ('input', 'a', 'b')}
        assert adapter.require_idle.await_count == 3
        adapter.launched.set()
    activate_mock = AsyncMock(side_effect=activate)
    visit = adapter.visit = EndpointPreparation(adapter,
        tuple({'device': key} for key in ('input', 'a', 'b')), activate_mock)
    task = asyncio.create_task(visit.run())
    try:
        await ready(visit)
        activate_mock.assert_awaited_once()
        assert len(adapter.sent) == 3
        visit.stopped.set()
        await task
        await visit.cleanup()
        assert visit.state == 'closed'
        assert set(adapter.removed) == {'input', 'a', 'b'}
        assert adapter.end_prepared_session.await_count == 3
        with pytest.raises(InvalidTransition, match='no longer active'):
            visit.request('a', ServingRequest(ServingAction.START, 'native-a'))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_busy_endpoint_prevents_every_wake():
    adapter = Adapter()
    adapter.require_idle.side_effect = [None, InvalidTransition('busy')]
    activate = AsyncMock()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), activate)
    await visit.run()
    assert visit.state == 'failed'
    assert not adapter.sent
    activate.assert_not_called()


async def test_cleanup_attempts_every_endpoint_and_retains_failure_for_retry():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), AsyncMock())
    visit.sessions = {'a': 'native-a', 'b': 'native-b'}
    adapter.end_prepared_session.side_effect = [RuntimeError('offline'), None]
    with pytest.raises(BackendUnavailable, match="cleanup unconfirmed"):
        await visit.cleanup()
    assert adapter.end_prepared_session.await_count == 2
    assert visit.state != 'closed'
    adapter.end_prepared_session.side_effect = None
    await visit.cleanup()
    assert visit.state == 'closed'


def test_foreign_device_and_duplicate_endpoints_are_rejected():
    with pytest.raises(InvalidTransition):
        EndpointPreparation(Adapter(), ({'device': 'a'}, {'device': 'a'}), AsyncMock())
    visit = EndpointPreparation(Adapter(), ({'device': 'a'},), AsyncMock())
    with pytest.raises(InvalidTransition, match='outside'):
        visit.request('foreign', ServingRequest(ServingAction.START, 'native'))


async def test_early_join_receipt_cleans_every_invited_device_without_session_request():
    adapter = Adapter()
    adapter.deliver_control = AsyncMock(return_value='succeeded')
    activate = AsyncMock()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), activate)
    await visit.run()
    assert visit.state == 'failed'
    assert 'ended early' in visit.error
    assert visit.sessions == {}
    activate.assert_not_awaited()
    await visit.cleanup()
    assert adapter.end_prepared_session.await_count == 2
    for handle in visit.handles:
        adapter.end_prepared_session.assert_any_await(handle, None,
            control_request_id=visit.command_ids[handle['device']])


async def test_partial_preparation_closes_known_session_and_pending_join():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), AsyncMock())
    visit.command_ids = {'a': 'join:a', 'b': 'join:b'}
    visit.sessions = {'a': 'native-a'}
    await visit.cleanup()
    adapter.end_prepared_session.assert_any_await({'device': 'a'}, 'native-a',
        control_request_id='join:a')
    adapter.end_prepared_session.assert_any_await({'device': 'b'}, None,
        control_request_id='join:b')


async def test_producer_barrier_precedes_every_device_close_and_is_retryable():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), AsyncMock())
    visit.sessions = {'a': 'native-a', 'b': 'native-b'}
    adapter.quiesce_prepared_sessions.side_effect = BackendUnavailable('worker still publishing')
    with pytest.raises(BackendUnavailable):
        await visit.cleanup()
    adapter.end_prepared_session.assert_not_awaited()
    adapter.quiesce_prepared_sessions.side_effect = None
    await visit.cleanup()
    assert adapter.quiesce_prepared_sessions.await_count == 2
    assert visit.state == 'closed'


async def test_lost_ack_retries_only_unknown_device_after_actual_execution():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), AsyncMock())
    visit.sessions = {'a': 'native-a', 'b': 'native-b'}
    applied, calls = set(), []
    async def end(handle, session, **kwargs):
        device = handle['device']
        calls.append(device)
        if device not in applied:
            applied.add(device)  # Physical close happened before the lost ACK.
            if device == 'a':
                raise BackendUnavailable('receipt lost')
        # A second correlated room.leave answers SESSION_ALREADY_ENDED.
    adapter.end_prepared_session.side_effect = end
    with pytest.raises(BackendUnavailable):
        await visit.cleanup()
    assert applied == {'a', 'b'}
    assert visit.state != 'closed'
    await asyncio.gather(visit.cleanup(), visit.cleanup())
    assert calls == ['a', 'b', 'a']
    assert adapter.quiesce_prepared_sessions.await_count == 1
    assert visit.state == 'closed'


async def test_cancellation_preserves_completed_device_checkpoint():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'}, {'device': 'b'}), AsyncMock())
    visit.sessions = {'a': 'native-a', 'b': 'native-b'}
    blocked = asyncio.Event()
    async def end(handle, *args, **kwargs):
        if handle['device'] == 'b':
            blocked.set()
            await asyncio.Event().wait()
    adapter.end_prepared_session.side_effect = end
    task = asyncio.create_task(visit.cleanup())
    await blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    adapter.end_prepared_session.reset_mock(side_effect=True)
    await visit.cleanup()
    adapter.end_prepared_session.assert_awaited_once_with({'device': 'b'}, 'native-b',
        control_request_id=None)


async def test_failed_terminal_checkpoint_never_releases_reservation_as_closed():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'},), AsyncMock())
    visit.sessions = {'a': 'native-a'}
    def disk_full():
        if visit._cleanup_complete:
            raise OSError('disk full')
    visit.checkpoint = disk_full
    with pytest.raises(BackendUnavailable, match='checkpoint'):
        await visit.cleanup()
    assert visit.state == 'failed' and not visit._cleanup_complete
    visit.checkpoint = lambda: None
    await visit.cleanup()
    assert visit.state == 'closed'
    adapter.end_prepared_session.assert_awaited_once()


def test_recovery_refuses_completed_state_without_every_owned_endpoint_proof():
    visit = EndpointPreparation(Adapter(), ({'device': 'a'},), AsyncMock())
    data = dict(state='closed', error='', sessions={'a': 'native-a'}, command_ids={},
        producers_revoked=True, ended_devices=[], cleanup_failed=False, cleanup_complete=True)
    with pytest.raises(InvalidTransition, match='terminal proof'):
        visit.restore_lifecycle(data)
    data.update(cleanup_complete=False, ended_devices=['foreign'])
    with pytest.raises(InvalidTransition, match='unowned endpoint'):
        visit.restore_lifecycle(data)


async def test_cleanup_failure_and_recovery_preserve_primary_failure_across_restart():
    adapter = Adapter()
    visit = EndpointPreparation(adapter, ({'device': 'a'},), AsyncMock())
    visit.sessions = {'a': 'native-a'}
    visit.state = 'ready'
    visit.session_ended('TEAM_WORKER_DISCONNECTED')
    adapter.end_prepared_session.side_effect = RuntimeError('receipt lost')
    with pytest.raises(BackendUnavailable):
        await visit.cleanup()
    assert 'TEAM_WORKER_DISCONNECTED' in visit.error
    assert 'cleanup unconfirmed' in visit.error
    restored = EndpointPreparation(adapter, visit.handles, AsyncMock())
    restored.restore_lifecycle(visit.lifecycle_checkpoint())
    adapter.end_prepared_session.side_effect = None
    await restored.cleanup()
    assert restored.closure_complete
    assert restored.error == 'TEAM_WORKER_DISCONNECTED'
    assert not restored.cleanup_error
