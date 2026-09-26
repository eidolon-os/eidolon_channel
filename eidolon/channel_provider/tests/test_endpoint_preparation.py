import asyncio
from unittest.mock import AsyncMock

import pytest
from eidolon.channel_provider.contracts import InvalidTransition
from eidolon.channel_provider.endpoint_preparation import EndpointPreparation
from eidolon.channel_provider.ports import ServingAction, ServingRequest


class Adapter:
    def __init__(self):
        self.launched = asyncio.Event()
        self.sent = []
        self.require_idle = AsyncMock()
        self.end_prepared_session = AsyncMock()
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
    with pytest.raises(ExceptionGroup):
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
