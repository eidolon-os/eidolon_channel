import asyncio
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError
from eidolon_sdk.biz.control.device_session import DeviceSessionSelection
from eidolon_sdk.biz.control.device_conversation import DeviceConversationSelection
from eidolon.channel_provider.contracts import Forbidden
from .test_device_conversation import prepare, wait_for

async def test_two_cycles_independent_catalog_and_one_endpoint(tmp_path):
    service, backend, directed = await prepare(tmp_path)
    async def activate(*args, **kwargs):
        backend.launched.set()
    backend.open_device_session = AsyncMock(side_effect=activate)
    for index in range(2):
        backend.launched.clear()
        selection = DeviceSessionSelection(session_id=f'single-{index}',
            device=directed.input_device, target_companion_id='selected')
        with pytest.raises(Forbidden):
            await service.open_device_session(selection, authenticated_owner_id='other')
        result = await service.open_device_session(selection, authenticated_owner_id='owner_1')
        assert result['state'] == 'preparing'
        visit = service._device_sessions[('owner_1', selection.session_id)]
        await wait_for(lambda: visit.state == 'ready')
        await service.open_device_session(selection, authenticated_owner_id='owner_1')
        assert len(backend.sent) == index + 1
        assert not service._device_conversations
        closed = await service.device_session(selection.session_id,
            authenticated_owner_id='owner_1', close=True)
        assert closed['state'] == 'closed'
        assert not service._transport_scopes
        assert not backend.observers
    assert backend.open_device_session.await_count == 2
    assert backend.end_prepared_session.await_count == 2
    assert not backend.directed
    await service.shutdown()

async def test_directed_contract_still_requires_two_devices(tmp_path):
    service, _, directed = await prepare(tmp_path)
    with pytest.raises(ValidationError, match='DISTINCT_ENDPOINTS_REQUIRED'):
        DeviceConversationSelection(session_id='bad', input_device=directed.input_device,
            output_device=directed.input_device, target_companion_id='selected')
    await service.shutdown()

async def test_close_before_open_fences_only_single_device_entry(tmp_path):
    service, backend, directed = await prepare(tmp_path)
    result = await service.device_session('cancel', authenticated_owner_id='owner_1', close=True)
    assert result == dict(session_id='cancel', state='closed', error='')
    assert not service._store.scene_cancelled('owner_1', 'cancel', 'device_conversation')
    await service.shutdown()
