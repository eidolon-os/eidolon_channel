import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from eidolon.channel_provider.voice_capture import VoiceCapture, Capture, register_voice_capture


async def test_owner_scope_and_single_consumption():
    capture = VoiceCapture(None, lambda: [])
    session = Capture('owner-a', SimpleNamespace(disconnect=AsyncMock()), 'mic')
    session.pcm.extend(bytes(32000 * 3))
    capture.sessions['key'] = session
    with pytest.raises(web.HTTPNotFound):
        await capture.finish('owner-b', 'key')
    assert len(session.pcm) == 96000
    result = await capture.finish('owner-a', 'key')
    assert len(result['pcm_base64']) == 128000
    assert not session.pcm and not capture.sessions
    session.room.disconnect.assert_awaited_once()
    with pytest.raises(web.HTTPNotFound):
        await capture.finish('owner-a', 'key')


async def test_rejects_short_or_failed_audio_and_cleans_up():
    for failed in (True, False):
        capture = VoiceCapture(None, lambda: [])
        session = Capture('owner', SimpleNamespace(disconnect=AsyncMock()), 'mic', failed=failed)
        session.pcm.extend(bytes(96000 if failed else 100))
        capture.sessions['key'] = session
        with pytest.raises(web.HTTPUnprocessableEntity):
            await capture.finish('owner', 'key')
        assert not capture.sessions and not session.pcm


async def test_owner_concurrency_limit_does_not_create_room():
    capture = VoiceCapture(None, lambda: [])
    capture.sessions['key'] = Capture('owner', None, 'mic')
    with pytest.raises(web.HTTPConflict):
        await capture.start('owner')


async def test_route_requires_service_auth_and_binds_owner():
    capture = SimpleNamespace(start=AsyncMock(return_value={'capture_id':'id'}), close=AsyncMock())
    app = web.Application()
    register_voice_capture(app, capture=capture,
        authorized=lambda request: request.headers.get('Authorization') == 'Bearer test')
    async with TestClient(TestServer(app)) as client:
        r = await client.post('/v1/voiceprints/captures', json={'owner_id':'owner'})
        assert r.status == 401
        capture.start.assert_not_awaited()
        r = await client.post('/v1/voiceprints/captures', json={'owner_id':'owner'},
                              headers={'Authorization':'Bearer test'})
        assert r.status == 200 and r.headers['Cache-Control'] == 'no-store'
        capture.start.assert_awaited_once_with('owner')


async def test_cancelled_finish_releases_pcm_and_connection():
    capture = VoiceCapture(None, lambda: [])
    session = Capture('owner', SimpleNamespace(disconnect=AsyncMock()), 'mic')
    session.pcm.extend(bytes(96000))
    capture.sessions['key'] = session
    task = asyncio.create_task(capture.finish('owner', 'key'))
    await asyncio.sleep(0)
    with pytest.raises(web.HTTPConflict):
        await capture.finish('owner', 'key')
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert not session.pcm and not capture.sessions
    session.room.disconnect.assert_awaited_once()
