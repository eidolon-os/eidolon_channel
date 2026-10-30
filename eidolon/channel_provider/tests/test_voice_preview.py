import asyncio
import base64
from unittest.mock import AsyncMock
import pytest
from eidolon_sdk.biz.control.voice_preview import VoicePreviewRequest
from eidolon.channel_provider.voice_preview import VoicePreviewCache

PCM = b'\x00\x01' * 24000


def cache(path, synthesize, profile=None, **kwargs):
    return VoicePreviewCache(path, profile=profile or (lambda: {'voice': 'configured', 'revision': 1}),
                             synthesize=synthesize, **kwargs)


async def settle(service):
    await asyncio.gather(*list(service._tasks.values()))


async def test_lazy_coalesced_owner_scoped_and_restart_safe(tmp_path):
    synthesize = AsyncMock(return_value=PCM)
    service = cache(tmp_path, synthesize)
    synthesize.assert_not_called()
    requests = [VoicePreviewRequest(request_id='one'), VoicePreviewRequest(request_id='two')]
    results = await asyncio.gather(*(service.request('owner', r) for r in requests))
    assert all(r.state == 'generating' for r in results)
    await settle(service)
    assert synthesize.await_count == 1
    for request in requests:
        ready = await service.request('owner', request)
        assert ready.state == 'ready' and ready.audio_available
    assert base64.b64decode(service.audio('owner', ready.key)['pcm_base64']) == PCM
    with pytest.raises(KeyError):
        service.audio('other', ready.key)
    restored = cache(tmp_path, synthesize)
    assert (await restored.request('owner', VoicePreviewRequest(request_id='after-restart'))).state == 'ready'
    assert synthesize.await_count == 1
    await restored.request('other', requests[0])
    await settle(restored)
    assert synthesize.await_count == 2


async def test_regeneration_failure_keeps_previous_and_same_operation_does_not_retry(tmp_path):
    synthesize = AsyncMock(side_effect=[PCM, RuntimeError('provider refused'), PCM])
    service = cache(tmp_path, synthesize)
    first = await service.request('owner', VoicePreviewRequest(request_id='first'))
    await settle(service)
    retry = VoicePreviewRequest(request_id='regenerate', regenerate=True)
    await service.request('owner', retry)
    await settle(service)
    status = await service.request('owner', retry)
    assert status.state == 'failed' and status.audio_available
    assert base64.b64decode(service.audio('owner', first.key)['pcm_base64']) == PCM
    assert synthesize.await_count == 2
    await service.request('owner', VoicePreviewRequest(request_id='regenerate-again', regenerate=True))
    await settle(service)
    assert synthesize.await_count == 3


async def test_voice_change_invalidates_download_and_empty_audio_never_caches(tmp_path):
    profile = {'voice': 'one'}
    synthesize = AsyncMock(side_effect=[PCM, b''])
    service = cache(tmp_path, synthesize, profile=lambda: profile)
    first = await service.request('owner', VoicePreviewRequest(request_id='first'))
    await settle(service)
    profile['voice'] = 'two'
    with pytest.raises(KeyError):
        service.audio('owner', first.key)
    request = VoicePreviewRequest(request_id='second')
    await service.request('owner', request)
    await settle(service)
    failed = await service.request('owner', request)
    assert failed.state == 'failed' and not failed.audio_available
    assert len(list(tmp_path.glob('*.json'))) == 1


async def test_cache_capacity_evicts_completed_samples(tmp_path):
    service = cache(tmp_path, AsyncMock(return_value=PCM), max_bytes=65000)
    first = await service.request('first', VoicePreviewRequest(request_id='a'))
    await settle(service)
    await service.request('second', VoicePreviewRequest(request_id='b'))
    await settle(service)
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 65000
    with pytest.raises(KeyError):
        service.audio('first', first.key)
