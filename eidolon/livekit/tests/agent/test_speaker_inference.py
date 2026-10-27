import asyncio
import threading

import pytest

from eidolon.livekit.plugins.speaker_verification.inference import (
    InferenceBusy, InferenceTimeout, ModelUnavailable, SpeakerInferenceEngine,
)


class Extractor:
    def __init__(self):
        self.warms = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def warm_up(self):
        self.warms += 1

    def extract(self, pcm, min_speech_ms=1500):
        self.entered.set()
        assert self.release.wait(2)
        return pcm


async def test_readiness_requires_warmup_and_warmup_is_shared():
    extractor = Extractor()
    engine = SpeakerInferenceEngine(extractor)
    try:
        with pytest.raises(ModelUnavailable):
            await engine.extract(b'pcm')
        await asyncio.gather(engine.start(), engine.start())
        assert extractor.warms == 1
        assert engine.state == 'ready'
    finally:
        await engine.close()
    with pytest.raises(ModelUnavailable):
        await engine.extract(b'pcm')


async def test_cancelled_caller_does_not_release_running_inference_capacity():
    extractor = Extractor()
    engine = SpeakerInferenceEngine(extractor, max_pending=1)
    await engine.start()
    task = asyncio.create_task(engine.extract(b'first'))
    try:
        assert await asyncio.to_thread(extractor.entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(InferenceBusy):
            await engine.extract(b'second')
    finally:
        extractor.release.set()
        await engine.close()


async def test_timeout_does_not_queue_unbounded_work():
    extractor = Extractor()
    engine = SpeakerInferenceEngine(extractor, max_pending=1, timeout_seconds=.01)
    await engine.start()
    try:
        with pytest.raises(InferenceTimeout):
            await engine.extract(b'first')
        with pytest.raises(InferenceBusy):
            await engine.extract(b'second')
    finally:
        extractor.release.set()
        await engine.close()


async def test_failed_model_never_reports_ready_or_leaks_exception_detail():
    class Broken(Extractor):
        def warm_up(self):
            raise RuntimeError('private filesystem path or credential')

    engine = SpeakerInferenceEngine(Broken())
    try:
        with pytest.raises(ModelUnavailable, match='model_load_failed'):
            await engine.start()
        assert engine.state == 'unavailable'
        with pytest.raises(ModelUnavailable):
            await engine.extract(b'pcm')
    finally:
        await engine.close()


async def test_successful_request_releases_capacity_without_blocking_event_loop():
    extractor = Extractor()
    extractor.release.set()
    engine = SpeakerInferenceEngine(extractor, max_pending=1)
    await engine.start()
    try:
        assert await engine.extract(b'one') == b'one'
        assert await engine.extract(b'two') == b'two'
    finally:
        await engine.close()
