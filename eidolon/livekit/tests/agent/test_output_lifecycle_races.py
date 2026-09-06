"""Audio-sink integration regressions: sequence and cancellation under backpressure."""
import asyncio

import pytest

from .test_ducking_mixer import DuckingMixer, _FakeInnerOutput, _make_frame, _samples_of


class BlockingOutput(_FakeInnerOutput):
    def __init__(self):
        super().__init__()
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def capture_frame(self, frame):
        if self.block:
            self.entered.set()
            await self.release.wait()
        await super().capture_frame(frame)


async def suspended(inner=None, **kwargs):
    inner = inner or _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10, **kwargs)
    mixer.duck()
    await mixer.capture_frame(_make_frame())
    inner.frames.clear()
    return mixer, inner


async def settle(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_resume_drains_finished_tts_tail_before_flush():
    mixer, inner = await suspended()
    await mixer.capture_frame(_make_frame(5000))
    await mixer.capture_frame(_make_frame(6000))
    mixer.flush()
    assert inner.flushed == 0, 'held audio must not report playback finished'
    mixer.unduck()
    await settle(lambda: inner.flushed == 1)
    assert len(inner.frames) == 2
    assert _samples_of(inner.frames[-1])[-1] == 6000


@pytest.mark.asyncio
async def test_duplicate_duck_preserves_unheard_content():
    mixer, inner = await suspended()
    await mixer.capture_frame(_make_frame(5000))
    mixer.duck()
    assert mixer.buffered_frames == 1
    mixer.unduck()
    await settle(lambda: len(inner.frames) == 1)


@pytest.mark.asyncio
async def test_full_buffer_backpressures_instead_of_losing_words():
    mixer, inner = await suspended(buffer_max_sec=0.02)
    await mixer.capture_frame(_make_frame(1000))
    await mixer.capture_frame(_make_frame(2000))
    producer = asyncio.create_task(mixer.capture_frame(_make_frame(3000)))
    await asyncio.sleep(0.01)
    try:
        assert not producer.done(), 'capacity must pause synthesis, not drop audio'
        assert mixer.buffered_frames == 2
        mixer.unduck()
        await asyncio.wait_for(producer, 1)
        assert [_samples_of(f)[-1] for f in inner.frames] == [1000, 2000, 3000]
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['cancel', 'clear_buffer', 'reset'])
async def test_inflight_old_audio_cannot_escape_after_invalidation(operation):
    inner = BlockingOutput()
    mixer, _ = await suspended(inner)
    await mixer.capture_frame(_make_frame(1000))
    await mixer.capture_frame(_make_frame(2000))
    inner.block = True
    mixer.unduck()
    producer = asyncio.create_task(mixer.capture_frame(_make_frame(3000)))
    await asyncio.wait_for(inner.entered.wait(), 1)
    getattr(mixer, operation)()
    inner.release.set()
    await asyncio.wait_for(producer, 1)
    assert inner.frames == [], 'old generation leaked after sink invalidation'


@pytest.mark.asyncio
async def test_resuspend_during_drain_keeps_remaining_frames_in_order():
    inner = BlockingOutput()
    mixer, _ = await suspended(inner)
    await mixer.capture_frame(_make_frame(1000))
    await mixer.capture_frame(_make_frame(2000))
    inner.block = True
    mixer.unduck()
    await asyncio.wait_for(inner.entered.wait(), 1)
    mixer.duck()
    inner.release.set()
    await settle(lambda: len(inner.frames) == 1)
    assert mixer.buffered_frames == 1
    mixer.unduck()
    await settle(lambda: len(inner.frames) == 2)
    assert [_samples_of(f)[-1] for f in inner.frames] == [1000, 2000]


@pytest.mark.asyncio
async def test_failed_tail_drain_terminates_playback_waiters():
    from .._harness.headless import RecordingAudioOutput

    class FailingOutput(RecordingAudioOutput):
        fail = False

        async def capture_frame(self, frame):
            if self.fail:
                raise RuntimeError('sink disconnected')
            await super().capture_frame(frame)

    inner = FailingOutput(sample_rate=32000)
    mixer = DuckingMixer(inner, fade_ms=10)
    mixer.duck()
    await mixer.capture_frame(_make_frame())
    await mixer.capture_frame(_make_frame())
    mixer.flush()
    inner.fail = True
    mixer.unduck()
    finished = await asyncio.wait_for(mixer.wait_for_playout(), 1)
    assert finished.interrupted
    assert mixer.state == 'CANCELLED'
    assert mixer.buffered_frames == 0
