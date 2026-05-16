"""G6: ``DuckingMixer.played_seconds`` tracks audio actually forwarded to
the inner sink, used by ``_snapshot_interrupted_context`` to enrich the
LLM context with "how much the user heard before the cancel".

Counter resets on each ``duck()`` (= new user-speech window, marking a
potential turn boundary). Buffered-during-SUSPENDED frames do NOT count
(user heard silence for those).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit import rtc
from livekit.agents.voice import io as lk_io

from eidolon.livekit.agent.ducking import DuckingMixer


def _silent_frame(samples: int = 800, sample_rate: int = 16000) -> rtc.AudioFrame:
    """Silent 16-bit mono frame."""
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class _StubInnerSink(lk_io.AudioOutput):
    """Minimal AudioOutput that just records the count of frames captured."""

    def __init__(self, sample_rate: int = 16000) -> None:
        super().__init__(
            label="stub-inner",
            capabilities=lk_io.AudioOutputCapabilities(pause=True),
            sample_rate=sample_rate,
        )
        self.captured: int = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.captured += 1

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


@pytest.mark.asyncio
async def test_played_seconds_starts_at_zero() -> None:
    inner = _StubInnerSink()
    mixer = DuckingMixer(inner, sample_rate=16000)
    assert mixer.played_seconds == 0.0


@pytest.mark.asyncio
async def test_played_seconds_counts_normal_flow() -> None:
    """Frames forwarded in NORMAL state count toward played_seconds."""
    inner = _StubInnerSink()
    mixer = DuckingMixer(inner, sample_rate=16000)
    # 5 frames × 800 samples / 16000 Hz = 0.25s total
    for _ in range(5):
        await mixer.capture_frame(_silent_frame(800))
    assert mixer.played_seconds == pytest.approx(0.25, abs=1e-3)


@pytest.mark.asyncio
async def test_played_seconds_resets_on_duck() -> None:
    """Each duck() marks a turn boundary; counter must reset so the value
    reflects "this turn's audio" not "session-cumulative"."""
    inner = _StubInnerSink()
    mixer = DuckingMixer(inner, sample_rate=16000)
    for _ in range(3):
        await mixer.capture_frame(_silent_frame(800))
    assert mixer.played_seconds > 0
    mixer.duck()
    assert mixer.played_seconds == 0.0


@pytest.mark.asyncio
async def test_played_seconds_excludes_buffered_suspended_frames() -> None:
    """During SUSPENDED phase 2 (after fade-out ramp), incoming frames are
    BUFFERED, not played — they must not bump played_seconds."""
    inner = _StubInnerSink()
    # Short fade so the test doesn't have to push many frames to exit phase 1
    mixer = DuckingMixer(inner, fade_ms=10, sample_rate=16000)
    mixer.duck()
    # First few frames go through the fade-out ramp (each is 50ms). Push
    # enough to exhaust the 10ms ramp on the first frame, then push more
    # which should be buffered (silent to user).
    for _ in range(5):
        await mixer.capture_frame(_silent_frame(800))  # 50ms each
    # Ramp consumed ~10ms inside the first frame; that frame WAS forwarded.
    # Frames 2-5 should be in buffer (NOT played).
    assert 0.04 < mixer.played_seconds < 0.06, (
        f"played_seconds={mixer.played_seconds}: expected only fade-out "
        f"chunk (~50ms) to count, buffered frames must not bump the counter"
    )
    assert mixer.buffered_frames > 0, "later frames should be buffered"


@pytest.mark.asyncio
async def test_played_seconds_includes_drained_buffer_on_unduck() -> None:
    """When the mixer unducks and drains the buffer through the inner sink
    (with fade-in), those frames DO count — the user hears them, just at
    ramp-up volume."""
    inner = _StubInnerSink()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10, sample_rate=16000)
    mixer.duck()
    for _ in range(3):
        await mixer.capture_frame(_silent_frame(800))  # 1 fade-out + 2 buffered
    before_unduck = mixer.played_seconds
    mixer.unduck()
    # Next frame triggers drain
    await mixer.capture_frame(_silent_frame(800))
    after_drain = mixer.played_seconds
    assert after_drain > before_unduck, (
        f"played_seconds should grow after drain: before={before_unduck} "
        f"after={after_drain}"
    )
