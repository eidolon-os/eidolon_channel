"""Explicit content-discard control and ordinary resume behavior.

The discard option was originally used on timeout to skip held TTS content.
A timeout now preserves that unheard content; discarding it merely skipped
part of the sentence while later audio still resumed. These tests retain
coverage of the explicit output control, separate from timeout policy.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice import io as lk_io

from eidolon.livekit.agent.output.controller import OutputController as DuckingMixer


SAMPLE_RATE = 32000
SAMPLES_PER_FRAME = 320  # 10ms @ 32kHz


class _FakeInnerOutput(lk_io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="FakeInner",
            capabilities=lk_io.AudioOutputCapabilities(pause=False),
            sample_rate=SAMPLE_RATE,
        )
        self.frames: List[rtc.AudioFrame] = []

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames.append(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


def _frame(value: int = 10000) -> rtc.AudioFrame:
    arr = np.full(SAMPLES_PER_FRAME, value, dtype=np.int16)
    return rtc.AudioFrame(
        data=arr.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


# ───────────────────────────────────────────────────────────────────────
# G17a: unduck(drop_buffered=True) — the new opt-in behaviour
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unduck_drop_buffered_discards_instead_of_drains() -> None:
    """Core regression: ``unduck(drop_buffered=True)`` must NOT replay
    buffered frames on the next capture_frame call."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)

    mixer.duck()
    await mixer.capture_frame(_frame(10000))  # fade-out, forwarded
    # Buffer 3 frames distinct from any subsequent live frame.
    await mixer.capture_frame(_frame(1111))
    await mixer.capture_frame(_frame(2222))
    await mixer.capture_frame(_frame(3333))
    assert mixer.buffered_frames == 3

    forwarded_before = len(inner.frames)

    mixer.unduck(drop_buffered=True)
    assert mixer.state == "NORMAL"
    assert mixer.buffered_frames == 0  # buffer cleared at unduck time

    # Push a live frame. Pre-fix bug: this would replay the 3 buffered
    # frames first. Post-fix: only the live frame is forwarded.
    await mixer.capture_frame(_frame(9999))

    new_frames = inner.frames[forwarded_before:]
    assert len(new_frames) == 1, (
        "Expected only the live frame; stale buffer must not drain"
    )
    # No stale values in the forwarded stream.
    assert (np.frombuffer(new_frames[0].data, dtype=np.int16) != 1111).any()


@pytest.mark.asyncio
async def test_unduck_default_still_drains() -> None:
    """Backwards-compat: ``unduck()`` without args drains as before.

    The fast-path soft-unduck (user-silent transition within ~300ms)
    and the timeout rollback both rely on this default behaviour.
    """
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)

    mixer.duck()
    await mixer.capture_frame(_frame(10000))  # fade-out
    await mixer.capture_frame(_frame(1111))
    await mixer.capture_frame(_frame(2222))
    assert mixer.buffered_frames == 2

    forwarded_before = len(inner.frames)

    mixer.unduck()  # default drop_buffered=False
    await mixer.capture_frame(_frame(9999))  # triggers drain + live

    new_frames = inner.frames[forwarded_before:]
    # 2 drained + 1 live = 3 frames in the forwarded stream
    assert len(new_frames) == 3


@pytest.mark.asyncio
async def test_unduck_drop_buffered_metrics_track() -> None:
    """G17a observability — ``total_buffer_frames_dropped_on_unduck``
    increments when we choose drop_buffered=True so we can monitor the
    rate of "stale buffer" decisions in production."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)

    mixer.duck()
    await mixer.capture_frame(_frame(10000))  # fade-out
    await mixer.capture_frame(_frame(1111))
    await mixer.capture_frame(_frame(2222))

    m0 = mixer.get_metrics()
    assert m0["total_buffer_frames_dropped_on_unduck"] == 0

    mixer.unduck(drop_buffered=True)

    m1 = mixer.get_metrics()
    assert m1["total_buffer_frames_dropped_on_unduck"] == 2


@pytest.mark.asyncio
async def test_unduck_drop_buffered_with_empty_buffer_is_noop() -> None:
    """Edge: if user spoke for ≥0.8s but TTS finished/paused so no frames
    accumulated, drop_buffered=True is a harmless no-op (no metric jump)."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)

    mixer.duck()
    await mixer.capture_frame(_frame(10000))  # only fade-out frame, no buffering
    assert mixer.buffered_frames == 0

    mixer.unduck(drop_buffered=True)

    m = mixer.get_metrics()
    assert m["total_buffer_frames_dropped_on_unduck"] == 0
    assert mixer.state == "NORMAL"


@pytest.mark.asyncio
async def test_unduck_drop_buffered_preserves_fadein_for_live_frames() -> None:
    """Even with the buffer dropped, the fade-in ramp must still apply to
    live frames arriving after unduck. (Avoid pop/click on resume.)"""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)

    mixer.duck()
    await mixer.capture_frame(_frame(10000))  # fade-out
    await mixer.capture_frame(_frame(1111))  # buffered
    inner.frames.clear()

    mixer.unduck(drop_buffered=True)

    # First live frame should ramp from near-silence up to full volume.
    await mixer.capture_frame(_frame(10000))
    samples = np.frombuffer(inner.frames[0].data, dtype=np.int16).astype(np.float32)
    assert samples[0] < 1000, f"expected silence at ramp start; got {samples[0]}"
    assert samples[-1] > 8000, f"expected near-full at ramp end; got {samples[-1]}"
