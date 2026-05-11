# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for :class:`DuckingMixer`.

Tests verify:
  - Fade-out ramp during duck (frames forwarded with gain ramp)
  - Buffering after fade-out completes (frames NOT forwarded)
  - Drain on unduck (buffered frames replayed with fade-in)
  - Cancel discards buffer
  - Idempotency / re-entry safety
  - Metrics counters
"""

from __future__ import annotations

import asyncio
from typing import List

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice import io as lk_io

from eidolon.livekit.agent.ducking import DuckingMixer

SAMPLE_RATE = 32000
FRAME_MS = 10  # standard livekit frame duration
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 320


class _FakeInnerOutput(lk_io.AudioOutput):
    """Records every frame it receives. Used as the next_in_chain."""

    def __init__(self) -> None:
        super().__init__(
            label="FakeInner",
            capabilities=lk_io.AudioOutputCapabilities(pause=False),
            sample_rate=SAMPLE_RATE,
        )
        self.frames: List[rtc.AudioFrame] = []
        self.flushed = 0
        self.cleared = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames.append(frame)

    def flush(self) -> None:
        super().flush()
        self.flushed += 1

    def clear_buffer(self) -> None:
        self.cleared += 1


def _make_frame(value: int = 10000) -> rtc.AudioFrame:
    """Frame of ``SAMPLES_PER_FRAME`` int16 samples all == value."""
    arr = np.full(SAMPLES_PER_FRAME, value, dtype=np.int16)
    return rtc.AudioFrame(
        data=arr.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


def _samples_of(frame: rtc.AudioFrame) -> np.ndarray:
    return np.frombuffer(frame.data, dtype=np.int16)


# ──────────────────────────────────────────────────────────────────
# Construction & defaults
# ──────────────────────────────────────────────────────────────────


def test_starts_in_normal_state_full_volume() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    assert mixer.state == "NORMAL"
    assert mixer.current_volume == 1.0


def test_label_includes_inner_label() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner)
    assert "FakeInner" in mixer.label


# ──────────────────────────────────────────────────────────────────
# NORMAL pass-through
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_normal_state_passes_frames_through_unchanged() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner)
    f = _make_frame(value=12345)
    await mixer.capture_frame(f)
    assert len(inner.frames) == 1
    assert (_samples_of(inner.frames[0]) == 12345).all()


# ──────────────────────────────────────────────────────────────────
# duck() → fade-out → SUSPENDED → buffer
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duck_fades_out_then_buffers() -> None:
    inner = _FakeInnerOutput()
    # 50 ms fade @ 32 kHz = 1600 samples = 5 frames of 320 samples each
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.duck()
    assert mixer.state == "SUSPENDED"

    # Push 6 frames: first 5 carry the fade-out ramp, 6th should be buffered.
    for _ in range(6):
        await mixer.capture_frame(_make_frame(value=10000))

    # 5 frames forwarded (fade-out ramp), 1 frame buffered.
    assert len(inner.frames) == 5
    assert mixer.buffered_frames == 1

    # First forwarded frame: starts near 1.0, decreasing.
    first = _samples_of(inner.frames[0]).astype(np.float32)
    assert first[0] > 9000
    assert first[-1] < first[0]

    # Last forwarded frame: ends at silence.
    last = _samples_of(inner.frames[-1])
    assert last[-1] == 0


@pytest.mark.asyncio
async def test_duck_with_partial_suspend_volume_then_buffers() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, suspend_volume=0.25)
    mixer.duck()
    # 10 ms fade @ 32 kHz = 320 samples = exactly one frame (ramp)
    await mixer.capture_frame(_make_frame(10000))
    # Ramp done. Next frame goes to buffer (not forwarded at 0.25).
    await mixer.capture_frame(_make_frame(10000))

    assert len(inner.frames) == 1  # only the ramp frame forwarded
    assert mixer.buffered_frames == 1


# ──────────────────────────────────────────────────────────────────
# unduck() → drain buffer → fade-in → NORMAL
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unduck_drains_buffer_with_fade_in() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)
    mixer.duck()
    # Drain the fade-out ramp (1 frame).
    await mixer.capture_frame(_make_frame(10000))
    # Buffer 3 frames while suspended.
    for _ in range(3):
        await mixer.capture_frame(_make_frame(10000))
    assert mixer.buffered_frames == 3
    forwarded_before = len(inner.frames)
    inner.frames.clear()

    mixer.unduck()
    assert mixer.state == "NORMAL"
    # Push a new frame — this triggers drain of buffered frames first.
    await mixer.capture_frame(_make_frame(10000))

    # 3 buffered frames + 1 new frame = 4 forwarded.
    assert len(inner.frames) == 4
    assert mixer.buffered_frames == 0

    # First drained frame: fade-in ramp from 0 → 1.
    first = _samples_of(inner.frames[0]).astype(np.float32)
    assert first[0] < 1000  # near silence at start
    assert first[-1] > 8000  # near full vol by end

    # Last frame (live): full volume.
    last = _samples_of(inner.frames[-1])
    assert (last == 10000).all()


@pytest.mark.asyncio
async def test_unduck_resumes_full_volume_after_fade() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)
    mixer.duck()
    # Drain fade-out ramp.
    await mixer.capture_frame(_make_frame(10000))
    inner.frames.clear()

    mixer.unduck()
    assert mixer.state == "NORMAL"
    # Push 2 frames to drain empty buffer + play live.
    await mixer.capture_frame(_make_frame(10000))
    await mixer.capture_frame(_make_frame(10000))

    # First frame: ramping up from 0 → 1 (fade-in on live frame).
    first = _samples_of(inner.frames[0]).astype(np.float32)
    assert first[0] < 1000
    assert first[-1] > 8000

    # Second frame: fully NORMAL, full pass-through.
    second = _samples_of(inner.frames[1])
    assert (second == 10000).all()


@pytest.mark.asyncio
async def test_buffer_content_preserved_on_drain() -> None:
    """Buffered frames contain the original TTS content, not silence."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)
    mixer.duck()
    # Drain fade-out.
    await mixer.capture_frame(_make_frame(10000))
    inner.frames.clear()

    # Buffer frames with distinct values.
    await mixer.capture_frame(_make_frame(5000))
    await mixer.capture_frame(_make_frame(6000))
    await mixer.capture_frame(_make_frame(7000))
    assert mixer.buffered_frames == 3

    mixer.unduck()
    # Trigger drain.
    await mixer.capture_frame(_make_frame(8000))

    # 3 buffered + 1 live = 4 frames.
    assert len(inner.frames) == 4
    # After fade-in ramp completes (320 samples @ 10ms), subsequent
    # frames should have the original sample values (with gain applied).
    # The last live frame (8000) should be at full volume.
    last = _samples_of(inner.frames[-1])
    assert (last == 8000).all()


# ──────────────────────────────────────────────────────────────────
# cancel() → CANCELLED → drop buffered + live frames
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_drops_buffered_and_subsequent_frames() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # fade-out
    # Buffer 2 frames.
    await mixer.capture_frame(_make_frame(10000))
    await mixer.capture_frame(_make_frame(10000))
    assert mixer.buffered_frames == 2
    inner.frames.clear()

    mixer.cancel()
    assert mixer.state == "CANCELLED"
    assert inner.cleared == 1
    assert mixer.buffered_frames == 0

    # Subsequent frames are dropped.
    for _ in range(3):
        await mixer.capture_frame(_make_frame(10000))
    assert inner.frames == []


@pytest.mark.asyncio
async def test_cancel_from_normal_state_also_drops() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.cancel()
    await mixer.capture_frame(_make_frame(10000))
    assert inner.frames == []
    assert inner.cleared == 1


# ──────────────────────────────────────────────────────────────────
# Buffer limits
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_buffer_respects_max_limit() -> None:
    inner = _FakeInnerOutput()
    # buffer_max_sec=0.02 → at 32kHz with 320 samples/frame, ~2 frames max.
    mixer = DuckingMixer(inner, fade_ms=10, buffer_max_sec=0.02)
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # fade-out ramp

    # Try to buffer 5 frames — should cap.
    for _ in range(5):
        await mixer.capture_frame(_make_frame(10000))

    assert mixer.buffered_frames <= 2


# ──────────────────────────────────────────────────────────────────
# Idempotency / re-entry
# ──────────────────────────────────────────────────────────────────


def test_duck_then_duck_is_idempotent() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.duck()
    mixer.duck()
    assert mixer.state == "SUSPENDED"


def test_unduck_when_already_normal_no_throw() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.unduck()  # already normal
    assert mixer.state == "NORMAL"


def test_unduck_from_cancelled_is_noop() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.cancel()
    mixer.unduck()
    assert mixer.state == "CANCELLED"


@pytest.mark.asyncio
async def test_duck_during_unduck_reverses_smoothly() -> None:
    """User starts speaking → duck → false-pos → unduck → speaks again →
    duck again. The mid-ramp reversal must not click."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # mid-fade-out
    mixer.unduck()
    await mixer.capture_frame(_make_frame(10000))  # mid-fade-in (reverse)
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # mid-fade-out again

    # No exceptions, frames forwarded during ramp phases.
    assert len(inner.frames) == 3


# ──────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_track_buffer_operations() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10, fade_in_ms=10)

    # Duck → buffer → unduck → drain.
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # fade-out
    await mixer.capture_frame(_make_frame(10000))  # buffered
    await mixer.capture_frame(_make_frame(10000))  # buffered
    mixer.unduck()
    await mixer.capture_frame(_make_frame(10000))  # triggers drain + live

    m = mixer.get_metrics()
    assert m["total_ducks"] == 1
    assert m["total_unducks"] == 1
    assert m["total_cancels"] == 0
    assert m["total_buffer_drains"] == 1
    assert m["total_buffer_frames_drained"] == 2

    # Duck → buffer → cancel → discard.
    mixer.reset()
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))  # fade-out
    await mixer.capture_frame(_make_frame(10000))  # buffered
    mixer.cancel()

    m = mixer.get_metrics()
    assert m["total_cancels"] == 1
    assert m["total_buffer_frames_dropped"] == 1


# ──────────────────────────────────────────────────────────────────
# Misc plumbing
# ──────────────────────────────────────────────────────────────────


def test_flush_propagates() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.flush()
    assert inner.flushed == 1


def test_clear_buffer_propagates() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=50)
    mixer.clear_buffer()
    assert inner.cleared == 1


def test_reset_returns_to_normal_full_volume() -> None:
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)
    mixer.duck()
    mixer.reset()
    assert mixer.state == "NORMAL"
    assert mixer.current_volume == 1.0
    assert mixer.buffered_frames == 0


@pytest.mark.asyncio
async def test_ramp_progresses_per_sample_not_per_frame() -> None:
    """Sanity: a single 10 ms fade with frame size 10 ms should produce
    a within-frame gradient (not a flat 0.5)."""
    inner = _FakeInnerOutput()
    mixer = DuckingMixer(inner, fade_ms=10)
    mixer.duck()
    await mixer.capture_frame(_make_frame(10000))

    samples = _samples_of(inner.frames[0]).astype(np.float32)
    # First sample close to 10000 (full vol), last close to 0 (silence)
    assert samples[0] > 8000
    assert samples[-1] < 200
    # Mean should be ~5000 (linear ramp)
    assert 4000 < samples.mean() < 6000
