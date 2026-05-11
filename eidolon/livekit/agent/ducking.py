# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Soft-cancel + soft-resume audio output (the "ducking mixer").

============================================================================
WHY
============================================================================

When the user interrupts the agent, two reactions are possible:

  1. **Hard cancel** — abruptly stop TTS and silence playback. Fast (~50 ms)
     but acoustically jarring; if the interrupt turns out to be a false
     positive (backchannel "嗯", echo, noise), framework's
     ``resume_false_interruption`` path needs to hard-restart playback
     from a partially-rendered TTS chunk, which sounds like a stutter or
     a re-synthesis seam.

  2. **Soft cancel + soft resume** (this module) — on VAD start, fade the
     output volume down to silence over ~50 ms, then **buffer** TTS frames
     in memory. While suspended, an "early-resume watcher" in
     :class:`StreamingPipeline` listens to STT interim and decides:

       * confirm cancel (real interrupt)  → call :meth:`cancel`; buffered
         frames are discarded, downstream receives ``clear_buffer()``,
         and our :meth:`capture_frame` drops subsequent frames.

       * confirm false (backchannel/noise) → call :meth:`unduck`; buffered
         frames are drained through the inner sink with a fade-in ramp,
         so the user hears the agent's speech **from the point it was
         suspended**, with no content loss and no seam.

============================================================================
FRAME BUFFERING (pause / resume at our layer)
============================================================================

During SUSPENDED the mixer operates in two sub-phases:

  Phase 1 — FADE-OUT (first ``duck_fade_ms`` worth of frames):
    Frames are attenuated via per-sample linear ramp (1.0 → suspend_volume)
    and forwarded to the inner sink. The user hears the agent "yielding".

  Phase 2 — BUFFERING (after ramp completes):
    Frames are stored in ``_buffer`` and NOT forwarded. The user hears
    silence. TTS continues generating in the background.

On unduck, the buffered frames are drained through the inner sink. The
first ``duck_fade_in_ms`` of drained audio has a fade-in ramp applied
for a smooth transition. LiveKit's ``AudioSource`` has internal queue
pacing (``queue_size_ms=200``, backpressure via ``_q_size``), so burst-
pushing buffered frames does NOT cause fast-forward — the downstream
pipeline paces them at wall-clock speed.

On cancel, the buffer is discarded.

============================================================================
ALL TUNABLES LIVE IN ``EidolonEOTConfig``
============================================================================

See ``plugins/eot/config.py`` for the full block under
"Ducking mixer + early-resume watcher". Quick reference:

  * ``duck_enabled``                         master switch
  * ``duck_fade_ms``                         fade-out duration (default 50 ms)
  * ``duck_fade_in_ms``                      fade-in duration (default 200 ms)
  * ``duck_suspend_volume``                  target during fade-out (0.0)
  * ``duck_buffer_max_sec``                  max buffer duration (2.0 s)
  * ``duck_suspend_timeout_sec``             default unduck after (0.8 s)
  * ``duck_early_cancel_score_threshold``    real-interrupt score floor (0.7)
  * ``duck_early_resume_score_threshold``    false-interrupt score ceiling (0.2)
  * ``duck_cooldown_sec``                    min interval between unduck→duck
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from livekit import rtc
from livekit.agents.voice import io as lk_io

if TYPE_CHECKING:
    pass

logger = logging.getLogger("agent.ducking")


_State = Literal["NORMAL", "SUSPENDED", "CANCELLED"]


class DuckingMixer(lk_io.AudioOutput):
    """Wraps an :class:`AudioOutput` with soft-cancel / soft-resume gain
    and frame buffering for zero-loss false-interrupt recovery.

    Args:
        inner: The audio sink to forward frames to.
        fade_ms: Linear ramp duration for fade-out (duck), in ms.
        fade_in_ms: Linear ramp duration for fade-in (unduck), in ms.
        suspend_volume: Target gain at end of fade-out. 0.0 = full silence.
        buffer_max_sec: Safety cap on buffer duration to bound memory.
    """

    def __init__(
        self,
        inner: lk_io.AudioOutput,
        *,
        fade_ms: int = 50,
        fade_in_ms: int = 200,
        suspend_volume: float = 0.0,
        buffer_max_sec: float = 2.0,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            label=f"DuckingMixer→{inner.label}",
            capabilities=lk_io.AudioOutputCapabilities(pause=False),
            next_in_chain=inner,
            sample_rate=sample_rate or inner.sample_rate,
        )
        self._inner = inner
        self._fade_ms = max(1, int(fade_ms))
        self._fade_in_ms = max(1, int(fade_in_ms))
        self._suspend_volume = max(0.0, min(1.0, float(suspend_volume)))

        self._buffer_max_sec = buffer_max_sec

        self._state: _State = "NORMAL"
        self._volume_current: float = 1.0
        self._ramp_target: float = 1.0
        self._ramp_samples_remaining: int = 0
        self._ramp_step_per_sample: float = 0.0

        # Frame buffer for SUSPENDED state (Phase 2).
        self._buffer: list[rtc.AudioFrame] = []
        self._buffer_duration_sec: float = 0.0

        # Metrics (monotonic, never reset during session).
        self._total_ducks: int = 0
        self._total_unducks: int = 0
        self._total_cancels: int = 0
        self._total_buffer_drains: int = 0
        self._total_buffer_frames_drained: int = 0
        self._total_buffer_frames_dropped: int = 0
        self._duck_start_time: float = 0.0
        self._total_suspend_ms: float = 0.0

        logger.info(
            "[DuckingMixer] init  fade_out=%dms  fade_in=%dms  "
            "suspend_vol=%.2f  buffer_max=%.1fs  sr=%s",
            self._fade_ms, self._fade_in_ms, self._suspend_volume,
            buffer_max_sec, self._sample_rate,
        )

    # ------------------------------------------------------------------
    # Public state-transition API (called from StreamingPipeline)
    # ------------------------------------------------------------------

    @property
    def state(self) -> _State:
        return self._state

    @property
    def current_volume(self) -> float:
        return self._volume_current

    @property
    def buffered_sec(self) -> float:
        """Duration of audio currently held in the buffer."""
        return self._buffer_duration_sec

    @property
    def buffered_frames(self) -> int:
        return len(self._buffer)

    def get_metrics(self) -> dict:
        """Session-lifetime ducking metrics for telemetry."""
        total = self._total_ducks or 1
        return {
            "total_ducks": self._total_ducks,
            "total_unducks": self._total_unducks,
            "total_cancels": self._total_cancels,
            "false_positive_rate": (
                self._total_unducks / total if self._total_ducks else 0.0
            ),
            "avg_suspend_ms": (
                self._total_suspend_ms / total if self._total_ducks else 0.0
            ),
            "total_buffer_drains": self._total_buffer_drains,
            "total_buffer_frames_drained": self._total_buffer_frames_drained,
            "total_buffer_frames_dropped": self._total_buffer_frames_dropped,
        }

    def duck(self) -> None:
        """Begin fade-out and arm frame buffering.

        Idempotent — calling while already SUSPENDED restarts the ramp
        from current volume (so duck-during-unduck reverses cleanly).
        """
        if self._state == "CANCELLED":
            return
        prev = self._state
        self._state = "SUSPENDED"
        self._total_ducks += 1
        self._duck_start_time = time.monotonic()
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        self._begin_ramp(target=self._suspend_volume, duration_ms=self._fade_ms)
        logger.info(
            "[DuckingMixer] duck  %s→SUSPENDED  target_vol=%.2f  "
            "fade=%dms  t=%.3f",
            prev, self._suspend_volume, self._fade_ms,
            self._duck_start_time,
        )

    def unduck(self) -> None:
        """Drain buffered frames with fade-in, then resume normal flow.

        No-op if CANCELLED. The actual drain happens in the next
        ``capture_frame`` call(s) — this method only transitions state
        and sets up the fade-in ramp.
        """
        if self._state == "CANCELLED":
            return
        suspend_ms = 0.0
        if self._duck_start_time > 0:
            suspend_ms = (time.monotonic() - self._duck_start_time) * 1000
            self._total_suspend_ms += suspend_ms
        self._state = "NORMAL"
        self._total_unducks += 1
        # Ramp from 0 → 1.0 will be applied to drained buffer frames.
        self._volume_current = 0.0
        self._begin_ramp(target=1.0, duration_ms=self._fade_in_ms)
        logger.info(
            "[DuckingMixer] unduck  SUSPENDED→NORMAL  buffered=%d frames "
            "(%.3fs)  suspend_ms=%.0f  fade_in=%dms",
            len(self._buffer), self._buffer_duration_sec,
            suspend_ms, self._fade_in_ms,
        )

    def cancel(self) -> None:
        """Mark CANCELLED — discard buffer and drop all subsequent frames.

        Caller must also invoke ``session.interrupt()`` to stop TTS.
        """
        previous = self._state
        dropped = len(self._buffer)
        dropped_sec = self._buffer_duration_sec
        self._state = "CANCELLED"
        self._total_cancels += 1
        self._total_buffer_frames_dropped += dropped
        if self._duck_start_time > 0:
            self._total_suspend_ms += (
                (time.monotonic() - self._duck_start_time) * 1000
            )
        self._volume_current = 0.0
        self._ramp_target = 0.0
        self._ramp_samples_remaining = 0
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        try:
            self._inner.clear_buffer()
        except Exception:
            logger.warning("[DuckingMixer] inner.clear_buffer() failed")
        logger.info(
            "[DuckingMixer] cancel  %s→CANCELLED  discarded=%d frames (%.3fs)",
            previous, dropped, dropped_sec,
        )

    def reset(self) -> None:
        """Reset to NORMAL/v=1.0 without ramp. For session boundaries."""
        self._state = "NORMAL"
        self._volume_current = 1.0
        self._ramp_target = 1.0
        self._ramp_samples_remaining = 0
        self._ramp_step_per_sample = 0.0
        self._buffer.clear()
        self._buffer_duration_sec = 0.0

    # ------------------------------------------------------------------
    # AudioOutput abstract methods
    # ------------------------------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Route frame based on state: forward, buffer, or drop."""
        await super().capture_frame(frame)

        if self._state == "CANCELLED":
            return

        if self._state == "SUSPENDED":
            await self._handle_suspended(frame)
            return

        # NORMAL state — drain any buffered frames first, then forward.
        if self._buffer:
            await self._drain_buffer()

        await self._forward_with_gain(frame)

    def flush(self) -> None:
        super().flush()
        self._inner.flush()

    def clear_buffer(self) -> None:
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        self._inner.clear_buffer()

    def on_attached(self) -> None:
        super().on_attached()

    def on_detached(self) -> None:
        super().on_detached()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _handle_suspended(self, frame: rtc.AudioFrame) -> None:
        """SUSPENDED: fade-out phase forwards attenuated frames;
        after ramp completes, buffer frames silently."""
        if self._ramp_samples_remaining > 0:
            scaled = self._scale_frame(frame)
            await self._inner.capture_frame(scaled)
        else:
            sr = frame.sample_rate or self._sample_rate or 16000
            frame_sec = frame.samples_per_channel / sr
            if self._buffer_duration_sec + frame_sec <= self._buffer_max_sec:
                self._buffer.append(frame)
                self._buffer_duration_sec += frame_sec

    async def _drain_buffer(self) -> None:
        """Drain all buffered frames through the inner sink with gain ramp.

        Called from ``capture_frame`` when state transitions back to NORMAL.
        LiveKit's AudioSource applies backpressure via queue_size_ms, so
        burst-pushing frames here is safe — they play at normal speed.
        """
        count = len(self._buffer)
        dur = self._buffer_duration_sec
        self._total_buffer_drains += 1
        self._total_buffer_frames_drained += count

        t0 = time.monotonic()
        for frame in self._buffer:
            await self._forward_with_gain(frame)
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        drain_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "[DuckingMixer] drain  frames=%d  buffered=%.3fs  "
            "drain_ms=%.1f  vol_after=%.2f",
            count, dur, drain_ms, self._volume_current,
        )

    async def _forward_with_gain(self, frame: rtc.AudioFrame) -> None:
        """Forward a frame with current gain ramp applied. Fast-path
        when volume is 1.0 and no ramp is active."""
        if (
            self._ramp_samples_remaining == 0
            and abs(self._volume_current - 1.0) < 1e-6
        ):
            await self._inner.capture_frame(frame)
            return
        scaled = self._scale_frame(frame)
        await self._inner.capture_frame(scaled)

    def _begin_ramp(self, *, target: float, duration_ms: int | None = None) -> None:
        """Set up a linear ramp from ``_volume_current`` to ``target``."""
        target = max(0.0, min(1.0, float(target)))
        self._ramp_target = target
        sr = self._sample_rate or 16000
        ms = duration_ms if duration_ms is not None else self._fade_ms
        total_samples = max(1, int(sr * ms / 1000))
        self._ramp_samples_remaining = total_samples
        self._ramp_step_per_sample = (target - self._volume_current) / total_samples

    def _scale_frame(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Return a new AudioFrame with per-sample gain ramp applied."""
        samples = np.frombuffer(frame.data, dtype=np.int16)
        n = samples.size

        if self._ramp_samples_remaining == 0:
            scaled = self._apply_static_gain(samples, self._volume_current)
        else:
            ramp_now = self._ramp_samples_remaining
            ramp_in_frame = min(ramp_now, n)
            gains = np.empty(n, dtype=np.float32)
            start = self._volume_current
            step = self._ramp_step_per_sample
            gains[:ramp_in_frame] = start + step * np.arange(
                1, ramp_in_frame + 1, dtype=np.float32
            )
            if ramp_in_frame < n:
                gains[ramp_in_frame:] = self._ramp_target
            if ramp_in_frame >= ramp_now:
                self._volume_current = self._ramp_target
                self._ramp_samples_remaining = 0
                self._ramp_step_per_sample = 0.0
            else:
                self._volume_current = gains[ramp_in_frame - 1]
                self._ramp_samples_remaining -= ramp_in_frame

            scaled = np.clip(
                samples.astype(np.float32) * gains, -32768, 32767
            ).astype(np.int16)

        return rtc.AudioFrame(
            data=scaled.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    @staticmethod
    def _apply_static_gain(samples: np.ndarray, gain: float) -> np.ndarray:
        if abs(gain) < 1e-6:
            return np.zeros_like(samples)
        return np.clip(
            samples.astype(np.float32) * gain, -32768, 32767
        ).astype(np.int16)
