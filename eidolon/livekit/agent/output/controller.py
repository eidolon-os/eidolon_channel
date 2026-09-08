# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Audio-sink middleware with reversible suspension and generation-scoped writes.

The SDK owns synthesis and playback pacing. This interposer owns admission to
its sink: one serialized writer preserves frame order; bounded buffering applies
backpressure to synthesis; resume drains even after synthesis has finished.
Cancel/clear/reset invalidate pending writes before touching the sink queue.
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from typing import Literal

import numpy as np
from livekit import rtc
from livekit.agents.voice import io as lk_io


logger = logging.getLogger("agent.output.controller")


_State = Literal["NORMAL", "SUSPENDED", "CANCELLED"]


class OutputController(lk_io.AudioOutput):
    """NORMAL forwards, SUSPENDED buffers, CANCELLED rejects until reset.

    Fade durations are milliseconds. ``buffer_max_sec`` bounds queued audio;
    one oversized provider frame is accepted to avoid a capacity deadlock.
    Only an explicit ``drop_buffered`` decision discards suspended content.
    Metrics describe samples forwarded to the sink, not physical device ACKs.
    """

    def __init__(
        self,
        inner: lk_io.AudioOutput,
        *,
        fade_ms: int = 30,
        fade_in_ms: int = 30,
        suspend_volume: float = 0.0,
        buffer_max_sec: float = 2.0,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            label=f"OutputController→{inner.label}",
            capabilities=lk_io.AudioOutputCapabilities(pause=True),
            next_in_chain=inner,
            sample_rate=sample_rate or inner.sample_rate,
        )
        # LiveKit may insert a sink proxy; frames and playback events must
        # traverse the same chain or its segment accounting never completes.
        self._inner = self.next_in_chain
        self._fade_ms = max(1, int(fade_ms))
        self._fade_in_ms = max(1, int(fade_in_ms))
        self._suspend_volume = max(0.0, min(1.0, float(suspend_volume)))

        self._buffer_max_sec = max(0.0, buffer_max_sec)

        self._state: _State = "NORMAL"
        self._volume_current: float = 1.0
        self._ramp_target: float = 1.0
        self._ramp_samples_remaining: int = 0
        self._ramp_step_per_sample: float = 0.0

        # Frame buffer for SUSPENDED state (Phase 2).
        self._buffer: deque[rtc.AudioFrame] = deque()
        self._write_lock = asyncio.Lock()
        self._state_changed = asyncio.Event()
        self._generation = 0
        self._write_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        self._flush_pending = False
        self._buffer_duration_sec: float = 0.0
        self._suspended_passthrough_volume: float | None = None
        self._suspended_passthrough_enabled_at: float | None = None
        self._first_suspended_passthrough_frame_at: float | None = None
        self._last_suspended_passthrough_frame_at: float | None = None

        # Metrics (monotonic, never reset during session).
        self._total_ducks: int = 0
        self._total_unducks: int = 0
        self._total_cancels: int = 0
        self._total_buffer_drains: int = 0
        self._total_buffer_frames_drained: int = 0
        self._total_buffer_frames_dropped: int = 0
        self._total_buffer_frames_dropped_on_unduck: int = 0
        self._total_buffer_frames_dropped_on_passthrough: int = 0
        self._total_suspended_passthrough_frames: int = 0
        self._duck_start_time: float = 0.0
        self._total_suspend_ms: float = 0.0
        # G6 (2026-05-17): track audio actually forwarded to the inner sink
        # during the current agent turn. Full-duplex context ledger snapshots
        # use this to compute how much of the agent's reply the user actually
        # heard before a cancel. Reset on each ``duck()`` (which fires at the
        # start of each user-speech window, marking a potential turn boundary).
        self._played_samples_this_turn: int = 0

        logger.info(
            "[OutputController] init  fade_out=%dms  fade_in=%dms  "
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

    @property
    def suspended_passthrough_volume(self) -> float | None:
        """Current opt-in passthrough volume while SUSPENDED, if enabled."""
        return self._suspended_passthrough_volume

    @property
    def played_seconds(self) -> float:
        """G6 (2026-05-17): seconds of audio actually forwarded to the inner
        sink since the last ``duck()`` reset. Approximates "how much of the
        agent's reply the user heard before the interruption point".

        Sub-frame attribution: frames that went through during fade-out
        count fully (the user heard them, just faded); frames buffered in
        SUSPENDED do NOT count (we held them back, user heard silence).
        """
        sr = self._sample_rate or 16000
        return self._played_samples_this_turn / sr

    def get_metrics(self) -> dict:
        """Session-lifetime ducking metrics for telemetry."""
        total = self._total_ducks or 1
        metrics = {
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
            # Explicit discard requests only; ordinary timeout rollback keeps
            # unheard audio and must not increment this counter.
            "total_buffer_frames_dropped_on_unduck": (
                self._total_buffer_frames_dropped_on_unduck
            ),
            "total_buffer_frames_dropped_on_passthrough": (
                self._total_buffer_frames_dropped_on_passthrough
            ),
            "total_suspended_passthrough_frames": (
                self._total_suspended_passthrough_frames
            ),
        }
        if self._duck_start_time > 0:
            if self._suspended_passthrough_enabled_at is not None:
                metrics["suspended_passthrough_enabled_since_duck_ms"] = (
                    self._suspended_passthrough_enabled_at
                    - self._duck_start_time
                ) * 1000.0
            if self._first_suspended_passthrough_frame_at is not None:
                metrics["suspended_passthrough_first_frame_since_duck_ms"] = (
                    self._first_suspended_passthrough_frame_at
                    - self._duck_start_time
                ) * 1000.0
            if self._last_suspended_passthrough_frame_at is not None:
                metrics["suspended_passthrough_last_frame_since_duck_ms"] = (
                    self._last_suspended_passthrough_frame_at
                    - self._duck_start_time
                ) * 1000.0
        return metrics

    def duck(self) -> None:
        """Begin fade-out and arm frame buffering.

        Repeated suspension is a no-op. Suspending during resume reverses
        the ramp while preserving every buffered, unheard frame.

        G22 (2026-05-18): no longer resets ``_played_samples_this_turn``.
        The previous behaviour was buggy: within a single agent turn the
        user could trigger 2-3 duck cycles (backchannels), each resetting
        the counter, so by the time a real interrupt landed the counter
        only reflected the LAST duck cycle's frames — drastically
        under-counting how much the user had actually heard. The counter
        now resets only on transitions to SPEAKING (see
        ``on_agent_started_speaking()``), which is the genuine turn
        boundary.
        """
        if self._state in ("CANCELLED", "SUSPENDED"):
            return
        prev = self._state
        self._state = "SUSPENDED"
        self._total_ducks += 1
        self._duck_start_time = time.monotonic()
        # A new speech onset during resume must retain the unheard tail.
        self._suspended_passthrough_volume = None
        self._suspended_passthrough_enabled_at = None
        self._first_suspended_passthrough_frame_at = None
        self._last_suspended_passthrough_frame_at = None
        self._begin_ramp(target=self._suspend_volume, duration_ms=self._fade_ms)
        logger.info(
            "[OutputController] duck  %s→SUSPENDED  target_vol=%.2f  "
            "fade=%dms  t=%.3f",
            prev, self._suspend_volume, self._fade_ms,
            self._duck_start_time,
        )

    def on_agent_started_speaking(self) -> None:
        """G22 (2026-05-18): mark the start of a new agent turn.

        Called by StreamingPipeline when ``agent_state`` transitions to
        ``speaking``. Resets the per-turn played-sample counter so
        ``played_seconds`` accurately reflects how much of THIS turn's
        reply the user has heard. Independent of duck/unduck cycles,
        which may fire multiple times within one turn (backchannel,
        echo, etc.) and used to spuriously clear the counter."""
        self._played_samples_this_turn = 0
        logger.debug(
            "[OutputController] on_agent_started_speaking: "
            "played counter reset for new turn"
        )

    def enable_suspended_passthrough(self, *, volume: float) -> bool:
        """Forward live SUSPENDED frames at ``volume`` instead of buffering.

        This is intentionally opt-in and only valid while the output is already
        SUSPENDED. If frames were buffered before passthrough was enabled, they
        are discarded because live passthrough makes the buffered audio stale
        and out of order. Terminal cancel/unduck behaviour remains unchanged.
        """
        if self._state != "SUSPENDED":
            return False
        clamped = max(0.0, min(1.0, float(volume)))
        if clamped <= 0.0:
            self.disable_suspended_passthrough()
            return False
        if (
            self._suspended_passthrough_volume is not None
            and abs(self._suspended_passthrough_volume - clamped) <= 1e-9
        ):
            return False
        dropped = len(self._buffer)
        if dropped:
            self._total_buffer_frames_dropped_on_passthrough += dropped
            self._buffer.clear()
            self._buffer_duration_sec = 0.0
        self._suspended_passthrough_volume = clamped
        self._state_changed.set()
        self._suspended_passthrough_enabled_at = time.monotonic()
        logger.info(
            "[OutputController] suspended passthrough enabled  "
            "volume=%.2f  discarded_stale=%d",
            clamped,
            dropped,
        )
        return True

    def disable_suspended_passthrough(self) -> None:
        """Return SUSPENDED handling to silent buffering for future frames."""
        self._suspended_passthrough_volume = None

    def unduck(self, *, drop_buffered: bool = False) -> None:
        """Drain buffered frames with fade-in, then resume normal flow.

        No-op unless suspended. Resume schedules the same serialized writer
        used by capture_frame, including when no further TTS frames arrive.

        Args:
            drop_buffered: Explicitly discard held content instead of draining
                it. Ordinary rollback, including evidence timeout, retains the
                default: elapsed time does not make unheard content obsolete.
        """
        if self._state != "SUSPENDED":
            return
        suspend_ms = 0.0
        if self._duck_start_time > 0:
            suspend_ms = (time.monotonic() - self._duck_start_time) * 1000
            self._total_suspend_ms += suspend_ms
        self._state = "NORMAL"
        self._total_unducks += 1

        dropped = 0
        dropped_sec = 0.0
        if drop_buffered and self._buffer:
            dropped = len(self._buffer)
            dropped_sec = self._buffer_duration_sec
            self._buffer.clear()
            self._buffer_duration_sec = 0.0
            self._total_buffer_frames_dropped_on_unduck += dropped

        # Ramp back to 1.0. Silent-buffering windows start from 0; if an
        # explicit suspended passthrough was active, fade from that audible
        # hold volume instead of dipping to silence first.
        resume_start_volume = self._suspended_passthrough_volume or 0.0
        self._suspended_passthrough_volume = None
        self._volume_current = resume_start_volume
        self._begin_ramp(target=1.0, duration_ms=self._fade_in_ms)
        logger.info(
            "[OutputController] unduck  SUSPENDED→NORMAL  "
            "%s=%d frames (%.3fs)  suspend_ms=%.0f  fade_in=%dms",
            "dropped_stale" if drop_buffered else "buffered",
            dropped if drop_buffered else len(self._buffer),
            dropped_sec if drop_buffered else self._buffer_duration_sec,
            suspend_ms, self._fade_in_ms,
        )

        self._state_changed.set()
        if self._buffer or self._flush_pending:
            self._schedule_drain()

    def cancel(self) -> None:
        """Mark CANCELLED — discard buffer and drop all subsequent frames.

        Caller must also invoke ``session.interrupt(force=True)`` to stop TTS.
        """
        self._invalidate_writes()
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
        self._suspended_passthrough_volume = None
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        try:
            self._inner.clear_buffer()
        except Exception:
            logger.warning("[OutputController] inner.clear_buffer() failed")
        logger.info(
            "[OutputController] cancel  %s→CANCELLED  discarded=%d frames (%.3fs)",
            previous, dropped, dropped_sec,
        )

    def reset(self) -> None:
        """Reset to NORMAL/v=1.0 without ramp. For session boundaries."""
        self._invalidate_writes()
        self._state = "NORMAL"
        self._volume_current = 1.0
        self._ramp_target = 1.0
        self._ramp_samples_remaining = 0
        self._ramp_step_per_sample = 0.0
        self._suspended_passthrough_volume = None
        self._suspended_passthrough_enabled_at = None
        self._first_suspended_passthrough_frame_at = None
        self._last_suspended_passthrough_frame_at = None
        self._buffer.clear()
        self._buffer_duration_sec = 0.0

    # ------------------------------------------------------------------
    # AudioOutput abstract methods
    # ------------------------------------------------------------------

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Serialize admission, applying backpressure when suspended and full."""
        generation = self._generation
        async with self._write_lock:
            if generation != self._generation or self._state == "CANCELLED":
                return
            await super().capture_frame(frame)
            while generation == self._generation and self._state != "CANCELLED":
                if self._state == "NORMAL":
                    await self._drain_buffer()
                    if generation != self._generation:
                        return
                    if self._state != "NORMAL":
                        continue
                    await self._forward_with_gain(frame)
                    break
                if self._can_accept_suspended(frame):
                    remainder = await self._handle_suspended(frame)
                    if remainder is None:
                        break
                    # A fade can end inside a provider frame. Read the current
                    # state again before admitting its untouched remainder.
                    frame = remainder
                    continue
                # No data is discarded at capacity. A mode transition wakes the
                # producer; cancellation also invalidates this waiting frame.
                self._state_changed.clear()
                await self._state_changed.wait()
        self._flush_if_ready()

    def flush(self) -> None:
        self._flush_pending = True
        self._flush_if_ready()

    def _flush_if_ready(self) -> None:
        if self._flush_pending and not self._buffer and not self._write_lock.locked():
            self._flush_pending = False
            super().flush()
            self._inner.flush()

    def _invalidate_writes(self) -> None:
        self._generation += 1
        self._flush_pending = False
        self._state_changed.set()
        if self._write_task is not None:
            self._write_task.cancel()
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None

    def clear_buffer(self) -> None:
        self._invalidate_writes()
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        self._inner.clear_buffer()

    def _can_accept_suspended(self, frame: rtc.AudioFrame) -> bool:
        return (
            self._ramp_samples_remaining > 0
            or self._suspended_passthrough_volume is not None
            # One oversized provider frame is allowed; never deadlock at zero
            # capacity or if the provider's chunk is larger than the limit.
            or not self._buffer
            or self._buffer_duration_sec + frame.duration <= self._buffer_max_sec + 1e-9
        )

    def _schedule_drain(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._resume_output())
            self._drain_task.add_done_callback(self._drain_done)

    def _drain_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            # No capture_frame caller remains when a finished TTS tail drains.
            # Terminate the segment through the sink's normal clear contract
            # rather than leave playback waiters suspended indefinitely.
            self.cancel()
            logger.error("[OutputController] resume failed", exc_info=error)

    async def _resume_output(self) -> None:
        async with self._write_lock:
            await self._drain_buffer()
        self._flush_if_ready()

    async def _write(self, frame: rtc.AudioFrame) -> bool:
        """Only writer to the sink; invalidation cancels an in-flight await."""
        generation = self._generation
        task = asyncio.create_task(self._inner.capture_frame(frame))
        self._write_task = task
        try:
            await task
            return generation == self._generation
        except asyncio.CancelledError:
            if generation == self._generation or asyncio.current_task().cancelling():
                raise
            return False
        finally:
            if self._write_task is task:
                self._write_task = None

    def on_attached(self) -> None:
        super().on_attached()

    def on_detached(self) -> None:
        self._invalidate_writes()
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
        super().on_detached()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _handle_suspended(self, frame: rtc.AudioFrame) -> rtc.AudioFrame | None:
        """SUSPENDED: fade-out phase forwards attenuated frames;
        after ramp completes, buffer frames silently unless explicit
        suspended passthrough is enabled."""
        if self._ramp_samples_remaining > 0:
            remainder = None
            if self._ramp_samples_remaining < frame.samples_per_channel:
                ramp_samples = self._ramp_samples_remaining
                split = ramp_samples * frame.num_channels
                remainder = rtc.AudioFrame(
                    data=frame.data[split:].tobytes(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.samples_per_channel - ramp_samples,
                )
                frame = rtc.AudioFrame(
                    data=frame.data[:split].tobytes(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=ramp_samples,
                )
            scaled = self._scale_frame(frame)
            if not await self._write(scaled):
                return None
            # Only the forwarded fade prefix contributes to this estimate;
            # sink admission does not acknowledge physical playout.
            self._played_samples_this_turn += frame.samples_per_channel
            return remainder
        elif self._suspended_passthrough_volume is not None:
            await self._forward_with_static_gain(
                frame,
                self._suspended_passthrough_volume,
            )
            now = time.monotonic()
            if self._first_suspended_passthrough_frame_at is None:
                self._first_suspended_passthrough_frame_at = now
            self._last_suspended_passthrough_frame_at = now
            self._total_suspended_passthrough_frames += 1
        else:
            sr = frame.sample_rate or self._sample_rate or 16000
            frame_sec = frame.samples_per_channel / sr
            self._buffer.append(frame)
            self._buffer_duration_sec += frame_sec
        return None

    async def _drain_buffer(self) -> None:
        """Drain in order; recheck admission after every awaited write.

        A sink may accept faster than playout; its capture acknowledgement is
        not a playback acknowledgement.
        """
        generation = self._generation
        if not self._buffer or self._state != "NORMAL":
            return
        self._total_buffer_drains += 1
        while self._buffer and self._state == "NORMAL" and generation == self._generation:
            frame = self._buffer.popleft()
            self._buffer_duration_sec = max(0.0, self._buffer_duration_sec - frame.duration)
            await self._forward_with_gain(frame)
            if generation == self._generation:
                self._total_buffer_frames_drained += 1

    async def _forward_with_gain(self, frame: rtc.AudioFrame) -> None:
        """Forward a frame with current gain ramp applied. Fast-path
        when volume is 1.0 and no ramp is active."""
        if (
            self._ramp_samples_remaining == 0
            and abs(self._volume_current - 1.0) < 1e-6
        ):
            if not await self._write(frame):
                return
            # G6 (2026-05-17): count frames actually played to user.
            self._played_samples_this_turn += frame.samples_per_channel
            return
        scaled = self._scale_frame(frame)
        if not await self._write(scaled):
            return
        # G6 (2026-05-17): fade-in frames count too — user hears them at
        # ramp-up volume but they ARE played.
        self._played_samples_this_turn += frame.samples_per_channel

    async def _forward_with_static_gain(
        self,
        frame: rtc.AudioFrame,
        gain: float,
    ) -> None:
        """Forward a frame at a fixed gain without touching ramp state."""
        gain = max(0.0, min(1.0, float(gain)))
        if abs(gain - 1.0) < 1e-6:
            if not await self._write(frame):
                return
        else:
            scaled = self._scale_frame_static(frame, gain)
            if not await self._write(scaled):
                return
        self._played_samples_this_turn += frame.samples_per_channel

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
        n = frame.samples_per_channel

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
                samples.astype(np.float32) * np.repeat(gains, frame.num_channels),
                -32768, 32767,
            ).astype(np.int16)

        return rtc.AudioFrame(
            data=scaled.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )

    def _scale_frame_static(self, frame: rtc.AudioFrame, gain: float) -> rtc.AudioFrame:
        samples = np.frombuffer(frame.data, dtype=np.int16)
        scaled = self._apply_static_gain(samples, gain)
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
