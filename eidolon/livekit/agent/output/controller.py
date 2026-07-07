# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Output controller — the audio-sink middleware between TTS and RoomIO.

============================================================================
G17b (2026-05-18) RENAME NOTE
============================================================================

This module was ``ducking.py`` / class ``DuckingMixer`` until Phase 2 of the
G17–G22 refactor. The new name reflects what the class actually does after
G18a + G17a removed the "wait for confidence" semantics: it's no longer
just a volume ducker, it's the **OutputController** that decides whether
TTS audio reaches the user (NORMAL), is held back during the interrupt-
decision window (SUSPENDED), or is dropped permanently after a confirmed
real interrupt (CANCELLED).

The class still implements three states for now — SUSPENDED retains the
fade-out + buffer machinery used by the fast-path soft-unduck. G17c
(Phase 2 tail) will collapse SUSPENDED → MUTED once we've verified
production runs no longer need the soft-buffer/drain path.

Import ``OutputController`` from ``eidolon.livekit.agent.output`` or
``eidolon.livekit.agent.output.controller``.

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

  Optional — SUSPENDED PASSTHROUGH:
    ``enable_suspended_passthrough(volume=...)`` is an explicit opt-in
    primitive for future reversible "audible hold" strategies. After the
    fade-out ramp completes, new live frames are forwarded at the capped
    volume instead of being buffered. Default production behaviour remains
    silent buffering until a caller deliberately enables passthrough.

On unduck (default ``drop_buffered=False``), the buffered frames are
drained through the inner sink with a fade-in ramp. LiveKit's
``AudioSource`` has internal queue pacing, so burst-pushing buffered
frames does NOT cause fast-forward — the downstream pipeline paces them
at wall-clock speed.

On unduck ``(drop_buffered=True)`` — used by the timeout fallback after
~500 ms — buffered frames are discarded (G17a: they are stale by then,
the user has been talking). Live TTS frames arriving after still go
through with the fade-in ramp.

On cancel, the buffer is discarded immediately.

============================================================================
ALL TUNABLES LIVE IN ``EidolonEOTConfig``
============================================================================

See ``plugins/eot/config.py`` for the full block. Quick reference:

  * ``duck_enabled``                         master switch
  * ``duck_fade_ms``                         fade-out duration (default 50 ms)
  * ``duck_fade_in_ms``                      fade-in duration (default 200 ms)
  * ``duck_suspend_volume``                  target during fade-out (0.0)
  * ``duck_buffer_max_sec``                  max buffer duration (2.0 s)
  * ``duck_suspend_timeout_sec``             decision-window deadline (now 0.5 s, G18a)
  * ``duck_early_cancel_score_threshold``    real-interrupt score floor (0.7)
  * ``duck_early_resume_score_threshold``    false-interrupt score ceiling (0.2)
  * ``duck_cooldown_sec``                    min interval between unduck→duck
  * ``interrupt_min_interim_chars``          first-signal threshold (G18a, default 2)
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

logger = logging.getLogger("agent.output.controller")


# G17c (2026-05-18) Note on state names:
# SUSPENDED is conceptually MUTED in the new flow. We keep the literal
# "SUSPENDED" string for API stability (StreamingPipeline + tests reference
# it by name in many places); a future cleanup may rename to MUTED. The
# buffer-and-drain machinery still exists for the rare fast-soft-unduck
# path (user_silent transition <300ms with score-based low EOT signal),
# but it is no longer the primary unduck behaviour — the timeout-deadline
# path (much more common) uses drop_buffered=True and short fades.
_State = Literal["NORMAL", "SUSPENDED", "CANCELLED"]


class OutputController(lk_io.AudioOutput):
    """Output middleware: mute / cancel / fade-in-on-resume around TTS.

    Three states:

      * NORMAL — frames pass through with volume 1.0.
      * SUSPENDED — interrupt decision window active. Default flow (G18a):
        first fade-out frame is sent (anti-click ramp), subsequent frames
        are HELD (buffered) until decision lands. On cancel they are
        discarded. On unduck(drop_buffered=True) — the timeout path —
        they are also discarded and live frames resume with a fade-in
        ramp. On unduck(drop_buffered=False) — the legacy soft-resume
        path — they are drained with fade-in (kept for now; G18a's
        first-signal path makes this rare).
      * CANCELLED — terminal until reset; all frames dropped.

    Args:
        inner: The audio sink to forward frames to.
        fade_ms: Linear ramp duration for fade-out (duck), in ms.
            G17c default reduced to 30ms (anti-click only).
        fade_in_ms: Linear ramp duration for fade-in (unduck), in ms.
            G17c default reduced to 30ms (anti-click only).
        suspend_volume: Target gain at end of fade-out. 0.0 = full silence.
        buffer_max_sec: Safety cap on buffer duration to bound memory.
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
        # G17c (2026-05-18): default fade durations reduced 50→30ms (out)
        # and 200→30ms (in). After G18a moved decisions onto a 500ms
        # budget, long fades have no design role — they were a holdover
        # from the original soft-duck model that assumed multi-second
        # suspend windows. 30ms is the minimum that avoids click/pop
        # artifacts on hard cut+resume (anti-click ramp only).
        # The constructor still accepts custom values; callers that
        # want a longer fade for special cases can opt in.
        # F4 (2026-05-16): advertise pause=True so framework's
        # ``resume_false_interruption`` mechanism actually runs. Previously
        # DuckingMixer declared pause=False, which caused framework to log
        # a warning and silently ignore the configured ``false_interruption_timeout``
        # (e.g. 6.0s). With pause=True the base-class pause/resume methods
        # (``AudioOutput.pause`` / ``resume``) cascade through to the inner
        # sink (TranscriptSynchronizer → RoomIO), which physically pauses
        # the rtc.AudioSource playback queue — frames already queued are
        # resumed-from-where-left-off when the framework calls resume().
        #
        # Coexistence with DuckingMixer's own duck/unduck state machine:
        # the framework calls pause() only when ``agent_state != "speaking"``
        # (agent_activity.py:1684), and (after F3) DuckingMixer.duck() only
        # runs when ``agent_state == speaking``. The two paths are mutually
        # exclusive, no double-pause race.
        super().__init__(
            label=f"OutputController→{inner.label}",
            capabilities=lk_io.AudioOutputCapabilities(pause=True),
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
            # G17a (2026-05-18): frames dropped because unduck(drop_buffered=True)
            # decided the buffer was stale (timeout fallback). Should be small
            # relative to total_buffer_frames_drained; high values indicate the
            # decision window is too long.
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

        Idempotent — calling while already SUSPENDED restarts the ramp
        from current volume (so duck-during-unduck reverses cleanly).

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
        if self._state == "CANCELLED":
            return
        prev = self._state
        self._state = "SUSPENDED"
        self._total_ducks += 1
        self._duck_start_time = time.monotonic()
        self._buffer.clear()
        self._buffer_duration_sec = 0.0
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

        No-op if CANCELLED. The actual drain happens in the next
        ``capture_frame`` call(s) — this method only transitions state
        and sets up the fade-in ramp.

        Args:
            drop_buffered: If True, discard the buffered frames instead of
                draining them. G17a (2026-05-18) — used by the timeout
                fallback in DuckSuspendTimeoutHandler:
                by the time the 0.8s timeout fires, any buffered TTS frames
                are stale (the user has been talking through the window),
                and replaying them after fade-in causes "agent talks over
                user" overlap. Fast-path soft-unduck (called on user-silent
                transition <300ms) keeps the default drain-with-fade-in
                because the buffer is genuinely fresh there.
        """
        if self._state == "CANCELLED":
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

    def cancel(self) -> None:
        """Mark CANCELLED — discard buffer and drop all subsequent frames.

        Caller must also invoke ``session.interrupt(force=True)`` to stop TTS.
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
        after ramp completes, buffer frames silently unless explicit
        suspended passthrough is enabled."""
        if self._ramp_samples_remaining > 0:
            scaled = self._scale_frame(frame)
            await self._inner.capture_frame(scaled)
            # G6 (2026-05-17): fade-out frames ARE played (just attenuated)
            # — the user heard them. Count toward played_seconds.
            self._played_samples_this_turn += frame.samples_per_channel
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
            "[OutputController] drain  frames=%d  buffered=%.3fs  "
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
            # G6 (2026-05-17): count frames actually played to user.
            self._played_samples_this_turn += frame.samples_per_channel
            return
        scaled = self._scale_frame(frame)
        await self._inner.capture_frame(scaled)
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
            await self._inner.capture_frame(frame)
        else:
            scaled = self._scale_frame_static(frame, gain)
            await self._inner.capture_frame(scaled)
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
