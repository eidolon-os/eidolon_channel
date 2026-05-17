# Copyright 2026 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""STT VAD-gated audio forwarder for cost reduction (G16, 2026-05-17).

============================================================================
PURPOSE
============================================================================

Bailian FunASR bills per second of audio processed (0.00033 RMB/s ~= 1.98 RMB/hour
when continuously streaming). Currently we send 100% of wall-clock time —
including user-silence windows, agent-speech windows, and AEC warmup periods —
which means ~55-65% of billing is wasted on silence-only audio.

This module implements a state machine that:
  * GATED: only sends 1Hz synthetic silence to keep the FunASR task alive
  * FORWARDING: passes real audio through to FunASR

Transitions are driven by VAD signal + RMS energy (fallback).

============================================================================
ZERO LATENCY GUARANTEE
============================================================================

Critical design: the FIRST-WORD must reach FunASR in full. VAD detection
inherently lags by 100-200ms behind the actual onset of speech. To avoid
losing the leading edge:

  1. While GATED, every incoming audio chunk is appended to ``_preroll_ring``
     (a deque capped at preroll_ms / chunk_ms entries — FIFO drop).
  2. On the FORWARDING transition, the *entire* preroll ring is flushed
     to FunASR before switching to passthrough.
  3. preroll_ms (default 500) is large enough to cover even pathological
     VAD lag plus a safety margin.

============================================================================
RELIABILITY VS COST TRADE-OFF
============================================================================

The big risk: VAD FALSE NEGATIVE → entire utterance dropped.

Defense:
  * Hysteresis: enter FORWARDING when probability ≥ ``vad_high_threshold``
    (default 0.6); exit only after probability < ``vad_low_threshold``
    (default 0.3) sustained for ``tail_window_ms`` (default 1500ms).
  * Energy fallback: if frame RMS exceeds ``rms_threshold`` (default 500)
    AND VAD is uncertain, force FORWARDING anyway. This catches plosives
    and short voiced bursts that don't trigger the neural VAD reliably.
  * Tail window: keeps FORWARDING for tail_window_ms after VAD goes silent,
    so trailing words / breath / glottal noises aren't cut off.

If both VAD and RMS miss → the utterance is lost. Operator should treat
this gate as opt-in via ``BAILIAN_STT_GATE_ENABLED`` and monitor
transcription quality before enabling broadly.

============================================================================
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np

logger = logging.getLogger("bailian.stt.gate")


def _silence_chunk(sample_rate: int, ms: int) -> bytes:
    """Generate a PCM-16 mono silence chunk."""
    num_samples = int(sample_rate * ms / 1000)
    return b"\x00\x00" * num_samples


def _rms_int16(audio_bytes: bytes) -> float:
    """RMS amplitude of int16 PCM bytes. Returns 0.0 on empty input."""
    if not audio_bytes:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    # Use float32 to avoid int overflow on large samples.
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


class SttGate:
    """VAD-gated audio forwarder.

    Lifecycle::

        gate = SttGate(sample_rate=16000, sender=conn.send_audio, ...)
        await gate.start()                       # spawns keepalive task

        # In send_loop:
        await gate.feed(audio_chunk_bytes)       # buffers or forwards

        # From VAD inference callback (per 50ms frame):
        gate.notify_vad_state(probability, rms)  # updates gate signal

        await gate.stop()                        # clean up keepalive task

    Thread / async safety: all public methods must be called from the same
    asyncio event loop. The state machine has a single writer (``feed`` /
    ``notify_vad_state`` / keepalive task — all on the same loop).
    """

    # Internal state
    _STATE_GATED = "GATED"
    _STATE_FORWARDING = "FORWARDING"

    def __init__(
        self,
        *,
        sample_rate: int,
        sender: Callable[[bytes], Awaitable[None]],
        preroll_ms: int = 500,
        tail_window_ms: int = 1500,
        keepalive_interval_sec: float = 1.0,
        keepalive_frame_ms: int = 100,
        chunk_ms: int = 100,
        vad_high_threshold: float = 0.6,
        vad_low_threshold: float = 0.3,
        rms_threshold: float = 500.0,
    ) -> None:
        if preroll_ms <= 0 or tail_window_ms < 0:
            raise ValueError("preroll_ms must be > 0, tail_window_ms >= 0")
        if vad_low_threshold >= vad_high_threshold:
            raise ValueError("vad_low_threshold must be < vad_high_threshold")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be > 0")
        if keepalive_interval_sec <= 0:
            raise ValueError("keepalive_interval_sec must be > 0")

        self._sample_rate = sample_rate
        self._sender = sender
        self._preroll_ms = preroll_ms
        self._tail_window_ms = tail_window_ms
        self._keepalive_interval = keepalive_interval_sec
        self._keepalive_frame_ms = keepalive_frame_ms
        self._chunk_ms = chunk_ms
        self._vad_high = vad_high_threshold
        self._vad_low = vad_low_threshold
        self._rms_threshold = rms_threshold

        # Preroll ring: capped deque of recent chunks (bytes).
        # Capacity = ceil(preroll_ms / chunk_ms).
        ring_capacity = max(1, (preroll_ms + chunk_ms - 1) // chunk_ms)
        self._preroll_ring: collections.deque[bytes] = collections.deque(
            maxlen=ring_capacity
        )

        self._state: str = self._STATE_GATED
        self._latest_vad_probability: float = 0.0
        self._latest_rms: float = 0.0
        self._last_above_high_time: float = 0.0  # monotonic
        self._last_above_low_time: float = 0.0  # last time prob >= vad_low

        self._silence_payload = _silence_chunk(sample_rate, keepalive_frame_ms)
        self._keepalive_task: Optional[asyncio.Task[None]] = None
        self._closed = False

        # Metrics
        self._total_forwarded_bytes: int = 0
        self._total_buffered_dropped_bytes: int = 0
        self._total_keepalive_sent: int = 0
        self._total_preroll_flushed_bytes: int = 0
        self._state_transitions: int = 0

    # ── Public API ─────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    async def start(self) -> None:
        """Spawn the keepalive task."""
        if self._keepalive_task is None:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def stop(self) -> None:
        """Cancel keepalive task and disable further sends."""
        self._closed = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

    async def feed(self, audio_chunk: bytes) -> None:
        """Process one audio chunk. Either buffers it (GATED) or forwards (FORWARDING).

        Call from the STT send_loop. The chunk's own RMS energy is computed
        here for the energy-fallback gate path (covers VAD false negatives).
        """
        if self._closed:
            return

        # G16: compute this chunk's RMS for the energy-fallback gate. The
        # VAD callback updates probability separately (per-frame, slightly
        # ahead of the chunk arriving here due to async paths). Use the
        # higher-recent RMS so a quiet last-frame after a loud burst still
        # keeps the gate open.
        self._latest_rms = max(_rms_int16(audio_chunk), self._latest_rms * 0.7)
        now = time.monotonic()
        if self._latest_rms >= self._rms_threshold:
            self._last_above_low_time = now

        prev_state = self._state
        self._maybe_transition()
        just_entered_forwarding = (
            prev_state == self._STATE_GATED
            and self._state == self._STATE_FORWARDING
        )

        if just_entered_forwarding:
            # Flush the preroll ring (recent chunks from before VAD trigger)
            # BEFORE the current chunk, so the first-word leading edge
            # reaches FunASR in temporal order.
            await self._flush_preroll()

        if self._state == self._STATE_FORWARDING:
            await self._sender(audio_chunk)
            self._total_forwarded_bytes += len(audio_chunk)
            # Maintain preroll ring even while forwarding — preserves the
            # most recent samples for a potential FORWARDING→GATED→FORWARDING
            # bounce.
            self._preroll_ring.append(audio_chunk)
        else:
            # GATED: ring buffer, count "would-have-been-sent" bytes
            # (savings metric).
            self._total_buffered_dropped_bytes += len(audio_chunk)
            self._preroll_ring.append(audio_chunk)

    async def _flush_preroll(self) -> None:
        """Drain preroll ring to sender (called once on GATED→FORWARDING)."""
        if not self._preroll_ring:
            return
        chunks = list(self._preroll_ring)
        self._preroll_ring.clear()
        flushed_bytes = 0
        for chunk in chunks:
            try:
                await self._sender(chunk)
                flushed_bytes += len(chunk)
            except Exception as e:
                logger.warning(
                    "[SttGate] preroll flush send failed mid-stream: %s", e
                )
                break
        self._total_preroll_flushed_bytes += flushed_bytes
        self._total_forwarded_bytes += flushed_bytes
        logger.info(
            "[SttGate] preroll flushed: %d chunks, %d bytes (%.1fs of audio)",
            len(chunks),
            flushed_bytes,
            flushed_bytes / (self._sample_rate * 2),  # 2 bytes per sample (PCM16)
        )

    def notify_vad_state(self, probability: float, rms: float) -> None:
        """Update VAD signal from upstream VAD inference. Per-frame (50ms).

        Cheap — only stores values; the actual decision happens in feed().
        """
        if self._closed:
            return
        now = time.monotonic()
        self._latest_vad_probability = probability
        self._latest_rms = rms

        # Track when probability last crossed each threshold so the tail
        # window can be computed precisely without per-chunk arithmetic.
        if probability >= self._vad_high:
            self._last_above_high_time = now
        if probability >= self._vad_low or rms >= self._rms_threshold:
            self._last_above_low_time = now

    def get_metrics(self) -> dict:
        """Snapshot of cost / state metrics."""
        return {
            "state": self._state,
            "state_transitions": self._state_transitions,
            "forwarded_bytes": self._total_forwarded_bytes,
            "buffered_dropped_bytes": self._total_buffered_dropped_bytes,
            "keepalive_sent": self._total_keepalive_sent,
            "preroll_flushed_bytes": self._total_preroll_flushed_bytes,
            "preroll_ring_depth": len(self._preroll_ring),
            "latest_vad_probability": self._latest_vad_probability,
            "latest_rms": self._latest_rms,
        }

    # ── Internals ──────────────────────────────────────────────────

    def _maybe_transition(self) -> None:
        """Inspect current VAD signal vs current state. May flip state."""
        now = time.monotonic()
        prob = self._latest_vad_probability
        rms = self._latest_rms

        if self._state == self._STATE_GATED:
            # Enter FORWARDING when:
            #   (a) VAD high — neural net is confident there's speech
            #   (b) RMS energy fallback — covers VAD false negatives
            if prob >= self._vad_high or rms >= self._rms_threshold:
                self._enter_forwarding(reason="vad_high" if prob >= self._vad_high else "rms_fallback")
            return

        # FORWARDING: exit only after tail window of sustained low signal.
        # "Sustained" = no probability >= vad_low AND no rms >= rms_threshold
        # during the tail window.
        sustained_low_duration = (now - self._last_above_low_time) * 1000.0
        if sustained_low_duration >= self._tail_window_ms:
            self._enter_gated(reason="tail_expired")

    def _enter_forwarding(self, *, reason: str) -> None:
        """Switch state to FORWARDING. Preroll flush happens in ``feed`` on
        the same tick (transition + immediate flush in a single async path)."""
        if self._state == self._STATE_FORWARDING:
            return
        self._state = self._STATE_FORWARDING
        self._state_transitions += 1
        ring_bytes = sum(len(c) for c in self._preroll_ring)
        logger.info(
            "[SttGate] GATED→FORWARDING (reason=%s, will flush preroll: %d chunks / %d bytes)",
            reason,
            len(self._preroll_ring),
            ring_bytes,
        )

    def _enter_gated(self, *, reason: str) -> None:
        if self._state == self._STATE_GATED:
            return
        self._state = self._STATE_GATED
        self._state_transitions += 1
        logger.info(
            "[SttGate] FORWARDING→GATED (reason=%s, tail elapsed)",
            reason,
        )

    async def _keepalive_loop(self) -> None:
        """Send a 100ms silence chunk every keepalive_interval_sec while
        the gate is GATED. Maintains the FunASR task without billing for
        a full audio stream.
        """
        try:
            while not self._closed:
                try:
                    await asyncio.sleep(self._keepalive_interval)
                except asyncio.CancelledError:
                    return
                if self._closed:
                    return
                if self._state != self._STATE_GATED:
                    continue
                try:
                    await self._sender(self._silence_payload)
                    self._total_keepalive_sent += 1
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    # Don't crash the keepalive task on a single send fail —
                    # the WS may be transiently bad, next loop will retry.
                    logger.debug("[SttGate] keepalive send failed: %s", e)
        except asyncio.CancelledError:
            return
