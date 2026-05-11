# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MockVAD — scripted in-process VAD stand-in.

Conforms to ``livekit.agents.vad.VAD``. Emits scripted START/END
speech events plus per-frame INFERENCE_DONE events with controlled
probability values. This is critical for testing G6 (VAD probability
into EOT state) and G2b (VAD-confidence gating) without needing the
real ONNX runtime.

Drive modes:
  * **Time-scripted**: ``MockVADEvent`` entries with absolute or
    relative timing — START at t=200ms, END at t=1500ms, etc.
  * **Auto-from-frames**: each pushed AudioFrame computes a
    probability from RMS (no neural net), emits INFERENCE_DONE per
    frame, and infers START/END at configurable threshold crossings.
    Useful when scenarios need probability to follow audio energy
    naturally.

The real FireRed VAD has ``update_interval`` ≈ 32 ms. We default to
the same so any test relying on inference rate matches production.
"""

from __future__ import annotations

import asyncio
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal, Optional

from livekit import rtc
from livekit.agents import vad as lk_vad


class _Phase(str, Enum):
    """Internal phase tracking."""

    SILENT = "silent"
    SPEAKING = "speaking"


@dataclass
class MockVADEvent:
    """A single scripted VAD transition."""

    type: Literal["start", "end"]
    """Whether this event marks the start or end of speech."""

    at_ms: int
    """Wall-clock milliseconds from stream start when this event fires."""

    probability: float = 0.95
    """Probability emitted in the corresponding INFERENCE_DONE just
    before the START/END is signalled. Must be in [0, 1]."""


class MockVAD(lk_vad.VAD):
    """Deterministic VAD mock.

    Examples:
        >>> # Scripted: speech 200ms→1500ms, then silence
        >>> vad = MockVAD.scripted([
        ...     MockVADEvent("start", at_ms=200, probability=0.9),
        ...     MockVADEvent("end", at_ms=1500, probability=0.1),
        ... ])

        >>> # Auto: probability follows audio RMS
        >>> vad = MockVAD.from_audio(speech_threshold=0.05)

        >>> # Always-quiet (no events; test silence handling)
        >>> vad = MockVAD.silent()
    """

    def __init__(
        self,
        *,
        scripts: Iterable[MockVADEvent] | None = None,
        mode: Literal["scripted", "from_audio", "silent"] = "scripted",
        speech_threshold: float = 0.05,
        inference_interval_sec: float = 0.032,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__(
            capabilities=lk_vad.VADCapabilities(
                update_interval=inference_interval_sec
            )
        )
        self._scripts = list(scripts or [])
        self._mode = mode
        self._speech_threshold = speech_threshold
        self._inference_interval_sec = inference_interval_sec
        self._sample_rate = sample_rate
        # Test counters / observability.
        self.streams_opened = 0
        self.frames_received = 0

    # ─────────────────────────────────────── factories

    @classmethod
    def scripted(
        cls, events: Iterable[MockVADEvent], **kwargs
    ) -> "MockVAD":
        return cls(scripts=events, mode="scripted", **kwargs)

    @classmethod
    def from_audio(
        cls, *, speech_threshold: float = 0.05, **kwargs
    ) -> "MockVAD":
        return cls(
            mode="from_audio", speech_threshold=speech_threshold, **kwargs
        )

    @classmethod
    def silent(cls, **kwargs) -> "MockVAD":
        return cls(mode="silent", **kwargs)

    # ─────────────────────────────────────── protocol

    @property
    def model(self) -> str:
        return "mock-vad"

    @property
    def provider(self) -> str:
        return "eidolon-test"

    def stream(self) -> lk_vad.VADStream:
        self.streams_opened += 1
        return _MockVADStream(vad=self)


def _frame_rms(frame: rtc.AudioFrame) -> float:
    """RMS energy ∈ [0, 1] of a 16-bit PCM AudioFrame."""
    raw = bytes(frame.data)
    n = len(raw) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", raw)
    s = 0
    for v in samples:
        s += v * v
    return math.sqrt(s / n) / 32_768.0


class _MockVADStream(lk_vad.VADStream):
    """Streaming impl. Two concurrent loops:
       * consume input frames, accumulate timing + (in from_audio mode)
         emit per-frame INFERENCE_DONE
       * (in scripted mode) fire scripted START/END at scheduled times
    """

    def __init__(self, *, vad: MockVAD) -> None:
        self._mock = vad
        self._stream_started = time.monotonic()
        self._phase = _Phase.SILENT
        self._segment_start_t: Optional[float] = None
        self._silence_start_t: Optional[float] = self._stream_started
        # Accumulated audio frames in the current speech segment (so
        # END_OF_SPEECH carries the full speech audio, mirroring real
        # plugins).
        self._segment_frames: list[rtc.AudioFrame] = []
        super().__init__(vad=vad)

    async def _main_task(self) -> None:
        consume_task = asyncio.create_task(
            self._consume_loop(), name="MockVAD._consume"
        )
        try:
            if self._mock._mode == "scripted":
                await self._scripted_emit_loop()
            # from_audio: events emitted inline by _consume_loop
            # silent: no events; just drain frames
            await consume_task
        finally:
            if not consume_task.done():
                consume_task.cancel()
                try:
                    await consume_task
                except (asyncio.CancelledError, BaseException):
                    pass

    async def _consume_loop(self) -> None:
        try:
            async for item in self._input_ch:
                if isinstance(item, lk_vad.VADStream._FlushSentinel):
                    # End-of-input flush — close out any open segment.
                    if self._phase == _Phase.SPEAKING:
                        self._emit_end(probability=0.0)
                    return
                frame: rtc.AudioFrame = item
                self._mock.frames_received += 1
                if self._mock._mode == "from_audio":
                    self._handle_frame_auto(frame)
                elif self._mock._mode == "silent":
                    # Still emit periodic INFERENCE_DONE with 0 prob so
                    # downstream avg-confidence calc has samples.
                    self._emit_inference(0.02, frame=frame)
                # scripted: no per-frame inference (would race the
                # scripted timing). Tests that need both time-driven
                # transitions AND per-frame probability should use
                # from_audio instead.
                if self._phase == _Phase.SPEAKING:
                    self._segment_frames.append(frame)
        except asyncio.CancelledError:
            return

    def _handle_frame_auto(self, frame: rtc.AudioFrame) -> None:
        rms = _frame_rms(frame)
        # Map RMS to a plausible-ish probability.
        # Below threshold: prob ∝ rms / threshold (scaled to <0.5).
        # Above threshold: prob = 0.5 + 0.5 * min(1, (rms - thr) / thr).
        thr = self._mock._speech_threshold
        if rms < thr:
            prob = max(0.0, min(0.49, rms / thr * 0.5))
        else:
            excess = (rms - thr) / max(thr, 1e-9)
            prob = min(1.0, 0.5 + 0.5 * min(1.0, excess))
        self._emit_inference(prob, frame=frame)

        if self._phase == _Phase.SILENT and prob >= 0.5:
            self._emit_start(probability=prob, frame=frame)
        elif self._phase == _Phase.SPEAKING and prob < 0.3:
            self._emit_end(probability=prob)

    async def _scripted_emit_loop(self) -> None:
        for ev in self._mock._scripts:
            target_t = self._stream_started + ev.at_ms / 1000.0
            now = time.monotonic()
            if target_t > now:
                await asyncio.sleep(target_t - now)
            self._emit_inference(ev.probability)
            if ev.type == "start":
                self._emit_start(probability=ev.probability)
            else:
                self._emit_end(probability=ev.probability)

    # ── emitters

    def _now(self) -> float:
        return time.monotonic()

    def _segment_duration(self) -> float:
        if self._segment_start_t is None:
            return 0.0
        return max(0.0, self._now() - self._segment_start_t)

    def _silence_duration(self) -> float:
        if self._silence_start_t is None:
            return 0.0
        return max(0.0, self._now() - self._silence_start_t)

    def _send(self, event: lk_vad.VADEvent) -> None:
        if not self._event_ch.closed:
            self._event_ch.send_nowait(event)

    def _emit_inference(
        self, probability: float, *, frame: Optional[rtc.AudioFrame] = None
    ) -> None:
        self._send(
            lk_vad.VADEvent(
                type=lk_vad.VADEventType.INFERENCE_DONE,
                samples_index=int(self._segment_duration() * self._mock._sample_rate),
                timestamp=time.time(),
                speech_duration=self._segment_duration(),
                silence_duration=self._silence_duration(),
                frames=[frame] if frame is not None else [],
                probability=probability,
                inference_duration=self._mock._inference_interval_sec,
                speaking=self._phase == _Phase.SPEAKING,
            )
        )

    def _emit_start(
        self,
        *,
        probability: float,
        frame: Optional[rtc.AudioFrame] = None,
    ) -> None:
        if self._phase == _Phase.SPEAKING:
            return
        self._phase = _Phase.SPEAKING
        self._segment_start_t = self._now()
        self._silence_start_t = None
        self._segment_frames = [frame] if frame is not None else []
        self._send(
            lk_vad.VADEvent(
                type=lk_vad.VADEventType.START_OF_SPEECH,
                samples_index=0,
                timestamp=time.time(),
                speech_duration=0.0,
                silence_duration=0.0,
                frames=self._segment_frames[:],
                probability=probability,
                speaking=True,
            )
        )

    def _emit_end(self, *, probability: float) -> None:
        if self._phase == _Phase.SILENT:
            return
        speech_dur = self._segment_duration()
        self._phase = _Phase.SILENT
        self._segment_start_t = None
        self._silence_start_t = self._now()
        self._send(
            lk_vad.VADEvent(
                type=lk_vad.VADEventType.END_OF_SPEECH,
                samples_index=int(speech_dur * self._mock._sample_rate),
                timestamp=time.time(),
                speech_duration=speech_dur,
                silence_duration=0.0,
                frames=self._segment_frames,
                probability=probability,
                speaking=False,
            )
        )
        self._segment_frames = []
