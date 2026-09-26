"""Half-duplex PTT turn controller.

This controller models the product contract for PTT:

* press opens a complete audio segment;
* press during agent output also preempts that output;
* release closes the segment and transcribes it as a whole;
* the terminal outcome is either commit(transcript) or reject(reason).

It does not consume streaming STT interim/final events and it does not use EOT.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from .ptt_segment import PttAudioSegment, PttAudioSegmentRecorder
from .ptt_transcriber import PttSegmentTranscriber

PttSegmentTurnAction = Literal["none", "commit", "reject"]
PttSegmentTurnState = Literal["idle", "recording", "transcribing"]


@dataclass(frozen=True)
class PttSegmentTurnResult:
    action: PttSegmentTurnAction
    reason: str = ""
    transcript: str = ""
    state: PttSegmentTurnState = "idle"
    preempted_agent_output: bool = False
    stt_mode: str = "none"
    stt_latency_ms: float = 0.0
    audio_duration_sec: float = 0.0
    audio_effective_duration_sec: float = 0.0
    audio_leading_silence_sec: float = 0.0
    audio_rms_ppm: int = 0


class HalfDuplexPttTurnController:
    """Own one half-duplex PTT turn at a time."""

    def __init__(
        self,
        *,
        recorder: PttAudioSegmentRecorder,
        transcriber: PttSegmentTranscriber,
        agent_output_active: Callable[[], bool] | None = None,
        preempt_agent_output: Callable[[], None] | None = None,
        tap_to_stop_max_audio_sec: float = 0.9,
    ) -> None:
        self._recorder = recorder
        self._transcriber = transcriber
        self._agent_output_active = agent_output_active or (lambda: False)
        self._preempt_agent_output = preempt_agent_output or (lambda: None)
        self._tap_to_stop_max_audio_sec = max(0.0, tap_to_stop_max_audio_sec)
        self._state: PttSegmentTurnState = "idle"
        self._preempted_agent_output = False
        self._generation = 0
        self._transcription_task: asyncio.Task | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def state(self) -> PttSegmentTurnState:
        return self._state

    def press(self) -> PttSegmentTurnResult:
        if self._state == "recording":
            return PttSegmentTurnResult(
                action="none",
                reason="already_recording",
                state=self._state,
                preempted_agent_output=self._preempted_agent_output,
            )
        self._generation += 1
        if self._transcription_task is not None:
            self._transcription_task.cancel()
            self._transcription_task = None
        preempted = self._agent_output_active()
        self._recorder.start()
        self._state = "recording"
        self._preempted_agent_output = preempted
        if preempted:
            self._preempt_agent_output()
        return PttSegmentTurnResult(
            action="none",
            reason="pressed",
            state=self._state,
            preempted_agent_output=preempted,
        )

    def abort(self) -> None:
        """Close capture and invalidate pending ASR when its serving scope ends."""
        self._generation += 1
        if self._transcription_task is not None:
            self._transcription_task.cancel()
            self._transcription_task = None
        self._recorder.reset()
        self._state = "idle"

    def push_frame(self, frame: Any) -> bool:
        if self._state != "recording":
            return False
        return self._recorder.push_frame(frame)

    def release(self) -> Coroutine[Any, Any, PttSegmentTurnResult]:
        """Close capture synchronously; only transcription runs asynchronously."""
        if self._state != "recording":
            return self._release_without_hold()
        self._state = "transcribing"
        segment = self._recorder.stop()
        return self._transcribe(segment, self._generation, self._preempted_agent_output)

    async def _release_without_hold(self) -> PttSegmentTurnResult:
        return PttSegmentTurnResult(
            action="reject", reason="release_without_active_hold", state=self._state,
        )

    async def _transcribe(
        self, segment: PttAudioSegment, generation: int, preempted: bool,
    ) -> PttSegmentTurnResult:
        if generation != self._generation:
            return PttSegmentTurnResult(action="none", reason="superseded", state=self._state)
        leading_silence_sec = segment.leading_silence_sec()
        effective_duration_sec = max(0.0, segment.duration_sec - leading_silence_sec)
        if (
            preempted
            and self._tap_to_stop_max_audio_sec > 0
            and effective_duration_sec <= self._tap_to_stop_max_audio_sec
        ):
            self._state = "idle"
            return PttSegmentTurnResult(
                action="reject",
                reason="tap_to_stop",
                state=self._state,
                preempted_agent_output=preempted,
                audio_duration_sec=segment.duration_sec,
                audio_effective_duration_sec=effective_duration_sec,
                audio_leading_silence_sec=leading_silence_sec,
                audio_rms_ppm=segment.rms_ppm,
            )
        task = asyncio.create_task(self._transcriber.transcribe(segment))
        self._transcription_task = task
        try:
            transcription = await task
        except (asyncio.CancelledError, Exception):
            if generation != self._generation:
                return PttSegmentTurnResult(action="none", reason="superseded", state=self._state)
            raise
        finally:
            if generation == self._generation:
                self._state = "idle"
                self._transcription_task = None
        if generation != self._generation:
            return PttSegmentTurnResult(action="none", reason="superseded", state=self._state)
        if not transcription.accepted:
            return PttSegmentTurnResult(
                action="reject",
                reason=transcription.rejected_reason or "transcription_rejected",
                state=self._state,
                preempted_agent_output=preempted,
                stt_mode=transcription.mode,
                stt_latency_ms=transcription.latency_ms,
                audio_duration_sec=transcription.audio_duration_sec,
                audio_effective_duration_sec=effective_duration_sec,
                audio_leading_silence_sec=leading_silence_sec,
                audio_rms_ppm=transcription.audio_rms_ppm,
            )

        return PttSegmentTurnResult(
            action="commit",
            reason="segment_transcribed",
            transcript=transcription.text,
            state=self._state,
            preempted_agent_output=preempted,
            stt_mode=transcription.mode,
            stt_latency_ms=transcription.latency_ms,
            audio_duration_sec=transcription.audio_duration_sec,
            audio_effective_duration_sec=effective_duration_sec,
            audio_leading_silence_sec=leading_silence_sec,
            audio_rms_ppm=transcription.audio_rms_ppm,
        )
