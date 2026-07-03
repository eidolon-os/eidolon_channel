"""Half-duplex PTT turn controller.

This controller models the product contract for PTT:

* press opens a complete audio segment;
* press during agent output also preempts that output;
* release closes the segment and transcribes it as a whole;
* the terminal outcome is either commit(transcript) or reject(reason).

It does not consume streaming STT interim/final events and it does not use EOT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .ptt_segment import PttAudioSegmentRecorder
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
        tap_to_stop_max_audio_sec: float = 0.35,
    ) -> None:
        self._recorder = recorder
        self._transcriber = transcriber
        self._agent_output_active = agent_output_active or (lambda: False)
        self._preempt_agent_output = preempt_agent_output or (lambda: None)
        self._tap_to_stop_max_audio_sec = max(0.0, tap_to_stop_max_audio_sec)
        self._state: PttSegmentTurnState = "idle"
        self._preempted_agent_output = False

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
        if self._state == "transcribing":
            return PttSegmentTurnResult(
                action="reject",
                reason="busy_transcribing",
                state=self._state,
                preempted_agent_output=self._preempted_agent_output,
            )
        preempted = False
        if self._agent_output_active():
            self._preempt_agent_output()
            preempted = True
        self._recorder.start()
        self._state = "recording"
        self._preempted_agent_output = preempted
        return PttSegmentTurnResult(
            action="none",
            reason="pressed",
            state=self._state,
            preempted_agent_output=preempted,
        )

    def push_frame(self, frame: Any) -> bool:
        if self._state != "recording":
            return False
        return self._recorder.push_frame(frame)

    async def release(self) -> PttSegmentTurnResult:
        if self._state != "recording":
            return PttSegmentTurnResult(
                action="reject",
                reason="release_without_active_hold",
                state=self._state,
                preempted_agent_output=self._preempted_agent_output,
            )

        self._state = "transcribing"
        segment = self._recorder.stop()
        if (
            self._preempted_agent_output
            and segment.duration_sec <= self._tap_to_stop_max_audio_sec
        ):
            self._state = "idle"
            return PttSegmentTurnResult(
                action="reject",
                reason="tap_to_stop",
                state=self._state,
                preempted_agent_output=self._preempted_agent_output,
                audio_duration_sec=segment.duration_sec,
                audio_rms_ppm=segment.rms_ppm,
            )
        try:
            transcription = await self._transcriber.transcribe(segment)
        finally:
            self._state = "idle"
        if not transcription.accepted:
            return PttSegmentTurnResult(
                action="reject",
                reason=transcription.rejected_reason or "transcription_rejected",
                state=self._state,
                preempted_agent_output=self._preempted_agent_output,
                stt_mode=transcription.mode,
                stt_latency_ms=transcription.latency_ms,
                audio_duration_sec=transcription.audio_duration_sec,
                audio_rms_ppm=transcription.audio_rms_ppm,
            )

        return PttSegmentTurnResult(
            action="commit",
            reason="segment_transcribed",
            transcript=transcription.text,
            state=self._state,
            preempted_agent_output=self._preempted_agent_output,
            stt_mode=transcription.mode,
            stt_latency_ms=transcription.latency_ms,
            audio_duration_sec=transcription.audio_duration_sec,
            audio_rms_ppm=transcription.audio_rms_ppm,
        )
