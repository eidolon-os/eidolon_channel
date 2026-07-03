"""Complete-audio transcription for half-duplex PTT turns."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .ptt_segment import PttAudioSegment

PttTranscriptionStrategy = Literal["auto", "offline", "streaming"]


@dataclass(frozen=True)
class PttSegmentTranscriberConfig:
    strategy: PttTranscriptionStrategy = "streaming"
    min_audio_duration_sec: float = 0.08
    min_rms_ppm: int = 0


@dataclass(frozen=True)
class PttSegmentTranscriptionResult:
    text: str = ""
    mode: Literal["offline", "streaming", "none"] = "none"
    rejected_reason: str = ""
    latency_ms: float = 0.0
    audio_duration_sec: float = 0.0
    audio_rms_ppm: int = 0

    @property
    def accepted(self) -> bool:
        return bool(self.text.strip()) and not self.rejected_reason


class PttSegmentTranscriber:
    """Transcribe the closed PTT audio window as one logical user turn."""

    def __init__(
        self,
        stt_stage: Any,
        *,
        config: PttSegmentTranscriberConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._stt_stage = stt_stage
        self._config = config or PttSegmentTranscriberConfig()
        self._clock = clock or time.monotonic

    async def transcribe(self, segment: PttAudioSegment) -> PttSegmentTranscriptionResult:
        duration = segment.duration_sec
        rms_ppm = segment.rms_ppm
        if not segment.audio:
            return self._reject("empty_audio", segment)
        if duration < self._config.min_audio_duration_sec:
            return self._reject("audio_too_short", segment)
        if self._config.min_rms_ppm > 0 and rms_ppm < self._config.min_rms_ppm:
            return self._reject("audio_below_rms_threshold", segment)

        started = self._clock()
        mode = self._select_mode()
        try:
            if mode == "offline":
                text = await self._stt_stage.recognize(segment.audio)
            else:
                text = await self._stt_stage.recognize_streaming(segment.audio)
        except NotImplementedError:
            if self._config.strategy != "auto" or mode != "offline":
                raise
            mode = "streaming"
            text = await self._stt_stage.recognize_streaming(segment.audio)

        latency_ms = (self._clock() - started) * 1000
        stripped = (text or "").strip()
        if not stripped:
            return PttSegmentTranscriptionResult(
                mode=mode,
                rejected_reason="empty_transcript",
                latency_ms=latency_ms,
                audio_duration_sec=duration,
                audio_rms_ppm=rms_ppm,
            )
        return PttSegmentTranscriptionResult(
            text=stripped,
            mode=mode,
            latency_ms=latency_ms,
            audio_duration_sec=duration,
            audio_rms_ppm=rms_ppm,
        )

    def _reject(
        self,
        reason: str,
        segment: PttAudioSegment,
    ) -> PttSegmentTranscriptionResult:
        return PttSegmentTranscriptionResult(
            rejected_reason=reason,
            audio_duration_sec=segment.duration_sec,
            audio_rms_ppm=segment.rms_ppm,
        )

    def _select_mode(self) -> Literal["offline", "streaming"]:
        strategy = self._config.strategy
        if strategy == "offline":
            return "offline"
        if strategy == "streaming":
            return "streaming"
        return "offline" if self._supports_offline_recognize() else "streaming"

    def _supports_offline_recognize(self) -> bool:
        stt_plugin = getattr(self._stt_stage, "stt", None)
        capabilities = getattr(stt_plugin, "capabilities", None)
        return bool(getattr(capabilities, "offline_recognize", False))
