"""PTT audio segment capture.

Press/release owns the user audio window in half-duplex mode.  This recorder is
side-effect-light: it only buffers frames while held and returns one immutable
segment on release.  It does not run STT, VAD, EOT, or interruption policy.
"""

from __future__ import annotations

import sys
import time
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

PttRecorderState = Literal["idle", "recording"]
DEFAULT_ACTIVE_AUDIO_RMS_PPM = 500
DEFAULT_ACTIVE_AUDIO_WINDOW_MS = 20


@dataclass(frozen=True)
class PttAudioSegmentConfig:
    sample_rate: int = 16_000
    num_channels: int = 1
    sample_width_bytes: int = 2
    max_duration_sec: float = 20.0


@dataclass(frozen=True)
class PttAudioSegment:
    audio: bytes
    sample_rate: int
    num_channels: int
    sample_width_bytes: int
    started_at: float
    ended_at: float
    frame_count: int
    truncated: bool = False

    @property
    def duration_sec(self) -> float:
        bytes_per_sample = self.sample_width_bytes * max(1, self.num_channels)
        if bytes_per_sample <= 0:
            return 0.0
        samples = len(self.audio) // bytes_per_sample
        return samples / float(self.sample_rate) if self.sample_rate > 0 else 0.0

    @property
    def rms_ppm(self) -> int:
        if not self.audio:
            return 0
        rms = _pcm_rms(self.audio, sample_width_bytes=self.sample_width_bytes)
        return int(round((rms / 32768.0) * 1_000_000))

    def leading_silence_sec(
        self,
        *,
        rms_threshold_ppm: int = DEFAULT_ACTIVE_AUDIO_RMS_PPM,
        window_ms: int = DEFAULT_ACTIVE_AUDIO_WINDOW_MS,
    ) -> float:
        """Return leading low-energy audio before the first active window."""

        if not self.audio or self.sample_width_bytes != 2 or self.sample_rate <= 0:
            return 0.0

        bytes_per_sample = self.sample_width_bytes * max(1, self.num_channels)
        even = len(self.audio) - (len(self.audio) % bytes_per_sample)
        if even <= 0:
            return 0.0

        threshold = max(0, int(rms_threshold_ppm))
        window_samples = max(1, int(self.sample_rate * max(1, window_ms) / 1000))
        window_bytes = window_samples * bytes_per_sample
        silence_bytes = 0
        for offset in range(0, even, window_bytes):
            chunk = self.audio[offset : min(offset + window_bytes, even)]
            if _pcm_rms_ppm(chunk, sample_width_bytes=self.sample_width_bytes) >= threshold:
                break
            silence_bytes += len(chunk)
        else:
            silence_bytes = even

        samples = silence_bytes // bytes_per_sample
        return samples / float(self.sample_rate)

    def active_duration_sec(
        self,
        *,
        rms_threshold_ppm: int = DEFAULT_ACTIVE_AUDIO_RMS_PPM,
        window_ms: int = DEFAULT_ACTIVE_AUDIO_WINDOW_MS,
    ) -> float:
        leading = self.leading_silence_sec(
            rms_threshold_ppm=rms_threshold_ppm,
            window_ms=window_ms,
        )
        return max(0.0, self.duration_sec - leading)


class PttAudioSegmentRecorder:
    """Buffer one press-to-release PCM segment."""

    def __init__(
        self,
        *,
        config: PttAudioSegmentConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or PttAudioSegmentConfig()
        self._clock = clock or time.monotonic
        self._state: PttRecorderState = "idle"
        self._started_at = 0.0
        self._chunks: list[bytes] = []
        self._frame_count = 0
        self._truncated = False

    @property
    def state(self) -> PttRecorderState:
        return self._state

    def start(self, *, now: float | None = None) -> None:
        self._state = "recording"
        self._started_at = self._now(now)
        self._chunks = []
        self._frame_count = 0
        self._truncated = False

    def push_frame(self, frame: Any) -> bool:
        if self._state != "recording":
            return False

        chunk = _frame_bytes(frame)
        if not chunk:
            return False

        max_bytes = self._max_audio_bytes()
        current_bytes = sum(len(part) for part in self._chunks)
        remaining = max_bytes - current_bytes
        if remaining <= 0:
            self._truncated = True
            return False
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
            self._truncated = True

        self._chunks.append(chunk)
        self._frame_count += 1
        return True

    def stop(self, *, now: float | None = None) -> PttAudioSegment:
        ended_at = self._now(now)
        if self._state != "recording":
            return PttAudioSegment(
                audio=b"",
                sample_rate=self._config.sample_rate,
                num_channels=self._config.num_channels,
                sample_width_bytes=self._config.sample_width_bytes,
                started_at=ended_at,
                ended_at=ended_at,
                frame_count=0,
            )

        segment = PttAudioSegment(
            audio=b"".join(self._chunks),
            sample_rate=self._config.sample_rate,
            num_channels=self._config.num_channels,
            sample_width_bytes=self._config.sample_width_bytes,
            started_at=self._started_at,
            ended_at=ended_at,
            frame_count=self._frame_count,
            truncated=self._truncated,
        )
        self.reset()
        return segment

    def reset(self) -> None:
        self._state = "idle"
        self._started_at = 0.0
        self._chunks = []
        self._frame_count = 0
        self._truncated = False

    def _max_audio_bytes(self) -> int:
        return int(
            self._config.sample_rate
            * self._config.max_duration_sec
            * self._config.num_channels
            * self._config.sample_width_bytes
        )

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now


def _frame_bytes(frame: Any) -> bytes:
    data = getattr(frame, "data", frame)
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    tobytes = getattr(data, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    return bytes(data)


def _pcm_rms(audio: bytes, *, sample_width_bytes: int) -> float:
    if sample_width_bytes != 2:
        return 0.0
    even = len(audio) - (len(audio) % sample_width_bytes)
    if even <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(audio[:even])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return mean_square**0.5


def _pcm_rms_ppm(audio: bytes, *, sample_width_bytes: int) -> int:
    rms = _pcm_rms(audio, sample_width_bytes=sample_width_bytes)
    return int(round((rms / 32768.0) * 1_000_000))
