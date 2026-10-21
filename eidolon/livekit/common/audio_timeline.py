"""Map sent ASR samples to normalized provider-input samples; never store PCM.

The input coordinate is AFTER the existing SDK resampler, BEFORE rechunking,
VAD billing gates and synthetic keepalive. It is not a wall-clock timestamp or
the 24 kHz stt_node coordinate. Source + generation distinguish connection runs.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class AudioSource:
    source: str
    generation: int
    sample_rate: int


@dataclass(frozen=True)
class AudioSpan(AudioSource):
    start: int
    end: int

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, value: Any):
        if not isinstance(value, dict) or set(value) != {
            "source",
            "generation",
            "sample_rate",
            "start",
            "end",
        }:
            return None
        if not isinstance(value["source"], str) or not 0 < len(value["source"]) <= 128:
            return None
        if any(type(value[k]) is not int for k in ("generation", "sample_rate", "start", "end")):
            return None
        if (
            value["generation"] < 0
            or value["sample_rate"] <= 0
            or value["start"] < 0
            or value["end"] <= value["start"]
        ):
            return None
        return cls(**value)

    def same_source(self, other: AudioSource) -> bool:
        return (self.source, self.generation, self.sample_rate) == (
            other.source,
            other.generation,
            other.sample_rate,
        )


class ProviderAudioTimeline:
    def __init__(
        self,
        *,
        source: str,
        generation: int,
        sample_rate: int,
        max_samples: int,
        max_entries: int = 4096,
    ):
        if not source or generation < 0 or sample_rate <= 0 or max_samples <= 0 or max_entries <= 0:
            raise ValueError("scoped source and positive limits required")
        self.source, self.generation, self.sample_rate = source, generation, sample_rate
        self._max_samples = max_samples
        self._spans: deque[tuple[int, int, int | None]] = deque(maxlen=max_entries)
        self._sent = 0

    def record(self, *, samples: int, input_start: int | None):
        if (
            type(samples) is not int
            or samples <= 0
            or (input_start is not None and (type(input_start) is not int or input_start < 0))
        ):
            raise ValueError("aligned positive sample count and optional input position required")
        self._spans.append((self._sent, self._sent + samples, input_start))
        self._sent += samples
        floor = self._sent - self._max_samples
        while self._spans and self._spans[0][1] <= floor:
            self._spans.popleft()

    def resolve_ms(self, start_ms: float, end_ms: float) -> AudioSpan | None:
        if (
            not math.isfinite(start_ms)
            or not math.isfinite(end_ms)
            or start_ms < 0
            or end_ms <= start_ms
        ):
            return None
        start = math.floor(start_ms * self.sample_rate / 1000)
        end = math.ceil(end_ms * self.sample_rate / 1000)
        if start < self._sent - self._max_samples or end > self._sent:
            return None
        cursor, first, last = start, None, None
        for wire_start, wire_end, input_start in self._spans:
            if wire_end <= cursor:
                continue
            if wire_start > cursor or input_start is None:
                return None
            mapped = input_start + cursor - wire_start
            if last is not None and mapped != last:
                return None  # gate omitted input or replayed a prior chunk
            first = mapped if first is None else first
            stop = min(end, wire_end)
            last = mapped + stop - cursor
            cursor = stop
            if cursor == end:
                return AudioSpan(self.source, self.generation, self.sample_rate, first, last)
        return None
