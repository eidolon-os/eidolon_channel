"""Bounded sample-addressed evidence, independent of VAD and identity policy.

The existing input owner supplies offsets after resampling. This object never
subscribes to a room or infers candidate boundaries from wall-clock events.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioEvidence:
    source: str
    generation: int
    candidate: str
    sample_rate: int
    start: int
    end: int
    complete: bool
    reason: str | None
    pcm: bytes = field(repr=False)
    session_id: str = ""
    endpoint_id: str = ""


class AudioEvidenceBuffer:
    def __init__(self, *, source: str, generation: int, sample_rate: int, max_samples: int):
        if not source or generation < 0 or sample_rate <= 0 or max_samples <= 0:
            raise ValueError("source, generation and positive audio limits required")
        self.source = source
        self.generation = generation
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self._chunks: deque[tuple[int, bytes]] = deque()
        self._end = 0

    @property
    def retained_samples(self):
        return sum(len(pcm) // 2 for _, pcm in self._chunks)

    def append(self, *, start: int, pcm: bytes):
        if start < self._end:
            raise ValueError("overlap or out-of-order audio")
        if start < 0 or not pcm or len(pcm) % 2:
            raise ValueError("nonempty aligned PCM16 required")
        data = bytes(pcm)
        self._end = start + len(data) // 2
        self._chunks.append((start, data))
        # Bound by sample timeline span, not chunk count. A gap cannot retain
        # an arbitrarily old frame and later masquerade as contiguous audio.
        floor = self._end - self.max_samples
        while self._chunks:
            offset, first = self._chunks[0]
            end = offset + len(first) // 2
            if end <= floor:
                self._chunks.popleft()
            elif offset < floor:
                self._chunks[0] = (floor, first[(floor - offset) * 2:])
                break
            else:
                break

    def snapshot(self, *, candidate: str, start: int, end: int, generation: int):
        if not candidate or start < 0 or end <= start:
            raise ValueError("nonempty candidate and sample range required")

        def result(reason, pcm=b""):
            return AudioEvidence(self.source, generation, candidate, self.sample_rate,
                                 start, end, reason is None, reason, pcm)

        if generation != self.generation:
            return result("source_generation_changed")
        if end - start > self.max_samples:
            return result("range_exceeds_limit")
        cursor, parts = start, []
        for offset, pcm in self._chunks:
            chunk_end = offset + len(pcm) // 2
            if chunk_end <= cursor:
                continue
            if offset > cursor:
                return result("missing_samples")
            stop = min(end, chunk_end)
            parts.append(pcm[(cursor - offset) * 2:(stop - offset) * 2])
            cursor = stop
            if cursor == end:
                return result(None, b"".join(parts))
        return result("missing_samples")

    def clear(self):
        self._chunks.clear()
        # Offsets never rewind within a generation. A different input source
        # requires a new buffer/generation, not reuse of previous identities.
