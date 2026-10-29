"""Bounded sample-addressed evidence, independent of VAD and identity policy.

The existing input owner supplies offsets after resampling. This object never
subscribes to a room or infers candidate boundaries from wall-clock events.
"""

import logging
from collections import deque
from dataclasses import dataclass, field, replace

from .audio_timeline import AudioSource, AudioSpan


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
                self._chunks[0] = (floor, first[(floor - offset) * 2 :])
                break
            else:
                break

    def snapshot(self, *, candidate: str, start: int, end: int, generation: int):
        if not candidate or start < 0 or end <= start:
            raise ValueError("nonempty candidate and sample range required")

        def result(reason, pcm=b""):
            return AudioEvidence(
                self.source,
                generation,
                candidate,
                self.sample_rate,
                start,
                end,
                reason is None,
                reason,
                pcm,
            )

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
            parts.append(pcm[(cursor - offset) * 2 : (stop - offset) * 2])
            cursor = stop
            if cursor == end:
                return result(None, b"".join(parts))
        return result("missing_samples")

    def clear(self):
        self._chunks.clear()
        # Offsets never rewind within a generation. A different input source
        # requires a new buffer/generation, not reuse of previous identities.


class AudioEvidenceCapture:
    """One optional, bounded capture owned by a fresh selected session input.

    The provider writes normalized PCM before rechunking or billing gates. No
    track subscription, timers, inference or storage. Input invalidation is
    terminal for this capture: reconnecting ASR cannot establish that upstream
    SDK queues contain only a new physical source. A new capture requires a
    new input boundary established by its owner, not a delayed callback.
    """

    def __init__(self, *, session_id: str, endpoint_id: str, max_seconds: float = 30):
        # Retention also covers final-transcript latency after speech stops.
        # This is a memory ceiling, not the admitted utterance duration.
        if not session_id or not endpoint_id or not 0 < max_seconds <= 60:
            raise ValueError("session, endpoint and bounded duration required")
        self._session_id = session_id
        self._endpoint_id = endpoint_id
        self._max_seconds = max_seconds
        self._selected_stream: str | None = None
        self._generation = -1
        self._active_source: AudioSource | None = None
        self._buffer: AudioEvidenceBuffer | None = None
        self._invalid_reason: str | None = None
        self.listener = None

    @property
    def retention_seconds(self) -> float:
        return self._max_seconds

    @property
    def active_source(self) -> AudioSource | None:
        return self._active_source

    @property
    def retained_samples(self) -> int:
        return self._buffer.retained_samples if self._buffer else 0

    def _notify(self, method, *args, **kwargs):
        listener = self.listener
        if listener is None:
            return
        try:
            getattr(listener, method)(*args, **kwargs)
        except Exception:
            # Optional inference must never fail the STT/media owner.
            self.listener = None
            logging.getLogger(__name__).warning("Audio evidence listener unavailable")
            try:
                listener.source_ended()
            except Exception:
                pass

    def _clear(self) -> None:
        self._notify("source_ended")
        if self._buffer is not None:
            self._buffer.clear()
        self._buffer = None
        self._active_source = None

    def select_stream(self, source: str) -> None:
        """Called synchronously on stream creation, never by a retry task."""
        if self._invalid_reason or source == self._selected_stream:
            return
        self._clear()
        self._selected_stream = source
        self._generation = -1

    def begin_run(self, source: AudioSource) -> None:
        if (
            self._invalid_reason
            or source.source != self._selected_stream
            or source.generation <= self._generation
        ):
            return
        self._clear()
        self._generation = source.generation
        self._buffer = AudioEvidenceBuffer(
            source=source.source,
            generation=source.generation,
            sample_rate=source.sample_rate,
            max_samples=max(1, int(source.sample_rate * self._max_seconds)),
        )
        self._active_source = source
        self._notify("source_started", source)

    def append(self, source: AudioSource, *, start: int, pcm: bytes) -> None:
        if source != self._active_source or self._buffer is None:
            return
        try:
            self._buffer.append(start=start, pcm=pcm)
            self._notify("append", source, start=start, pcm=pcm)
        except ValueError:
            # Evidence faults must not break normal media delivery. Fail closed
            # for this input instead of silently returning a later valid suffix.
            self.invalidate("invalid_audio_sequence")

    def end_run(self, source: AudioSource) -> None:
        if source == self._active_source:
            self._clear()

    def invalidate(self, reason: str) -> None:
        if not reason:
            raise ValueError("invalidation reason required")
        self._invalid_reason = self._invalid_reason or reason
        self._clear()

    def snapshot(self, *, candidate: str, span: AudioSpan) -> AudioEvidence:
        if not candidate or span.start < 0 or span.end <= span.start:
            raise ValueError("candidate and positive sample range required")
        reason = self._invalid_reason
        if reason is None and (
            self._active_source is None or not span.same_source(self._active_source)
        ):
            reason = "audio_source_changed"
        if reason is not None:
            return AudioEvidence(
                source=span.source,
                generation=span.generation,
                candidate=candidate,
                sample_rate=span.sample_rate,
                start=span.start,
                end=span.end,
                complete=False,
                reason=reason,
                pcm=b"",
                session_id=self._session_id,
                endpoint_id=self._endpoint_id,
            )
        assert self._buffer is not None
        return replace(
            self._buffer.snapshot(
                candidate=candidate, start=span.start, end=span.end, generation=span.generation
            ),
            session_id=self._session_id,
            endpoint_id=self._endpoint_id,
        )
