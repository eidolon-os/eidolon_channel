"""Provider-neutral optional evidence attached to LiveKit STT events.

The canonical STT protocol remains LiveKit's ``SpeechEvent``/``SpeechData``.
This module only describes product evidence that is not part of LiveKit's
portable event contract.  Every field is optional: providers with only
``text`` and ``is_final`` remain fully supported.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping


TRANSCRIPT_EVIDENCE_METADATA_KEY = "eidolon.transcript_evidence"


class TranscriptEvidenceCapability(str, Enum):
    REVISION_IDENTITY = "revision_identity"
    SOURCE_TIMESTAMPS = "source_timestamps"
    WORD_TIMESTAMPS = "word_timestamps"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PROVIDER_END_OF_SPEECH = "provider_end_of_speech"
    STABLE_PREFIX = "stable_prefix"
    PROVIDER_SEQUENCE = "provider_sequence"


class TranscriptBoundary(str, Enum):
    SENTENCE_BEGIN = "sentence_begin"
    SENTENCE_END = "sentence_end"
    PROVIDER_END_OF_SPEECH = "provider_end_of_speech"


@dataclass(frozen=True)
class TranscriptSourceSpan:
    start_time: float
    end_time: float

    def overlaps(self, other: TranscriptSourceSpan) -> bool:
        return max(self.start_time, other.start_time) <= min(self.end_time, other.end_time)


@dataclass(frozen=True)
class TranscriptEvidence:
    """Optional facts about one transcript revision.

    ``stream_key`` and ``revision_key`` are opaque identities.  Core turn code
    never interprets their provider format and never branches on provider name.
    """

    stream_key: str | None = None
    revision_key: str | None = None
    sequence: int | None = None
    source_span: TranscriptSourceSpan | None = None
    word_spans: tuple[TranscriptSourceSpan, ...] = ()
    boundaries: frozenset[TranscriptBoundary] = frozenset()
    stable_prefix: str | None = None

    @property
    def capabilities(self) -> frozenset[TranscriptEvidenceCapability]:
        values: set[TranscriptEvidenceCapability] = set()
        if self.revision_key:
            values.add(TranscriptEvidenceCapability.REVISION_IDENTITY)
        if self.source_span is not None:
            values.add(TranscriptEvidenceCapability.SOURCE_TIMESTAMPS)
        if self.word_spans:
            values.add(TranscriptEvidenceCapability.WORD_TIMESTAMPS)
        if self.boundaries & {
            TranscriptBoundary.SENTENCE_BEGIN,
            TranscriptBoundary.SENTENCE_END,
        }:
            values.add(TranscriptEvidenceCapability.SENTENCE_BOUNDARY)
        if TranscriptBoundary.PROVIDER_END_OF_SPEECH in self.boundaries:
            values.add(TranscriptEvidenceCapability.PROVIDER_END_OF_SPEECH)
        if self.stable_prefix:
            values.add(TranscriptEvidenceCapability.STABLE_PREFIX)
        if self.sequence is not None:
            values.add(TranscriptEvidenceCapability.PROVIDER_SEQUENCE)
        return frozenset(values)

    @property
    def available(self) -> bool:
        return bool(
            self.stream_key
            or self.revision_key
            or self.sequence is not None
            or self.source_span is not None
            or self.word_spans
            or self.boundaries
            or self.stable_prefix
        )

    def same_revision(self, other: TranscriptEvidence) -> bool:
        if not self.revision_key or not other.revision_key:
            return False
        if self.stream_key and other.stream_key and self.stream_key != other.stream_key:
            return False
        return self.revision_key == other.revision_key

    def same_source_span(self, other: TranscriptEvidence) -> bool:
        if self.source_span is None or other.source_span is None:
            return False
        if self.stream_key and other.stream_key and self.stream_key != other.stream_key:
            return False
        return self.source_span.overlaps(other.source_span)

    def merge(self, other: TranscriptEvidence | None) -> TranscriptEvidence:
        """Fill missing fields from ``other`` while preserving this evidence.

        Callers combining successive transcript revisions should invoke this on
        the newer revision so growing timestamps, sequence numbers, word spans,
        and stable prefixes replace stale values.  Boundaries are cumulative.
        """

        if other is None or not other.available:
            return self
        if not self.available:
            return other
        return replace(
            self,
            stream_key=self.stream_key or other.stream_key,
            revision_key=self.revision_key or other.revision_key,
            sequence=self.sequence if self.sequence is not None else other.sequence,
            source_span=self.source_span or other.source_span,
            word_spans=self.word_spans or other.word_spans,
            boundaries=self.boundaries | other.boundaries,
            stable_prefix=self.stable_prefix or other.stable_prefix,
        )

    def as_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.stream_key:
            payload["stream_key"] = self.stream_key
        if self.revision_key:
            payload["revision_key"] = self.revision_key
        if self.sequence is not None:
            payload["sequence"] = self.sequence
        if self.source_span is not None:
            payload["source_span"] = {
                "start_time": self.source_span.start_time,
                "end_time": self.source_span.end_time,
            }
        if self.word_spans:
            payload["word_spans"] = [
                {"start_time": span.start_time, "end_time": span.end_time}
                for span in self.word_spans
            ]
        if self.boundaries:
            payload["boundaries"] = sorted(boundary.value for boundary in self.boundaries)
        if self.stable_prefix:
            payload["stable_prefix"] = self.stable_prefix
        return {TRANSCRIPT_EVIDENCE_METADATA_KEY: payload}


def transcript_evidence_from_speech_event(event: Any) -> TranscriptEvidence:
    """Extract generic evidence from a public LiveKit ``SpeechEvent``."""

    alternatives = getattr(event, "alternatives", None) or ()
    data = alternatives[0] if alternatives else None
    metadata = getattr(data, "metadata", None) if data is not None else None
    evidence = transcript_evidence_from_metadata(metadata)

    request_id = str(getattr(event, "request_id", "") or "")
    source_span = transcript_source_span(
        getattr(data, "start_time", None),
        getattr(data, "end_time", None),
    )
    word_spans = _word_spans(getattr(data, "words", None) or ())
    standard = TranscriptEvidence(
        stream_key=request_id or None,
        source_span=source_span,
        word_spans=word_spans,
    )
    return evidence.merge(standard)


def transcript_evidence_from_metadata(metadata: Any) -> TranscriptEvidence:
    if not isinstance(metadata, Mapping):
        return TranscriptEvidence()
    raw = metadata.get(TRANSCRIPT_EVIDENCE_METADATA_KEY, metadata)
    if not isinstance(raw, Mapping):
        return TranscriptEvidence()

    source_raw = raw.get("source_span")
    source_span = (
        transcript_source_span(source_raw.get("start_time"), source_raw.get("end_time"))
        if isinstance(source_raw, Mapping)
        else None
    )
    word_spans_raw = raw.get("word_spans")
    word_spans: list[TranscriptSourceSpan] = []
    if isinstance(word_spans_raw, Iterable) and not isinstance(word_spans_raw, (str, bytes)):
        for item in word_spans_raw:
            if not isinstance(item, Mapping):
                continue
            span = transcript_source_span(item.get("start_time"), item.get("end_time"))
            if span is not None:
                word_spans.append(span)

    boundaries_raw = raw.get("boundaries") or ()
    if isinstance(boundaries_raw, str):
        boundaries_raw = (boundaries_raw,)
    boundaries: set[TranscriptBoundary] = set()
    for value in boundaries_raw:
        try:
            boundaries.add(TranscriptBoundary(str(value)))
        except ValueError:
            continue

    sequence_raw = raw.get("sequence")
    try:
        sequence = int(sequence_raw) if sequence_raw is not None else None
    except (TypeError, ValueError):
        sequence = None
    return TranscriptEvidence(
        stream_key=_optional_string(raw.get("stream_key")),
        revision_key=_optional_string(raw.get("revision_key")),
        sequence=sequence,
        source_span=source_span,
        word_spans=tuple(word_spans),
        boundaries=frozenset(boundaries),
        stable_prefix=_optional_string(raw.get("stable_prefix")),
    )


def transcript_evidence_from_user_event(event: Any) -> TranscriptEvidence:
    """Extract fields exposed by LiveKit's normalized user transcript event."""

    evidence = transcript_evidence_from_metadata(getattr(event, "metadata", None))
    item_id = _optional_string(getattr(event, "item_id", None))
    if item_id:
        evidence = evidence.merge(TranscriptEvidence(revision_key=item_id))
    return evidence


def transcript_source_span(start: Any, end: Any) -> TranscriptSourceSpan | None:
    """Build a source span only when a provider supplied a valid interval."""

    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError):
        return None
    if start_value < 0 or end_value < start_value:
        return None
    if start_value == 0.0 and end_value == 0.0:
        return None
    return TranscriptSourceSpan(start_time=start_value, end_time=end_value)


def _word_spans(words: Iterable[Any]) -> tuple[TranscriptSourceSpan, ...]:
    result: list[TranscriptSourceSpan] = []
    for word in words:
        span = transcript_source_span(
            getattr(word, "start_time", None),
            getattr(word, "end_time", None),
        )
        if span is not None:
            result.append(span)
    return tuple(result)


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
