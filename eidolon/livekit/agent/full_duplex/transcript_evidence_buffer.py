"""Correlate public LiveKit SpeechEvents with normalized transcript callbacks."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import logging
from typing import Any

from eidolon.livekit.common.transcript_evidence import (
    TranscriptEvidence,
    transcript_evidence_from_speech_event,
)

logger = logging.getLogger("agent.transcript_evidence")


@dataclass(frozen=True)
class _BufferedTranscriptEvidence:
    transcript: str
    is_final: bool
    evidence: TranscriptEvidence


class TranscriptEvidenceBuffer:
    """A bounded bridge across LiveKit's public ``Agent.stt_node`` boundary.

    LiveKit currently omits ``SpeechData.metadata`` when it emits
    ``UserInputTranscribedEvent`` for normal plugin STT.  The public ``stt_node``
    override observes the original event and this buffer pairs it with the
    immediately-following normalized callback without touching framework internals.
    """

    def __init__(self, *, max_entries: int = 64) -> None:
        self._max_entries = max(1, max_entries)
        self._entries: deque[_BufferedTranscriptEvidence] = deque(maxlen=self._max_entries)
        self._sequences: OrderedDict[tuple[str, str], int] = OrderedDict()

    def observe_speech_event(self, event: Any) -> bool:
        """Admit provider order before the SDK consumes text or runs endpointing.

        Sequence comparisons require explicit stream AND revision identities.
        Plain transcripts, different revisions and reconnects remain independent.
        Retain at most ``max_entries`` revision watermarks across buffer takes.
        """
        event_type = getattr(event, "type", "")
        event_value = str(getattr(event_type, "value", event_type) or "")
        if event_value not in {
            "interim_transcript",
            "preflight_transcript",
            "final_transcript",
        }:
            return True
        alternatives = getattr(event, "alternatives", None) or ()
        if not alternatives:
            return True
        transcript = str(getattr(alternatives[0], "text", "") or "").strip()
        if not transcript:
            return True
        evidence = transcript_evidence_from_speech_event(event)
        if not evidence.available:
            return True
        if evidence.stream_key and evidence.revision_key and evidence.sequence is not None:
            key = (evidence.stream_key, evidence.revision_key)
            previous = self._sequences.get(key)
            if previous is not None and evidence.sequence < previous:
                logger.info(
                    "[STT] dropped stale provider revision sequence=%s latest=%s",
                    evidence.sequence, previous,
                )
                return False
            self._sequences[key] = evidence.sequence
            self._sequences.move_to_end(key)
            if len(self._sequences) > self._max_entries:
                self._sequences.popitem(last=False)
        self._entries.append(
            _BufferedTranscriptEvidence(
                transcript=transcript,
                is_final=event_value == "final_transcript",
                evidence=evidence,
            )
        )
        return True

    def take(self, transcript: str, *, is_final: bool) -> TranscriptEvidence | None:
        target = transcript.strip()
        for index, item in enumerate(self._entries):
            if item.transcript == target and item.is_final is bool(is_final):
                del self._entries[index]
                return item.evidence
        return None

    def clear(self) -> None:
        self._entries.clear()
        self._sequences.clear()
