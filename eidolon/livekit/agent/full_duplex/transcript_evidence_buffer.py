"""Correlate public LiveKit SpeechEvents with normalized transcript callbacks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from eidolon.livekit.common.transcript_evidence import (
    TranscriptEvidence,
    transcript_evidence_from_speech_event,
)


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
        self._entries: deque[_BufferedTranscriptEvidence] = deque(maxlen=max(1, max_entries))

    def observe_speech_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "")
        event_value = str(getattr(event_type, "value", event_type) or "")
        if event_value not in {
            "interim_transcript",
            "preflight_transcript",
            "final_transcript",
        }:
            return
        alternatives = getattr(event, "alternatives", None) or ()
        if not alternatives:
            return
        transcript = str(getattr(alternatives[0], "text", "") or "").strip()
        if not transcript:
            return
        evidence = transcript_evidence_from_speech_event(event)
        if not evidence.available:
            return
        self._entries.append(
            _BufferedTranscriptEvidence(
                transcript=transcript,
                is_final=event_value == "final_transcript",
                evidence=evidence,
            )
        )

    def take(self, transcript: str, *, is_final: bool) -> TranscriptEvidence | None:
        target = transcript.strip()
        for index, item in enumerate(self._entries):
            if item.transcript == target and item.is_final is bool(is_final):
                del self._entries[index]
                return item.evidence
        return None

    def clear(self) -> None:
        self._entries.clear()
