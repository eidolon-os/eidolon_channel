"""Normalized transcript event shape for full-duplex sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eidolon.livekit.common.transcript_evidence import (
    TranscriptEvidence,
    transcript_evidence_from_user_event,
)


@dataclass(frozen=True)
class FullDuplexTranscriptEvent:
    transcript: str = ""
    speaker_id: str | None = None
    is_final: bool = False
    language: str | None = None
    item_id: str | None = None
    created_at: float | None = None
    evidence: TranscriptEvidence = TranscriptEvidence()

    @classmethod
    def from_event(cls, event: Any) -> FullDuplexTranscriptEvent:
        if isinstance(event, cls):
            return event
        item_id = getattr(event, "item_id", None)
        return cls(
            transcript=getattr(event, "transcript", "") or "",
            speaker_id=getattr(event, "speaker_id", None),
            is_final=bool(getattr(event, "is_final", False)),
            language=getattr(event, "language", None),
            item_id=str(item_id) if item_id else None,
            created_at=getattr(event, "created_at", None),
            evidence=transcript_evidence_from_user_event(event),
        )

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript)

    @property
    def timeline_mark(self) -> str:
        return "transcript_final_at" if self.is_final else "transcript_interim_first_at"
