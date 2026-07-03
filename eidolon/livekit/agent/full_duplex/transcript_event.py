"""Normalized transcript event shape for full-duplex sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FullDuplexTranscriptEvent:
    transcript: str = ""
    speaker_id: str | None = None
    is_final: bool = False

    @classmethod
    def from_event(cls, event: Any) -> FullDuplexTranscriptEvent:
        if isinstance(event, cls):
            return event
        return cls(
            transcript=getattr(event, "transcript", "") or "",
            speaker_id=getattr(event, "speaker_id", None),
            is_final=bool(getattr(event, "is_final", False)),
        )

    @property
    def has_transcript(self) -> bool:
        return bool(self.transcript)

    @property
    def timeline_mark(self) -> str:
        return "transcript_final_at" if self.is_final else "transcript_interim_first_at"
