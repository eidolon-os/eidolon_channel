"""Timeline-safe speaker verification signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SpeakerSignal:
    """Observe-only speaker identity result for one user turn."""

    provider: str
    model: str
    speaker_user_id: str | None = None
    known: bool = False
    owner_confidence: float | None = None
    score: float | None = None
    latency_ms: float | None = None
    audio_ms: int | None = None
    profile_id: str | None = None
    error: str | None = None

    @classmethod
    def error_signal(
        cls,
        *,
        provider: str,
        model: str,
        error: str,
        latency_ms: float | None = None,
        audio_ms: int | None = None,
        profile_id: str | None = None,
    ) -> "SpeakerSignal":
        return cls(
            provider=provider,
            model=model,
            known=False,
            latency_ms=latency_ms,
            audio_ms=audio_ms,
            profile_id=profile_id,
            error=error,
        )

    def as_timeline_attrs(self) -> dict[str, Any]:
        """Return a compact JSON-serializable payload for timeline attrs."""
        return {k: v for k, v in asdict(self).items() if v is not None}
