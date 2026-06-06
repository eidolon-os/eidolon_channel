"""Transcript evidence gate for irreversible interrupt decisions."""

from __future__ import annotations

from dataclasses import dataclass

from eidolon.livekit.common.config import InterruptPolicyConfig


@dataclass(frozen=True)
class TranscriptEvidence:
    allow_cancel: bool
    reason: str
    cjk_chars: int
    latin_chars: int


class TranscriptEvidenceGate:
    """Separate transcript quality from interrupt intent classification.

    A first ASR interim can be useful enough to duck or hold, but it should not
    always be strong enough to cancel current agent audio. This gate keeps the
    hot path deterministic while filtering common low-quality artifacts such as
    short latin-only fragments in a Chinese room ("If", "I", "OK").
    """

    def __init__(self, config: InterruptPolicyConfig | None = None) -> None:
        self._config = config or InterruptPolicyConfig()

    def evaluate(
        self,
        text: str,
        *,
        is_final: bool = False,
        eot_score: float = 0.0,
    ) -> TranscriptEvidence:
        stripped = text.strip()
        cjk = _count_cjk(stripped)
        latin = _count_latin(stripped)
        if not self._config.transcript_evidence_gate_enabled:
            return TranscriptEvidence(True, "gate_disabled", cjk, latin)
        if not stripped:
            return TranscriptEvidence(False, "empty_transcript", cjk, latin)
        if is_final:
            return TranscriptEvidence(True, "final_transcript", cjk, latin)
        if (
            cjk == 0
            and latin > 0
            and latin <= self._config.latin_artifact_hold_max_chars
        ):
            return TranscriptEvidence(False, "short_latin_artifact", cjk, latin)
        if cjk >= self._config.min_normal_interim_cjk_chars:
            return TranscriptEvidence(True, "enough_cjk_interim", cjk, latin)
        if eot_score >= self._config.early_cancel_score_threshold and cjk > 0:
            return TranscriptEvidence(True, "high_eot_with_cjk", cjk, latin)
        return TranscriptEvidence(False, "insufficient_transcript_evidence", cjk, latin)


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _count_latin(text: str) -> int:
    return sum(1 for ch in text if "a" <= ch.lower() <= "z")
