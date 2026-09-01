"""Pure turn-policy decisions for interruptions and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from eidolon.livekit.common.config import InterruptPolicyConfig
from .intent_classifier import (
    InterruptIntent,
    InterruptIntentClassifier,
    InterruptIntentResult,
    NoopModelInterruptClassifier,
    canonicalize_interrupt_text,
)
from .constants import (
    DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX,
    SEMANTIC_SCORE_WAIT_REASON_PREFIX,
    TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX,
    WEAK_SIGNAL_SHORT_TRANSCRIPT_REASON_PREFIX,
)
from .evidence import TranscriptEvidenceGate


class Action(Enum):
    NONE = "none"
    CANCEL = "cancel"
    ROLLBACK = "rollback"
    HOLD = "hold"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    rollback_drop_buffered: bool = False
    intent: InterruptIntent | None = None
    intent_source: str = ""
    intent_confidence: float = 0.0
    topic_switch_hint: bool = False
    correction_hint: bool = False
    tier: str = ""
    tier_reason: str = ""
    hold_recheck_ms: float | None = None


class _HintFields(TypedDict):
    intent: InterruptIntent
    intent_source: str
    intent_confidence: float
    topic_switch_hint: bool
    correction_hint: bool


class InterruptDecider:
    """Pure decision policy. No side effects, timers, or LiveKit calls."""

    def __init__(
        self,
        config: InterruptPolicyConfig | None = None,
        *,
        classifier: InterruptIntentClassifier | None = None,
        min_interim_chars: int | None = None,
        early_cancel_score_threshold: float | None = None,
        early_resume_score_threshold: float | None = None,
    ) -> None:
        base = config or InterruptPolicyConfig()
        if (
            min_interim_chars is not None
            or early_cancel_score_threshold is not None
            or early_resume_score_threshold is not None
        ):
            from dataclasses import replace

            base = replace(
                base,
                min_interim_chars=(
                    min_interim_chars if min_interim_chars is not None else base.min_interim_chars
                ),
                early_cancel_score_threshold=(
                    early_cancel_score_threshold
                    if early_cancel_score_threshold is not None
                    else base.early_cancel_score_threshold
                ),
                early_resume_score_threshold=(
                    early_resume_score_threshold
                    if early_resume_score_threshold is not None
                    else base.early_resume_score_threshold
                ),
            )
        self._config = base
        self._classifier = classifier or NoopModelInterruptClassifier()
        self._evidence_gate = TranscriptEvidenceGate(base)

    def classify_intent_hint(
        self,
        text: str,
        *,
        vad_active: bool,
        agent_speaking: bool,
        eot_score: float,
    ) -> InterruptIntentResult:
        """Return a replaceable hint; callers must supply decision evidence."""

        return self._classifier.classify(
            text,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            eot_score=eot_score,
        )

    def on_stt_interim(
        self,
        text: str,
        score: float,
        *,
        vad_active: bool = True,
        agent_speaking: bool = True,
        is_final: bool = False,
    ) -> Decision:
        intent = self.classify_intent_hint(
            text,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            eot_score=score,
        )
        stripped = canonicalize_interrupt_text(text)
        evidence = self._evidence_gate.evaluate(
            stripped,
            is_final=is_final,
            eot_score=score,
        )
        hint_fields = self._hint_fields(intent)

        if not evidence.allow_cancel:
            if len(stripped) < self._config.min_interim_chars and score == 0.0:
                reason = f"{WEAK_SIGNAL_SHORT_TRANSCRIPT_REASON_PREFIX} len={len(stripped)}"
            else:
                reason = (
                    f"{TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX}{evidence.reason} "
                    f"cjk={evidence.cjk_chars} latin={evidence.latin_chars}"
                )
            return Decision(
                action=Action.HOLD,
                reason=reason,
                **hint_fields,
            )

        if score >= self._config.early_cancel_score_threshold:
            return Decision(
                action=Action.CANCEL,
                reason=(
                    "eot_score_high "
                    f"score={score:.2f}>={self._config.early_cancel_score_threshold:.2f} "
                    f"evidence={evidence.reason}"
                ),
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source="eot",
                intent_confidence=score,
            )

        if is_final and not vad_active and 0.0 < score <= self._config.early_resume_score_threshold:
            return Decision(
                action=Action.ROLLBACK,
                reason=(
                    "final_terminal_eot_low "
                    f"score={score:.2f}<={self._config.early_resume_score_threshold:.2f}"
                ),
                rollback_drop_buffered=False,
                intent=InterruptIntent.UNCERTAIN,
                intent_source="eot",
                intent_confidence=score,
            )

        return Decision(
            action=Action.HOLD,
            reason=(
                f"{SEMANTIC_SCORE_WAIT_REASON_PREFIX} "
                f"score={score:.2f} evidence={evidence.reason}"
                f"{' final=true' if is_final else ''}"
            ),
            **hint_fields,
        )

    def on_decision_deadline(
        self,
        vad_still_active: bool,
        *,
        has_transcript: bool = False,
        transcript: str = "",
        eot_score: float = 0.0,
    ) -> Decision:
        if vad_still_active:
            if not has_transcript:
                return Decision(
                    action=Action.HOLD,
                    reason="deadline_wait_for_transcript",
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="timeout",
                    intent_confidence=0.0,
                )
            text = canonicalize_interrupt_text(transcript)
            intent = self.classify_intent_hint(
                transcript,
                vad_active=True,
                agent_speaking=True,
                eot_score=0.0,
            )
            if len(text) < self._config.min_interim_chars:
                return Decision(
                    action=Action.HOLD,
                    reason=f"deadline_wait_for_more_transcript len={len(text)}",
                    **self._hint_fields(intent),
                )
            evidence = self._evidence_gate.evaluate(
                text,
                is_final=False,
                eot_score=eot_score,
            )
            if not evidence.allow_cancel:
                return Decision(
                    action=Action.HOLD,
                    reason=(
                        f"{DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX}{evidence.reason} "
                        f"cjk={evidence.cjk_chars} latin={evidence.latin_chars}"
                    ),
                    **self._hint_fields(intent),
                )
            if eot_score < self._config.early_cancel_score_threshold:
                return Decision(
                    action=Action.HOLD,
                    reason=(
                        f"deadline_wait_for_semantic_score:{evidence.reason} score={eot_score:.2f}"
                    ),
                    **self._hint_fields(intent),
                )
            return Decision(
                action=Action.CANCEL,
                reason=(
                    "deadline_trust_vad_with_transcript "
                    f"score={eot_score:.2f} evidence={evidence.reason}"
                ),
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source="eot",
                intent_confidence=eot_score,
            )
        if has_transcript:
            intent = self.classify_intent_hint(
                transcript,
                vad_active=False,
                agent_speaking=True,
                eot_score=eot_score,
            )
            return Decision(
                action=Action.HOLD,
                reason="deadline_wait_for_final_commit",
                **self._hint_fields(intent),
            )
        return Decision(
            action=Action.ROLLBACK,
            reason="deadline_vad_idle_drop_stale",
            rollback_drop_buffered=True,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="timeout",
        )

    def on_user_silent(self, transcript: str = "") -> Decision:
        text = canonicalize_interrupt_text(transcript)
        if text:
            intent = self.classify_intent_hint(
                transcript,
                vad_active=False,
                agent_speaking=True,
                eot_score=0.0,
            )
            return Decision(
                action=Action.HOLD,
                reason="user_silent_wait_for_final_commit",
                **self._hint_fields(intent),
            )
        return Decision(
            action=Action.ROLLBACK,
            reason="user_silent_fast_rollback",
            rollback_drop_buffered=False,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="vad",
        )

    @staticmethod
    def _hint_fields(intent: InterruptIntentResult) -> _HintFields:
        return {
            "intent": intent.intent,
            "intent_source": intent.source,
            "intent_confidence": intent.confidence,
            "topic_switch_hint": intent.intent is InterruptIntent.TOPIC_SWITCH,
            "correction_hint": intent.intent is InterruptIntent.CORRECTION,
        }
