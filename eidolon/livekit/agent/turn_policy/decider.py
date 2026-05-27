"""Pure turn-policy decisions for interruptions and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eidolon.livekit.common.config import InterruptPolicyConfig

from .intent_classifier import (
    InterruptIntent,
    InterruptIntentClassifier,
    InterruptIntentResult,
    LexiconInterruptClassifier,
    normalize_interrupt_text,
)


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
                    min_interim_chars
                    if min_interim_chars is not None
                    else base.min_interim_chars
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
        self._classifier = classifier or LexiconInterruptClassifier(base)

    def on_strong_intent(self) -> Decision:
        return Decision(
            action=Action.CANCEL,
            reason="strong_intent",
            intent=InterruptIntent.HARD_STOP,
            intent_source="legacy",
            intent_confidence=1.0,
        )

    def on_stt_interim(
        self,
        text: str,
        score: float,
        *,
        vad_active: bool = True,
        agent_speaking: bool = True,
    ) -> Decision:
        intent = self._classifier.classify(
            text,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            eot_score=score,
        )
        forced = self._decision_from_intent(intent)
        if forced is not None:
            return forced

        stripped = normalize_interrupt_text(text)
        if len(stripped) >= self._config.min_interim_chars:
            return Decision(
                action=Action.CANCEL,
                reason=f"first_signal_interim len={len(stripped)}",
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source=intent.source,
                intent_confidence=0.75,
            )

        if score >= self._config.early_cancel_score_threshold:
            return Decision(
                action=Action.CANCEL,
                reason=(
                    "eot_score_high "
                    f"score={score:.2f}>={self._config.early_cancel_score_threshold:.2f}"
                ),
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source="eot",
                intent_confidence=score,
            )

        if 0.0 < score <= self._config.early_resume_score_threshold:
            return Decision(
                action=Action.ROLLBACK,
                reason=(
                    "eot_score_low "
                    f"score={score:.2f}<={self._config.early_resume_score_threshold:.2f}"
                ),
                rollback_drop_buffered=False,
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
            )

        return Decision(
            action=Action.HOLD,
            reason=f"eot_score_mid score={score:.2f}",
            intent=intent.intent,
            intent_source=intent.source,
            intent_confidence=intent.confidence,
        )

    def on_decision_deadline(self, vad_still_active: bool) -> Decision:
        if vad_still_active:
            return Decision(
                action=Action.CANCEL,
                reason="deadline_trust_vad",
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source="timeout",
                intent_confidence=0.70,
            )
        return Decision(
            action=Action.ROLLBACK,
            reason="deadline_vad_idle_drop_stale",
            rollback_drop_buffered=True,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="timeout",
        )

    def on_user_silent(self) -> Decision:
        return Decision(
            action=Action.ROLLBACK,
            reason="user_silent_fast_rollback",
            rollback_drop_buffered=False,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="vad",
        )

    @staticmethod
    def _decision_from_intent(intent: InterruptIntentResult) -> Decision | None:
        if intent.intent == InterruptIntent.HARD_STOP:
            return Decision(
                action=Action.CANCEL,
                reason=f"intent:{intent.reason}",
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
            )
        if intent.intent == InterruptIntent.TOPIC_SWITCH:
            return Decision(
                action=Action.CANCEL,
                reason=f"intent:{intent.reason}",
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
                topic_switch_hint=True,
            )
        if intent.intent == InterruptIntent.CORRECTION:
            return Decision(
                action=Action.CANCEL,
                reason=f"intent:{intent.reason}",
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
                correction_hint=True,
            )
        if intent.intent in (InterruptIntent.BACKCHANNEL, InterruptIntent.NOISE):
            return Decision(
                action=Action.ROLLBACK,
                reason=f"intent:{intent.reason}",
                rollback_drop_buffered=False,
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
            )
        return None
