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
    canonicalize_interrupt_text,
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
        self._semantic_prefixes = tuple(
            normalize_interrupt_text(item)
            for item in (
                tuple(base.topic_switch_lexicon)
                + tuple(base.correction_lexicon)
            )
        )

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
        stripped = canonicalize_interrupt_text(text)
        forced = self._decision_from_intent(
            intent,
            normalized_text=stripped,
            vad_active=vad_active,
        )
        if forced is not None:
            return forced

        if self._is_semantic_prefix(stripped):
            return Decision(
                action=Action.HOLD,
                reason=f"semantic_prefix text={stripped}",
                intent=InterruptIntent.UNCERTAIN,
                intent_source="lexicon",
                intent_confidence=0.0,
            )

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

    def on_decision_deadline(
        self,
        vad_still_active: bool,
        *,
        has_transcript: bool = False,
        transcript: str = "",
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
            if len(text) < self._config.min_interim_chars:
                return Decision(
                    action=Action.HOLD,
                    reason=f"deadline_wait_for_more_transcript len={len(text)}",
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="timeout",
                    intent_confidence=0.0,
                )
            intent = self._classifier.classify(
                transcript,
                vad_active=True,
                agent_speaking=True,
                eot_score=0.0,
            )
            forced = self._decision_from_intent(
                intent,
                normalized_text=text,
                vad_active=True,
            )
            if forced is not None:
                return forced
            if self._is_semantic_prefix(text):
                return Decision(
                    action=Action.HOLD,
                    reason=f"deadline_semantic_prefix text={text}",
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="lexicon",
                    intent_confidence=0.0,
                )
            return Decision(
                action=Action.CANCEL,
                reason="deadline_trust_vad_with_transcript",
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

    def on_user_silent(self, transcript: str = "") -> Decision:
        text = canonicalize_interrupt_text(transcript)
        if text:
            intent = self._classifier.classify(
                transcript,
                vad_active=False,
                agent_speaking=True,
                eot_score=0.0,
            )
            if intent.intent in (InterruptIntent.BACKCHANNEL, InterruptIntent.NOISE):
                return Decision(
                    action=Action.ROLLBACK,
                    reason=f"user_silent_intent:{intent.reason}",
                    rollback_drop_buffered=False,
                    intent=intent.intent,
                    intent_source=intent.source,
                    intent_confidence=intent.confidence,
                )
        return Decision(
            action=Action.ROLLBACK,
            reason="user_silent_fast_rollback",
            rollback_drop_buffered=False,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="vad",
        )

    @staticmethod
    def _decision_from_intent(
        intent: InterruptIntentResult,
        *,
        normalized_text: str,
        vad_active: bool,
    ) -> Decision | None:
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
            if vad_active and len(normalized_text) <= 1:
                return Decision(
                    action=Action.HOLD,
                    reason=f"intent:{intent.reason}_await_more_speech",
                    intent=intent.intent,
                    intent_source=intent.source,
                    intent_confidence=intent.confidence,
                )
            return Decision(
                action=Action.ROLLBACK,
                reason=f"intent:{intent.reason}",
                rollback_drop_buffered=False,
                intent=intent.intent,
                intent_source=intent.source,
                intent_confidence=intent.confidence,
            )
        return None

    def _is_semantic_prefix(self, text: str) -> bool:
        if len(text) < self._config.min_interim_chars:
            return False
        return any(
            candidate.startswith(text) and candidate != text
            for candidate in self._semantic_prefixes
        )
