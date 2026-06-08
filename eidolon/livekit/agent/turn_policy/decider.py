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
    is_semantic_interrupt_prefix,
    semantic_interrupt_prefix_intent,
)
from .constants import (
    DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX,
    SEMANTIC_SCORE_WAIT_REASON_PREFIX,
    TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX,
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
        self._classifier = classifier or LexiconInterruptClassifier()
        self._evidence_gate = TranscriptEvidenceGate(base)

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
        is_final: bool = False,
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

        prefix_intent = semantic_interrupt_prefix_intent(
            stripped,
            min_chars=self._config.min_interim_chars,
            min_cjk_chars=self._config.min_normal_interim_cjk_chars,
        )
        if prefix_intent in (InterruptIntent.TOPIC_SWITCH, InterruptIntent.CORRECTION):
            return Decision(
                action=Action.CANCEL,
                reason=f"prefix_intent:{prefix_intent.value} text={stripped}",
                intent=prefix_intent,
                intent_source="lexicon_prefix",
                intent_confidence=0.65,
                topic_switch_hint=prefix_intent is InterruptIntent.TOPIC_SWITCH,
                correction_hint=prefix_intent is InterruptIntent.CORRECTION,
            )

        if is_semantic_interrupt_prefix(
            stripped,
            min_chars=self._config.min_interim_chars,
        ):
            return Decision(
                action=Action.HOLD,
                reason=f"semantic_prefix text={stripped}",
                intent=InterruptIntent.UNCERTAIN,
                intent_source="lexicon",
                intent_confidence=0.0,
            )

        if len(stripped) >= self._config.min_interim_chars:
            evidence = self._evidence_gate.evaluate(
                stripped,
                is_final=is_final,
                eot_score=score,
            )
            if not evidence.allow_cancel:
                return Decision(
                    action=Action.HOLD,
                    reason=(
                        f"{TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX}{evidence.reason} "
                        f"cjk={evidence.cjk_chars} latin={evidence.latin_chars}"
                    ),
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source=intent.source,
                    intent_confidence=0.0,
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
            return Decision(
                action=Action.HOLD,
                reason=(
                    f"{SEMANTIC_SCORE_WAIT_REASON_PREFIX} "
                    f"score={score:.2f} evidence={evidence.reason}"
                    f"{' final=true' if is_final else ''}"
                ),
                intent=InterruptIntent.UNCERTAIN,
                intent_source=intent.source,
                intent_confidence=0.0,
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
            if is_semantic_interrupt_prefix(
                text,
                min_chars=self._config.min_interim_chars,
            ):
                return Decision(
                    action=Action.HOLD,
                    reason=f"deadline_semantic_prefix text={text}",
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="lexicon",
                    intent_confidence=0.0,
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
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="timeout",
                    intent_confidence=0.0,
                )
            if eot_score < self._config.early_cancel_score_threshold:
                return Decision(
                    action=Action.HOLD,
                    reason=(
                        "deadline_wait_for_semantic_score:"
                        f"{evidence.reason} score={eot_score:.2f}"
                    ),
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="timeout",
                    intent_confidence=0.0,
                )
            return Decision(
                action=Action.CANCEL,
                reason=(
                    "deadline_trust_vad_with_transcript "
                    f"score={eot_score:.2f} evidence={evidence.reason}"
                ),
                intent=InterruptIntent.NORMAL_INTERRUPT,
                intent_source="timeout",
                intent_confidence=max(0.70, eot_score),
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
