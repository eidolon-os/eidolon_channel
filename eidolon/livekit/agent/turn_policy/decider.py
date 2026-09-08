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
    normalize_transcript_text,
    intent_requires_reply,
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
    RESUME = "resume"
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
        intent_result: InterruptIntentResult | None = None,
    ) -> Decision:
        intent = intent_result or self.classify_intent_hint(
            text,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            eot_score=score,
        )
        stripped = normalize_transcript_text(text)
        evidence = self._evidence_gate.evaluate(
            stripped,
            is_final=is_final,
            eot_score=score,
        )
        hint_fields = self._hint_fields(intent)

        if self._config.intent_provider == "llm":
            # Model meaning and provider finality are independent evidence.
            # A transient prefix must never authorize an irreversible cut.
            if not is_final or intent_result is None:
                return Decision(Action.HOLD, "awaiting_final_intent", **hint_fields)
            if not stripped:
                return Decision(Action.HOLD, "empty_intent_transcript", **hint_fields)
            if intent.intent is InterruptIntent.HARD_STOP or intent_requires_reply(intent.intent):
                return Decision(Action.CANCEL, "confirmed_final_intent", **hint_fields)
            if not vad_active:
                return Decision(
                    Action.ROLLBACK, "final_intent_resume", rollback_drop_buffered=False,
                    **hint_fields,
                )
            return Decision(Action.HOLD, "intent_wait_for_speech_end", **hint_fields)

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

        # Sentence finality and low completeness are not evidence of a false
        # interruption. The owner bounds waiting and playback separately.
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
        if self._config.intent_provider == "llm":
            # The bounded classifier request or the owner deadline settles it;
            # the old EOT-only timer must not race the semantic decision.
            if vad_still_active or has_transcript:
                return Decision(Action.HOLD, "awaiting_final_intent")
            return Decision(Action.ROLLBACK, "deadline_no_speech_evidence")
        if vad_still_active:
            if not has_transcript:
                return Decision(
                    action=Action.HOLD,
                    reason="deadline_wait_for_transcript",
                    intent=InterruptIntent.UNCERTAIN,
                    intent_source="timeout",
                    intent_confidence=0.0,
                )
            text = normalize_transcript_text(transcript)
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
            reason="deadline_vad_idle_resume",
            rollback_drop_buffered=False,
            intent=InterruptIntent.UNCERTAIN,
            intent_source="timeout",
        )

    def on_user_silent(self, transcript: str = "") -> Decision:
        text = normalize_transcript_text(transcript)
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
