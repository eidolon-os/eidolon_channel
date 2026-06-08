"""Runtime adapter that connects signal events to pure turn-policy decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from eidolon.livekit.common.config import TurnPolicyConfig

from .attention import AttentionAdmission, AttentionDecision, AttentionInput
from .constants import (
    SEMANTIC_SCORE_WAIT_REASON_PREFIX,
    STABLE_NORMAL_INTERRUPT_REASON_PREFIX,
    STABLE_SIGNAL_WAIT_REASON_PREFIX,
    WEAK_SIGNAL_HOLD_REASON_PREFIXES,
)
from .decider import Action, Decision, InterruptDecider
from .intent_classifier import InterruptIntent, canonicalize_interrupt_text
from .tiers.chain import TierPolicyChain


@dataclass
class TurnControlSignal:
    intent: str
    confidence: float
    source: str
    reason: str
    topic_switch_hint: bool = False
    correction_hint: bool = False
    interrupted_text_excerpt: str = ""
    played_seconds: float = 0.0
    latency_ms: float = 0.0

    def as_metadata(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
            "topic_switch_hint": self.topic_switch_hint,
            "correction_hint": self.correction_hint,
            "interrupted_text_excerpt": self.interrupted_text_excerpt,
            "played_seconds": self.played_seconds,
            "latency_ms": self.latency_ms,
        }


class TurnPolicyRuntime:
    """Small orchestration layer around the pure decider.

    LiveKit-facing code owns side effects; this object owns policy state and
    conversion to metadata-friendly control signals.
    """

    def __init__(self, config: TurnPolicyConfig) -> None:
        self.config = config
        self.decider = InterruptDecider(config.interrupt)
        self.attention = AttentionAdmission(config)
        self.tiers = TierPolicyChain()
        self._stable_signal = _StableSignalStabilizer(config)
        self._stabilizer = _WeakSignalFollowupStabilizer(config)

    @property
    def decision_timeout_sec(self) -> float:
        return self.config.interrupt.decision_timeout_ms / 1000.0

    def decide_from_transcript(
        self,
        text: str,
        score: float,
        *,
        vad_active: bool,
        agent_speaking: bool,
        is_final: bool = False,
        event_time_ms: float | None = None,
    ) -> Decision:
        decision = self.decider.on_stt_interim(
            text,
            score,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
            is_final=is_final,
        )
        now_ms = event_time_ms if event_time_ms is not None else time.monotonic() * 1000
        decision = self._stable_signal.apply(
            decision,
            text=text,
            score=score,
            vad_active=vad_active,
            is_final=is_final,
            now_ms=now_ms,
        )
        decision = self._stabilizer.apply(decision, now_ms=now_ms)
        return self.tiers.annotate_decision(decision)

    def admit_attention(self, signal: AttentionInput) -> AttentionDecision:
        return self.tiers.annotate_attention(self.attention.decide(signal))

    def deadline_decision(
        self,
        vad_still_active: bool,
        *,
        has_transcript: bool = False,
        transcript: str = "",
        eot_score: float = 0.0,
    ) -> Decision:
        return self.tiers.annotate_decision(
            self.decider.on_decision_deadline(
                vad_still_active,
                has_transcript=has_transcript,
                transcript=transcript,
                eot_score=eot_score,
            )
        )

    def user_silent_decision(self, transcript: str = "") -> Decision:
        return self.tiers.annotate_decision(self.decider.on_user_silent(transcript))

    @staticmethod
    def control_signal_from_decision(
        decision: Decision,
        *,
        interrupted_text_excerpt: str = "",
        played_seconds: float = 0.0,
        latency_ms: float = 0.0,
    ) -> TurnControlSignal:
        return TurnControlSignal(
            intent=decision.intent.value if decision.intent is not None else "unknown",
            confidence=decision.intent_confidence,
            source=decision.intent_source,
            reason=decision.reason,
            topic_switch_hint=decision.topic_switch_hint,
            correction_hint=decision.correction_hint,
            interrupted_text_excerpt=interrupted_text_excerpt,
            played_seconds=played_seconds,
            latency_ms=latency_ms,
        )


@dataclass
class _StableCandidate:
    intent: InterruptIntent
    text: str
    first_seen_ms: float
    last_seen_ms: float
    event_count: int = 1


class _StableSignalStabilizer:
    """Require short transcript stability for non-hard-stop hot-path cancels."""

    def __init__(self, config: TurnPolicyConfig) -> None:
        self._config = config
        self._intent_candidate: _StableCandidate | None = None
        self._normal_candidate: _StableCandidate | None = None

    def apply(
        self,
        decision: Decision,
        *,
        text: str,
        score: float,
        vad_active: bool,
        is_final: bool,
        now_ms: float,
    ) -> Decision:
        if not vad_active:
            self._clear()
            return decision
        if decision.intent is InterruptIntent.HARD_STOP:
            self._clear()
            return decision
        if decision.action is Action.CANCEL and decision.intent in (
            InterruptIntent.CORRECTION,
            InterruptIntent.TOPIC_SWITCH,
        ):
            return self._stabilize_explicit_redirect(
                decision,
                text=text,
                is_final=is_final,
                now_ms=now_ms,
            )
        if self._is_normal_interrupt_wait(decision) and self._is_substantive_text(text):
            return self._stabilize_normal_interrupt(
                decision,
                text=text,
                score=score,
                now_ms=now_ms,
            )
        if decision.action in (Action.CANCEL, Action.ROLLBACK):
            self._clear()
        return decision

    def _stabilize_explicit_redirect(
        self,
        decision: Decision,
        *,
        text: str,
        is_final: bool,
        now_ms: float,
    ) -> Decision:
        window_ms = self._config.interrupt.correction_topic_stability_window_ms
        if window_ms <= 0 or is_final:
            self._intent_candidate = None
            return decision
        candidate = self._update_candidate(
            self._intent_candidate,
            intent=decision.intent or InterruptIntent.UNCERTAIN,
            text=text,
            now_ms=now_ms,
        )
        self._intent_candidate = candidate
        age_ms = now_ms - candidate.first_seen_ms
        if candidate.event_count >= 2 and age_ms >= window_ms:
            self._intent_candidate = None
            return decision
        return Decision(
            action=Action.HOLD,
            reason=(
                f"{STABLE_SIGNAL_WAIT_REASON_PREFIX} "
                f"intent={decision.intent.value if decision.intent else 'unknown'} "
                f"age_ms={age_ms:.0f} window_ms={window_ms}"
            ),
            intent=InterruptIntent.UNCERTAIN,
            intent_source=decision.intent_source or "stable_signal",
            intent_confidence=0.0,
            topic_switch_hint=decision.topic_switch_hint,
            correction_hint=decision.correction_hint,
            hold_recheck_ms=max(0.0, window_ms - age_ms),
        )

    def _stabilize_normal_interrupt(
        self,
        decision: Decision,
        *,
        text: str,
        score: float,
        now_ms: float,
    ) -> Decision:
        window_ms = self._config.interrupt.normal_interrupt_stability_window_ms
        if window_ms <= 0:
            self._normal_candidate = None
            return self._normal_cancel(decision, score=score, age_ms=0.0, window_ms=0)
        candidate = self._update_candidate(
            self._normal_candidate,
            intent=InterruptIntent.NORMAL_INTERRUPT,
            text=text,
            now_ms=now_ms,
        )
        self._normal_candidate = candidate
        age_ms = now_ms - candidate.first_seen_ms
        if candidate.event_count >= 2 and age_ms >= window_ms:
            self._normal_candidate = None
            return self._normal_cancel(
                decision,
                score=score,
                age_ms=age_ms,
                window_ms=window_ms,
            )
        return replace(decision, hold_recheck_ms=max(0.0, window_ms - age_ms))

    @staticmethod
    def _normal_cancel(
        decision: Decision,
        *,
        score: float,
        age_ms: float,
        window_ms: int,
    ) -> Decision:
        return Decision(
            action=Action.CANCEL,
            reason=(
                f"{STABLE_NORMAL_INTERRUPT_REASON_PREFIX} "
                f"age_ms={age_ms:.0f} window_ms={window_ms} "
                f"score={score:.2f} base_reason={decision.reason}"
            ),
            intent=InterruptIntent.NORMAL_INTERRUPT,
            intent_source="stable_signal",
            intent_confidence=max(0.70, score),
        )

    def _update_candidate(
        self,
        candidate: _StableCandidate | None,
        *,
        intent: InterruptIntent,
        text: str,
        now_ms: float,
    ) -> _StableCandidate:
        normalized = canonicalize_interrupt_text(text)
        if (
            candidate is None
            or candidate.intent is not intent
            or not self._is_text_consistent(candidate.text, normalized)
        ):
            return _StableCandidate(
                intent=intent,
                text=normalized,
                first_seen_ms=now_ms,
                last_seen_ms=now_ms,
            )
        return _StableCandidate(
            intent=intent,
            text=normalized if len(normalized) >= len(candidate.text) else candidate.text,
            first_seen_ms=candidate.first_seen_ms,
            last_seen_ms=now_ms,
            event_count=candidate.event_count + 1,
        )

    @staticmethod
    def _is_text_consistent(previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        return previous.startswith(current) or current.startswith(previous)

    @staticmethod
    def _is_normal_interrupt_wait(decision: Decision) -> bool:
        return (
            decision.action is Action.HOLD
            and decision.reason.startswith(SEMANTIC_SCORE_WAIT_REASON_PREFIX)
        )

    def _is_substantive_text(self, text: str) -> bool:
        normalized = canonicalize_interrupt_text(text)
        cjk = sum(1 for ch in normalized if "\u4e00" <= ch <= "\u9fff")
        latin = sum(1 for ch in normalized if "a" <= ch.lower() <= "z")
        intr = self._config.interrupt
        return (
            cjk >= intr.min_normal_interim_cjk_chars
            or latin > intr.latin_artifact_hold_max_chars
        )

    def _clear(self) -> None:
        self._intent_candidate = None
        self._normal_candidate = None


class _WeakSignalFollowupStabilizer:
    """Hold normal-interrupt candidates briefly after weak/noisy evidence."""

    def __init__(self, config: TurnPolicyConfig) -> None:
        self._config = config
        self._last_weak_signal_ms: float | None = None

    def apply(self, decision: Decision, *, now_ms: float) -> Decision:
        if self._is_weak_signal_hold(decision):
            self._last_weak_signal_ms = now_ms
            return decision
        if self._should_hold_after_weak_signal(decision, now_ms):
            return Decision(
                action=Action.HOLD,
                reason=(
                    "weak_signal_followup_hold "
                    f"age_ms={now_ms - (self._last_weak_signal_ms or now_ms):.0f}"
                ),
                intent=InterruptIntent.UNCERTAIN,
                intent_source=decision.intent_source or "runtime",
                intent_confidence=0.0,
            )
        if decision.action is Action.CANCEL:
            self._last_weak_signal_ms = None
        return decision

    def _is_weak_signal_hold(self, decision: Decision) -> bool:
        if decision.action is not Action.HOLD:
            return False
        return decision.reason.startswith(WEAK_SIGNAL_HOLD_REASON_PREFIXES)

    def _should_hold_after_weak_signal(self, decision: Decision, now_ms: float) -> bool:
        window_ms = self._config.interrupt.weak_signal_followup_hold_ms
        if window_ms <= 0 or self._last_weak_signal_ms is None:
            return False
        if now_ms - self._last_weak_signal_ms > window_ms:
            return False
        if (
            decision.intent_source == "eot"
            and decision.intent_confidence
            >= self._config.interrupt.early_cancel_score_threshold
        ):
            return False
        return (
            decision.action is Action.CANCEL
            and decision.intent == InterruptIntent.NORMAL_INTERRUPT
        )
