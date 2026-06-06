"""Runtime adapter that connects signal events to pure turn-policy decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass

from eidolon.livekit.common.config import TurnPolicyConfig

from .attention import AttentionAdmission, AttentionDecision, AttentionInput
from .decider import Action, Decision, InterruptDecider
from .intent_classifier import InterruptIntent


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
        return self._stabilizer.apply(decision, now_ms=now_ms)

    def admit_attention(self, signal: AttentionInput) -> AttentionDecision:
        return self.attention.decide(signal)

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


class _WeakSignalFollowupStabilizer:
    """Hold normal-interrupt candidates briefly after weak/noisy evidence."""

    _WEAK_HOLD_REASON_PREFIXES = (
        "transcript_evidence_hold:",
        "deadline_wait_for_better_transcript:",
        "intent:noise_",
        "intent:backchannel_",
    )

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
        return decision.reason.startswith(self._WEAK_HOLD_REASON_PREFIXES)

    def _should_hold_after_weak_signal(self, decision: Decision, now_ms: float) -> bool:
        window_ms = self._config.interrupt.weak_signal_followup_hold_ms
        if window_ms <= 0 or self._last_weak_signal_ms is None:
            return False
        if now_ms - self._last_weak_signal_ms > window_ms:
            return False
        return (
            decision.action is Action.CANCEL
            and decision.intent == InterruptIntent.NORMAL_INTERRUPT
        )
