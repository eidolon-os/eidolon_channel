"""Runtime adapter that connects signal events to pure turn-policy decisions."""

from __future__ import annotations

from dataclasses import dataclass

from eidolon.livekit.common.config import TurnPolicyConfig

from .decider import Decision, InterruptDecider


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
    ) -> Decision:
        return self.decider.on_stt_interim(
            text,
            score,
            vad_active=vad_active,
            agent_speaking=agent_speaking,
        )

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

