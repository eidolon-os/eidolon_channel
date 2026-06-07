"""Tier 0: explicit hard stop, the fastest cancellation path."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.decider import Decision
from eidolon.livekit.agent.turn_policy.intent_classifier import InterruptIntent

from .model import Tier, TierEvidence


def classify(decision: Decision) -> TierEvidence | None:
    if decision.intent is InterruptIntent.HARD_STOP:
        return TierEvidence(Tier.TIER0_HARD_STOP, decision.reason or "hard_stop")
    if decision.reason in {"strong_intent", "explicit_client_interrupt"}:
        return TierEvidence(Tier.TIER0_HARD_STOP, decision.reason)
    return None
