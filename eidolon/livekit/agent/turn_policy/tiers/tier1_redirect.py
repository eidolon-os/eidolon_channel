"""Tier 1: explicit correction, topic switch, or redirect."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.decider import Decision
from eidolon.livekit.agent.turn_policy.intent_classifier import InterruptIntent

from .model import Tier, TierEvidence


def classify(decision: Decision) -> TierEvidence | None:
    if decision.intent in (InterruptIntent.TOPIC_SWITCH, InterruptIntent.CORRECTION):
        return TierEvidence(Tier.TIER1_REDIRECT, decision.reason or "redirect")
    if decision.topic_switch_hint:
        return TierEvidence(Tier.TIER1_REDIRECT, decision.reason or "topic_switch")
    if decision.correction_hint:
        return TierEvidence(Tier.TIER1_REDIRECT, decision.reason or "correction")
    return None
