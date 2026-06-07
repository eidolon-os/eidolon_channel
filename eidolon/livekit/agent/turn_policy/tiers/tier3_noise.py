"""Tier 3: backchannel, cough/noise, and false-interrupt rollback."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.constants import WEAK_SIGNAL_HOLD_REASON_PREFIXES
from eidolon.livekit.agent.turn_policy.decider import Action, Decision
from eidolon.livekit.agent.turn_policy.intent_classifier import InterruptIntent

from .model import Tier, TierEvidence


def classify(decision: Decision) -> TierEvidence | None:
    if decision.intent in (InterruptIntent.BACKCHANNEL, InterruptIntent.NOISE):
        return TierEvidence(Tier.TIER3_BACKCHANNEL_NOISE, decision.reason or "weak_signal")
    if decision.action is Action.ROLLBACK:
        return TierEvidence(Tier.TIER3_BACKCHANNEL_NOISE, decision.reason or "rollback")
    if decision.reason.startswith(WEAK_SIGNAL_HOLD_REASON_PREFIXES):
        return TierEvidence(Tier.TIER3_BACKCHANNEL_NOISE, decision.reason)
    return None
