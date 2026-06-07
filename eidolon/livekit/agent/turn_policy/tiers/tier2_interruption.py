"""Tier 2: substantive ordinary interruption."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.constants import (
    SEMANTIC_SCORE_WAIT_REASON_PREFIX,
    STABLE_NORMAL_INTERRUPT_REASON_PREFIX,
)
from eidolon.livekit.agent.turn_policy.decider import Action, Decision
from eidolon.livekit.agent.turn_policy.intent_classifier import InterruptIntent

from .model import Tier, TierEvidence


def classify(decision: Decision) -> TierEvidence | None:
    if decision.intent is InterruptIntent.NORMAL_INTERRUPT:
        return TierEvidence(Tier.TIER2_INTERRUPTION, decision.reason or "normal_interrupt")
    if decision.reason.startswith(STABLE_NORMAL_INTERRUPT_REASON_PREFIX):
        return TierEvidence(Tier.TIER2_INTERRUPTION, decision.reason)
    if decision.reason.startswith(SEMANTIC_SCORE_WAIT_REASON_PREFIX):
        return TierEvidence(Tier.TIER2_INTERRUPTION, decision.reason)
    if decision.action is Action.CANCEL and decision.intent is InterruptIntent.UNCERTAIN:
        return TierEvidence(Tier.TIER2_INTERRUPTION, decision.reason or "uncertain_cancel")
    return None
