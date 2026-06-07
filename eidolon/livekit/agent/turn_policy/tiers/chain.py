"""Tier policy chain adapter.

This first migration step does not change the underlying policy behavior. It
adds one explicit place where an already-produced decision is mapped onto the
five-tier architecture, so logs and benchmark reports can speak the same
language as the design.
"""

from __future__ import annotations

from dataclasses import replace

from eidolon.livekit.agent.turn_policy.attention import AttentionDecision
from eidolon.livekit.agent.turn_policy.decider import Decision

from . import tier0_hard_stop, tier1_redirect, tier2_interruption, tier3_noise
from .model import Tier, TierEvidence
from .tier4_attention import classify_attention


class TierPolicyChain:
    """Annotate decisions with the tier that dominates them."""

    def annotate_decision(self, decision: Decision) -> Decision:
        evidence = self._decision_tier(decision)
        return replace(
            decision,
            tier=evidence.tier.value,
            tier_reason=evidence.reason,
        )

    def annotate_attention(self, decision: AttentionDecision) -> AttentionDecision:
        evidence = classify_attention(decision)
        if evidence is None:
            return decision
        return replace(
            decision,
            tier=evidence.tier.value,
            tier_reason=evidence.reason,
        )

    @staticmethod
    def _decision_tier(decision: Decision) -> TierEvidence:
        for classifier in (
            tier0_hard_stop.classify,
            tier1_redirect.classify,
            tier3_noise.classify,
            tier2_interruption.classify,
        ):
            evidence = classifier(decision)
            if evidence is not None:
                return evidence
        return TierEvidence(Tier.UNKNOWN, decision.reason or "unclassified")
