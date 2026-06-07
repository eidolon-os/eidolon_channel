"""Tier 4: client/playback-state attention admission."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.attention import AdmissionAction, AttentionDecision

from .model import Tier, TierEvidence


def classify_attention(decision: AttentionDecision) -> TierEvidence | None:
    if decision.reason in {
        "client_mic_muted",
        "client_playback_active_without_direct_signal",
    }:
        return TierEvidence(Tier.TIER4_ATTENTION, decision.reason)
    if decision.action in (AdmissionAction.IGNORE, AdmissionAction.OBSERVE):
        return TierEvidence(Tier.TIER4_ATTENTION, decision.reason or decision.action.value)
    return None
