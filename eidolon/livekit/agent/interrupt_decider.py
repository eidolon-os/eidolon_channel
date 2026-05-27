"""Backward import shim for the turn-policy decider module."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy.decider import Action, Decision, InterruptDecider
from eidolon.livekit.agent.turn_policy.intent_classifier import (
    LexiconInterruptClassifier,
    normalize_interrupt_text,
)


def is_backchannel_text(text: str) -> bool:
    cfg = __import__(
        "eidolon.livekit.common.config",
        fromlist=["InterruptPolicyConfig"],
    ).InterruptPolicyConfig()
    classifier = LexiconInterruptClassifier(cfg)
    result = classifier.classify(
        text,
        vad_active=True,
        agent_speaking=True,
        eot_score=0.0,
    )
    return result.intent.value in ("backchannel", "noise")


__all__ = [
    "Action",
    "Decision",
    "InterruptDecider",
    "is_backchannel_text",
    "normalize_interrupt_text",
]
