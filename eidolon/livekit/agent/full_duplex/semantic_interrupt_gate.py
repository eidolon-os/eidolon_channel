"""Semantic interrupt trigger gate for full-duplex transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticInterruptGateAction = Literal[
    "inactive",
    "needs_attention",
    "run",
    "forward_and_stop",
]


@dataclass(frozen=True)
class SemanticInterruptGateDecision:
    action: SemanticInterruptGateAction
    reason: str

    @property
    def needs_attention(self) -> bool:
        return self.action == "needs_attention"

    @property
    def should_run(self) -> bool:
        return self.action == "run"

    @property
    def should_forward_and_stop(self) -> bool:
        return self.action == "forward_and_stop"

    def with_attention_result(self, allowed: bool) -> SemanticInterruptGateDecision:
        if not self.needs_attention:
            return self
        if not allowed:
            return SemanticInterruptGateDecision(
                action="forward_and_stop",
                reason="attention_blocked",
            )
        return SemanticInterruptGateDecision(action="run", reason="eligible")


def evaluate_semantic_interrupt_gate(
    *,
    allow_interruptions: bool,
    native_adaptive: bool,
    transcript: str,
    agent_output_active: bool,
    interrupt_window_active: bool,
    decision_suppressed: bool,
) -> SemanticInterruptGateDecision:
    if not allow_interruptions:
        return SemanticInterruptGateDecision("inactive", "interruptions_disabled")
    if native_adaptive:
        return SemanticInterruptGateDecision("inactive", "native_adaptive_owner")
    if not transcript:
        return SemanticInterruptGateDecision("inactive", "empty_transcript")
    if not (agent_output_active or interrupt_window_active):
        return SemanticInterruptGateDecision("inactive", "no_interrupt_window")
    if decision_suppressed:
        return SemanticInterruptGateDecision(
            action="forward_and_stop",
            reason="decision_suppressed",
        )
    return SemanticInterruptGateDecision("needs_attention", "needs_attention")
