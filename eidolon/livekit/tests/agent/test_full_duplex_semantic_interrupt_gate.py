from __future__ import annotations

import pytest

from eidolon.livekit.agent.full_duplex.semantic_interrupt_gate import (
    SemanticInterruptGateDecision,
    evaluate_semantic_interrupt_gate,
)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"allow_interruptions": False}, "interruptions_disabled"),
        ({"native_adaptive": True}, "native_adaptive_owner"),
        ({"transcript": ""}, "empty_transcript"),
        (
            {"agent_output_active": False, "interrupt_window_active": False},
            "no_interrupt_window",
        ),
    ],
)
def test_semantic_interrupt_gate_inactive_preconditions(
    kwargs: dict[str, object],
    reason: str,
) -> None:
    params = {
        "allow_interruptions": True,
        "native_adaptive": False,
        "transcript": "停一下",
        "agent_output_active": True,
        "interrupt_window_active": False,
        "decision_suppressed": False,
    }
    params.update(kwargs)
    decision = evaluate_semantic_interrupt_gate(**params)

    assert decision.action == "inactive"
    assert decision.reason == reason
    assert not decision.needs_attention
    assert not decision.should_run
    assert not decision.should_forward_and_stop


def test_semantic_interrupt_gate_forwards_and_stops_when_suppressed() -> None:
    decision = evaluate_semantic_interrupt_gate(
        allow_interruptions=True,
        native_adaptive=False,
        transcript="停一下",
        agent_output_active=True,
        interrupt_window_active=False,
        decision_suppressed=True,
    )

    assert decision.action == "forward_and_stop"
    assert decision.reason == "decision_suppressed"
    assert decision.should_forward_and_stop


def test_semantic_interrupt_gate_requires_attention_before_running() -> None:
    decision = evaluate_semantic_interrupt_gate(
        allow_interruptions=True,
        native_adaptive=False,
        transcript="那你现在能帮我做什么",
        agent_output_active=False,
        interrupt_window_active=True,
        decision_suppressed=False,
    )

    assert decision.action == "needs_attention"
    assert decision.needs_attention

    blocked = decision.with_attention_result(False)
    assert blocked.action == "forward_and_stop"
    assert blocked.reason == "attention_blocked"

    allowed = decision.with_attention_result(True)
    assert allowed.action == "run"
    assert allowed.reason == "eligible"
    assert allowed.should_run


def test_attention_result_is_noop_for_terminal_decision() -> None:
    decision = SemanticInterruptGateDecision(
        action="forward_and_stop",
        reason="decision_suppressed",
    )

    assert decision.with_attention_result(True) is decision
