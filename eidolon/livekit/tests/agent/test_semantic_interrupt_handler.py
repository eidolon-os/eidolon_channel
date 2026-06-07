"""Focused tests for the semantic interrupt session helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.semantic_interrupt import SemanticInterruptHandler
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent


def _eot_model(
    *,
    strong_intent: bool = False,
    should_interrupt: bool = False,
    score: float = 0.0,
) -> MagicMock:
    model = MagicMock()
    model._turn_end_policy.is_strong_interrupt_intent.return_value = strong_intent
    model.should_interrupt.return_value = should_interrupt
    model.current_eot_score = score
    model.hard_interrupt_score_threshold = 0.75
    return model


def _handler(
    *,
    eot_model: MagicMock,
    turn_runtime: MagicMock,
    duck_active: bool = False,
    vad_active: bool = True,
    soft_interrupt_active: bool = False,
) -> tuple[SemanticInterruptHandler, SimpleNamespace]:
    calls = SimpleNamespace(
        apply_decision=MagicMock(),
        record_decision_attrs=MagicMock(),
        publish_turn_control=MagicMock(),
        cancel_duck_and_interrupt=MagicMock(),
        interrupt_current_turn=MagicMock(),
        enter_soft_interrupt=MagicMock(),
        timeline=MagicMock(),
    )
    handler = SemanticInterruptHandler(
        get_eot_model=lambda: eot_model,
        turn_runtime=turn_runtime,
        get_timeline=lambda: calls.timeline,
        get_duck_active=lambda: duck_active,
        get_duck_stats=lambda: DuckingStats(suspend_ms=123.0),
        get_vad_active=lambda: vad_active,
        soft_interrupt_active=lambda: soft_interrupt_active,
        soft_interrupt_timeout=lambda: 0.5,
        apply_decision=calls.apply_decision,
        record_decision_attrs=calls.record_decision_attrs,
        publish_turn_control=calls.publish_turn_control,
        cancel_duck_and_interrupt=calls.cancel_duck_and_interrupt,
        interrupt_current_turn=calls.interrupt_current_turn,
        enter_soft_interrupt=calls.enter_soft_interrupt,
    )
    return handler, calls


def test_strong_interrupt_without_duck_interrupts_immediately() -> None:
    runtime = MagicMock()
    decision = Decision(action=Action.CANCEL, reason="hard_stop")
    runtime.strong_intent_decision.return_value = decision
    runtime.control_signal_from_decision.return_value.as_metadata.return_value = {
        "action": "cancel",
    }
    handler, calls = _handler(
        eot_model=_eot_model(strong_intent=True),
        turn_runtime=runtime,
        duck_active=False,
    )

    handler.run("别说了")

    calls.publish_turn_control.assert_called_once_with({"action": "cancel"})
    calls.record_decision_attrs.assert_called_once()
    calls.cancel_duck_and_interrupt.assert_not_called()
    calls.interrupt_current_turn.assert_called_once()
    calls.timeline.set_attr.assert_any_call("turn_control", {"action": "cancel"})
    calls.timeline.set_attr.assert_any_call("cancel_reason", "strong_intent_cancel")


def test_duck_active_delegates_decision_to_effect_applier() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.CANCEL,
        reason="semantic_cancel",
        intent=InterruptIntent.CORRECTION,
    )
    runtime.decide_from_transcript.return_value = decision
    handler, calls = _handler(
        eot_model=_eot_model(score=0.82),
        turn_runtime=runtime,
        duck_active=True,
    )

    handler.run("我刚才说错了", is_final=False)

    runtime.decide_from_transcript.assert_called_once_with(
        "我刚才说错了",
        0.82,
        vad_active=True,
        agent_speaking=True,
        is_final=False,
    )
    calls.apply_decision.assert_called_once_with(
        decision,
        eot_score=0.82,
        transcript="我刚才说错了",
        vad_active=True,
    )
    calls.cancel_duck_and_interrupt.assert_not_called()


def test_fallback_eot_hard_score_uses_direct_interrupt_path() -> None:
    runtime = MagicMock()
    runtime.decide_from_transcript.return_value = Decision(
        action=Action.HOLD,
        reason="uncertain",
        intent=InterruptIntent.UNCERTAIN,
    )
    handler, calls = _handler(
        eot_model=_eot_model(should_interrupt=True, score=0.8),
        turn_runtime=runtime,
        duck_active=False,
    )

    handler.run("那你继续解释一下")

    calls.apply_decision.assert_not_called()
    calls.timeline.record_decision.assert_called_once()
    calls.interrupt_current_turn.assert_called_once()
    calls.enter_soft_interrupt.assert_not_called()
