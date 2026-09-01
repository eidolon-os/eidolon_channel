"""Focused tests for the semantic interrupt session helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.semantic_interrupt import SemanticInterruptHandler
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent


def _eot_model(
    *,
    should_interrupt: bool = False,
    score: float = 0.0,
) -> MagicMock:
    model = MagicMock()
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
        interrupt_current_turn=calls.interrupt_current_turn,
        enter_soft_interrupt=calls.enter_soft_interrupt,
    )
    return handler, calls


def test_hard_stop_hint_without_evidence_has_no_direct_side_effect() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.HOLD,
        reason="semantic_score_wait",
        intent=InterruptIntent.HARD_STOP,
    )
    runtime.decide_from_transcript.return_value = decision
    handler, calls = _handler(
        eot_model=_eot_model(),
        turn_runtime=runtime,
        duck_active=False,
    )

    handler.run("别说了")

    runtime.decide_from_transcript.assert_called_once()
    calls.apply_decision.assert_called_once_with(
        decision,
        eot_score=0.0,
        transcript="别说了",
        vad_active=True,
    )
    calls.interrupt_current_turn.assert_not_called()


def test_strong_interrupt_with_duck_uses_decision_effect_path() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.CANCEL,
        reason="intent:hard_stop",
        intent=InterruptIntent.HARD_STOP,
    )
    runtime.decide_from_transcript.return_value = decision
    handler, calls = _handler(
        eot_model=_eot_model(score=0.9),
        turn_runtime=runtime,
        duck_active=True,
    )

    handler.run("别说了")

    calls.apply_decision.assert_called_once_with(
        decision,
        eot_score=0.9,
        transcript="别说了",
        vad_active=True,
    )
    calls.interrupt_current_turn.assert_not_called()


def test_model_intent_keeps_correction_hint() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.CANCEL,
        reason="intent:correction",
        intent=InterruptIntent.CORRECTION,
        correction_hint=True,
    )
    runtime.decide_from_transcript.return_value = decision
    handler, calls = _handler(
        eot_model=_eot_model(score=0.2),
        turn_runtime=runtime,
        duck_active=True,
    )

    handler.run("我刚才说错了")

    runtime.decide_from_transcript.assert_called_once_with(
        "我刚才说错了",
        0.2,
        vad_active=True,
        agent_speaking=True,
        is_final=False,
    )
    calls.apply_decision.assert_called_once_with(
        decision,
        eot_score=0.2,
        transcript="我刚才说错了",
        vad_active=True,
    )
    calls.interrupt_current_turn.assert_not_called()


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


def test_fallback_semantic_runs_for_normalized_redirect_hint() -> None:
    runtime = MagicMock()
    decision = Decision(
        action=Action.CANCEL,
        reason="intent:correction",
        intent=InterruptIntent.NORMAL_INTERRUPT,
        correction_hint=True,
    )
    runtime.decide_from_transcript.return_value = decision
    handler, calls = _handler(
        eot_model=_eot_model(score=0.1),
        turn_runtime=runtime,
        duck_active=False,
    )

    handler.run("不是，我刚才说错了", is_final=True)

    calls.apply_decision.assert_called_once_with(
        decision,
        eot_score=0.1,
        transcript="不是，我刚才说错了",
        vad_active=True,
    )
    calls.interrupt_current_turn.assert_not_called()


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
