"""DecisionEffectApplier boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.session import DecisionEffectApplier
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import TurnPolicyConfig


def _applier(timeline=None):
    setter = MagicMock()
    factory = SimpleNamespace(llm=SimpleNamespace(llm=SimpleNamespace(
        set_turn_control_metadata=setter,
    )))
    on_cancel = MagicMock()
    on_rollback = MagicMock()
    applier = DecisionEffectApplier(
        factory=factory,
        turn_runtime=TurnPolicyRuntime(TurnPolicyConfig()),
        get_timeline=lambda: timeline,
        on_cancel=on_cancel,
        on_rollback=on_rollback,
    )
    return applier, setter, on_cancel, on_rollback


def test_apply_cancel_publishes_turn_control_and_calls_cancel() -> None:
    timeline = MagicMock()
    applier, setter, on_cancel, on_rollback = _applier(timeline)
    decision = Decision(
        action=Action.CANCEL,
        reason="intent:hard_stop",
        intent=InterruptIntent.HARD_STOP,
        intent_source="lexicon",
        intent_confidence=1.0,
        tier="tier0_hard_stop",
        tier_reason="intent:hard_stop",
    )

    applier.apply(decision, transcript="别说了", vad_active=True, eot_score=0.9)

    on_cancel.assert_called_once_with()
    on_rollback.assert_not_called()
    setter.assert_called_once()
    metadata = setter.call_args.args[0]
    assert metadata["intent"] == "hard_stop"
    assert metadata["reason"] == "intent:hard_stop"
    timeline.set_attr.assert_called_once_with("turn_control", metadata)
    timeline.record_decision.assert_called_once()


def test_apply_rollback_calls_rollback_with_resolved_reason() -> None:
    applier, setter, on_cancel, on_rollback = _applier()
    decision = Decision(
        action=Action.ROLLBACK,
        reason="eot_score_low",
        rollback_drop_buffered=True,
        intent=InterruptIntent.BACKCHANNEL,
        intent_source="lexicon",
        intent_confidence=0.8,
    )

    applier.apply(decision, resolved_reason="timeout")

    on_cancel.assert_not_called()
    on_rollback.assert_called_once_with("timeout", True)
    setter.assert_called_once()


def test_record_decision_attrs_noops_without_timeline() -> None:
    applier, setter, on_cancel, on_rollback = _applier(timeline=None)

    applier.record_decision_attrs(Decision(action=Action.HOLD, reason="wait"))

    setter.assert_not_called()
    on_cancel.assert_not_called()
    on_rollback.assert_not_called()


def test_apply_hold_calls_hold_callback() -> None:
    on_hold = MagicMock()
    factory = SimpleNamespace(llm=SimpleNamespace(llm=SimpleNamespace()))
    applier = DecisionEffectApplier(
        factory=factory,
        turn_runtime=TurnPolicyRuntime(TurnPolicyConfig()),
        get_timeline=lambda: None,
        on_cancel=MagicMock(),
        on_rollback=MagicMock(),
        on_hold=on_hold,
    )
    decision = Decision(action=Action.HOLD, reason="stable_signal_wait")

    applier.apply(decision, transcript="不是", eot_score=0.1, vad_active=True)

    on_hold.assert_called_once_with(decision, "不是", 0.1, True)
