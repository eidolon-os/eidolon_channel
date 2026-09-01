"""DecisionEffectApplier boundary tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from eidolon.livekit.agent.session.decision_effects import DecisionEffectApplier
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import TurnPolicyConfig


def _applier(timeline=None):
    on_cancel = MagicMock()
    on_rollback = MagicMock()
    applier = DecisionEffectApplier(
        turn_runtime=TurnPolicyRuntime(TurnPolicyConfig()),
        get_timeline=lambda: timeline,
        on_cancel=on_cancel,
        on_rollback=on_rollback,
    )
    return applier, on_cancel, on_rollback


def test_apply_cancel_records_and_calls_cancel_without_cross_turn_metadata() -> None:
    timeline = MagicMock()
    applier, on_cancel, on_rollback = _applier(timeline)
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
    timeline.set_attr.assert_not_called()
    timeline.record_decision.assert_called_once()


def test_apply_rollback_calls_rollback_with_resolved_reason() -> None:
    applier, on_cancel, on_rollback = _applier()
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


def test_record_decision_attrs_noops_without_timeline() -> None:
    applier, on_cancel, on_rollback = _applier(timeline=None)

    applier.record_decision_attrs(Decision(action=Action.HOLD, reason="wait"))

    on_cancel.assert_not_called()
    on_rollback.assert_not_called()


def test_apply_hold_calls_hold_callback() -> None:
    on_hold = MagicMock()
    applier = DecisionEffectApplier(
        turn_runtime=TurnPolicyRuntime(TurnPolicyConfig()),
        get_timeline=lambda: None,
        on_cancel=MagicMock(),
        on_rollback=MagicMock(),
        on_hold=on_hold,
    )
    decision = Decision(action=Action.HOLD, reason="stable_signal_wait")

    applier.apply(decision, transcript="不是", eot_score=0.1, vad_active=True)

    on_hold.assert_called_once_with(decision, "不是", 0.1, True)


def test_apply_notifies_owner_before_side_effects() -> None:
    order: list[str] = []
    on_cancel = MagicMock(side_effect=lambda: order.append("cancel"))
    on_decision = MagicMock(side_effect=lambda *args, **kwargs: order.append("owner"))
    record_transition = MagicMock(side_effect=lambda *args, **kwargs: order.append("contract"))
    applier = DecisionEffectApplier(
        turn_runtime=TurnPolicyRuntime(TurnPolicyConfig()),
        get_timeline=lambda: None,
        on_cancel=on_cancel,
        on_rollback=MagicMock(),
        on_decision=on_decision,
        record_full_duplex_transition=record_transition,
    )
    decision = Decision(
        action=Action.CANCEL,
        reason="intent:hard_stop",
        intent=InterruptIntent.HARD_STOP,
    )

    applier.apply(decision, transcript="停一下", vad_active=True, eot_score=0.9)

    on_decision.assert_called_once_with(
        decision,
        source="turn_policy",
        transcript="停一下",
        vad_active=True,
        eot_score=0.9,
    )
    record_transition.assert_called_once_with(
        decision,
        source="turn_policy",
        transcript="停一下",
        vad_active=True,
        eot_score=0.9,
    )
    on_cancel.assert_called_once_with()
    assert order == ["owner", "contract", "cancel"]
