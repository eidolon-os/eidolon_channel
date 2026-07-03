"""Stable-signal recheck timer tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.turn_policy import Action, Decision
from eidolon.livekit.common.config import TurnPolicyConfig


class _SuspendedDucking:
    is_suspended = True


@pytest.mark.asyncio
async def test_stable_signal_hold_rechecks_latest_transcript() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._ducking = _SuspendedDucking()
    pipeline._latest_asr_text = "不是"
    pipeline._stable_signal_timer = None
    pipeline._semantic_interrupts = SimpleNamespace(run=MagicMock())
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=correction age_ms=0 window_ms=120",
    )

    pipeline._handle_hold_decision(
        decision,
        "不",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.14)

    pipeline._semantic_interrupts.run.assert_called_once_with("不是", is_final=False)
    assert pipeline._stable_signal_timer is None


@pytest.mark.asyncio
async def test_stable_signal_timer_is_cancelled_before_recheck() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._ducking = _SuspendedDucking()
    pipeline._latest_asr_text = "不是"
    pipeline._stable_signal_timer = None
    pipeline._semantic_interrupts = SimpleNamespace(run=MagicMock())
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=correction age_ms=0 window_ms=120",
    )

    pipeline._handle_hold_decision(decision, "不是", eot_score=0.0, vad_active=True)
    pipeline._cancel_stable_signal_timer()
    await asyncio.sleep(0.14)

    pipeline._semantic_interrupts.run.assert_not_called()
    assert pipeline._stable_signal_timer is None


@pytest.mark.asyncio
async def test_stable_signal_hold_uses_remaining_recheck_ms() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._ducking = _SuspendedDucking()
    pipeline._latest_asr_text = "换个话"
    pipeline._stable_signal_timer = None
    pipeline._semantic_interrupts = SimpleNamespace(run=MagicMock())
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=topic_switch age_ms=80 window_ms=120",
        hold_recheck_ms=40,
    )

    pipeline._handle_hold_decision(
        decision,
        "换个话",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.06)

    pipeline._semantic_interrupts.run.assert_called_once_with("换个话", is_final=False)
    assert pipeline._stable_signal_timer is None


@pytest.mark.asyncio
async def test_normal_interrupt_hold_uses_policy_recheck_ms() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = TurnPolicyConfig()
    pipeline._ducking = _SuspendedDucking()
    pipeline._latest_asr_text = "那它的主要风险是什么"
    pipeline._stable_signal_timer = None
    pipeline._semantic_interrupts = SimpleNamespace(run=MagicMock())
    decision = Decision(
        action=Action.HOLD,
        reason="semantic_score_wait score=0.00 evidence=interim_substantive",
        hold_recheck_ms=30,
    )

    pipeline._handle_hold_decision(
        decision,
        "那它的主要风险是什么",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.05)

    pipeline._semantic_interrupts.run.assert_called_once_with(
        "那它的主要风险是什么",
        is_final=False,
    )
    assert pipeline._stable_signal_timer is None
