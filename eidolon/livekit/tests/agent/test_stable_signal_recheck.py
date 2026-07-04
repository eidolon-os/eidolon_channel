"""Stable-signal recheck timer tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.interruption_effects import (
    FullDuplexInterruptionEffects,
)
from eidolon.livekit.agent.turn_policy import Action, Decision
from eidolon.livekit.common.config import TurnPolicyConfig


class _SuspendedDucking:
    is_suspended = True
    is_cancelled = False


class _CancelledDucking:
    is_suspended = False
    is_cancelled = True


def _effects(
    *,
    latest_asr_text: str,
    semantic_interrupts: SimpleNamespace,
    ducking: object | None = None,
    playback_evidence_active: bool = False,
) -> FullDuplexInterruptionEffects:
    policy = TurnPolicyConfig()
    return FullDuplexInterruptionEffects(
        ducking=ducking or _SuspendedDucking(),
        callbacks=MagicMock(),
        get_session=lambda: None,
        allow_interruptions=lambda: True,
        get_eot_model=lambda: MagicMock(),
        get_timeline=lambda: None,
        get_latest_asr_text=lambda: latest_asr_text,
        get_state_label=lambda: "SPEAKING",
        get_interruption_orchestrator=MagicMock(),
        publish_playback_stop=MagicMock(),
        snapshot_interrupted_context=MagicMock(),
        commit_post_speech_interruption_candidate=MagicMock(return_value=False),
        reject_post_speech_interruption_candidate=MagicMock(),
        cancel_residual_commit_suppress_sec=lambda: 0.0,
        semantic_interrupt_run=lambda text: semantic_interrupts.run(
            text,
            is_final=False,
        ),
        correction_topic_stability_window_ms=(
            lambda: policy.interrupt.correction_topic_stability_window_ms
        ),
        set_interrupt_cancel_suppression=MagicMock(),
        soft_interrupt_timeout_sec=lambda: 0.5,
        playback_evidence_active=lambda: playback_evidence_active,
    )


@pytest.mark.asyncio
async def test_stable_signal_hold_rechecks_latest_transcript() -> None:
    semantic_interrupts = SimpleNamespace(run=MagicMock())
    effects = _effects(
        latest_asr_text="不是",
        semantic_interrupts=semantic_interrupts,
    )
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=correction age_ms=0 window_ms=120",
    )

    effects.handle_hold_decision(
        decision,
        "不",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.14)

    semantic_interrupts.run.assert_called_once_with("不是", is_final=False)
    assert effects._stable_signal_timer is None


@pytest.mark.asyncio
async def test_stable_signal_timer_is_cancelled_before_recheck() -> None:
    semantic_interrupts = SimpleNamespace(run=MagicMock())
    effects = _effects(
        latest_asr_text="不是",
        semantic_interrupts=semantic_interrupts,
    )
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=correction age_ms=0 window_ms=120",
    )

    effects.handle_hold_decision(decision, "不是", eot_score=0.0, vad_active=True)
    effects.cancel_stable_signal_timer()
    await asyncio.sleep(0.14)

    semantic_interrupts.run.assert_not_called()
    assert effects._stable_signal_timer is None


@pytest.mark.asyncio
async def test_stable_signal_hold_uses_remaining_recheck_ms() -> None:
    semantic_interrupts = SimpleNamespace(run=MagicMock())
    effects = _effects(
        latest_asr_text="换个话",
        semantic_interrupts=semantic_interrupts,
    )
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=topic_switch age_ms=80 window_ms=120",
        hold_recheck_ms=40,
    )

    effects.handle_hold_decision(
        decision,
        "换个话",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.06)

    semantic_interrupts.run.assert_called_once_with("换个话", is_final=False)
    assert effects._stable_signal_timer is None


@pytest.mark.asyncio
async def test_stable_signal_hold_rechecks_with_fresh_playback_evidence() -> None:
    semantic_interrupts = SimpleNamespace(run=MagicMock())
    effects = _effects(
        latest_asr_text="换个话题",
        semantic_interrupts=semantic_interrupts,
        ducking=_CancelledDucking(),
        playback_evidence_active=True,
    )
    decision = Decision(
        action=Action.HOLD,
        reason="stable_signal_wait intent=topic_switch age_ms=80 window_ms=120",
        hold_recheck_ms=40,
    )

    effects.handle_hold_decision(
        decision,
        "换个话",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.06)

    semantic_interrupts.run.assert_called_once_with("换个话题", is_final=False)
    assert effects._stable_signal_timer is None


@pytest.mark.asyncio
async def test_normal_interrupt_hold_uses_policy_recheck_ms() -> None:
    semantic_interrupts = SimpleNamespace(run=MagicMock())
    effects = _effects(
        latest_asr_text="那它的主要风险是什么",
        semantic_interrupts=semantic_interrupts,
    )
    decision = Decision(
        action=Action.HOLD,
        reason="semantic_score_wait score=0.00 evidence=interim_substantive",
        hold_recheck_ms=30,
    )

    effects.handle_hold_decision(
        decision,
        "那它的主要风险是什么",
        eot_score=0.0,
        vad_active=True,
    )

    await asyncio.sleep(0.05)

    semantic_interrupts.run.assert_called_once_with(
        "那它的主要风险是什么",
        is_final=False,
    )
    assert effects._stable_signal_timer is None
