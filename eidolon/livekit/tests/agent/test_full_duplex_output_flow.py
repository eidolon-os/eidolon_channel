"""FullDuplexOutputFlow tests."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.output_flow import FullDuplexOutputFlow
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.shared.types import PipelineState
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator


def _duck_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        duck_fade_ms=30,
        duck_fade_in_ms=30,
        duck_suspend_volume=0.0,
        duck_buffer_max_sec=2.0,
        duck_early_cancel_score_threshold=0.7,
        duck_early_resume_score_threshold=0.2,
        duck_suspend_timeout_sec=0.5,
        duck_cooldown_sec=0.0,
    )


def test_output_flow_installs_duck_mixer_via_ducking_controller() -> None:
    cfg = _duck_cfg()
    ducking = SimpleNamespace(install=MagicMock(return_value=object()))
    pipeline = SimpleNamespace(
        _ducking=ducking,
        _user_turns=UserTurnCoordinator(speech_merge_grace_sec=.8),
        _get_eot_model=lambda: SimpleNamespace(_config=cfg),
    )
    session = object()

    FullDuplexOutputFlow(pipeline).install_duck_mixer(session)

    ducking.install.assert_called_once_with(session, cfg)


def test_output_flow_does_not_arm_when_ducking_not_installed() -> None:
    ducking = SimpleNamespace(
        installed=False,
        last_unduck_time=0.0,
        cancel_timeout=MagicMock(),
        duck=MagicMock(),
    )
    pipeline = SimpleNamespace(
        _ducking=ducking,
        _user_turns=UserTurnCoordinator(speech_merge_grace_sec=.8),
        _state=PipelineState.IDLE,
        _get_eot_model=lambda: SimpleNamespace(_config=_duck_cfg()),
        _filler=None,
    )

    armed = FullDuplexOutputFlow(pipeline).duck_and_arm_timeout()

    assert armed is False
    ducking.duck.assert_not_called()
    ducking.cancel_timeout.assert_not_called()


@pytest.mark.asyncio
async def test_output_flow_arms_after_attention_admits_playback_even_if_state_idle() -> None:
    async def _deadline_run(timeout_sec: float) -> None:
        deadline.timeout_sec = timeout_sec

    cfg = _duck_cfg()
    ducking = SimpleNamespace(
        installed=True,
        last_unduck_time=0.0,
        timeout_task=None,
        cancel_timeout=MagicMock(),
        duck=MagicMock(return_value=True),
    )
    deadline = SimpleNamespace(run=_deadline_run)
    timeline = TurnTimeline("turn-client-playback")
    pipeline = SimpleNamespace(
        _ducking=ducking,
        _user_turns=UserTurnCoordinator(speech_merge_grace_sec=.8),
        _state=PipelineState.IDLE,
        _get_eot_model=lambda: SimpleNamespace(_config=cfg),
        _filler=None,
        _timeline=timeline,
        _user_speaking_start_time=time.monotonic() - 0.1,
        _callbacks=SimpleNamespace(on_duck_started=MagicMock()),
        _duck_deadline=deadline,
        _record_full_duplex_transition=MagicMock(),
    )

    armed = FullDuplexOutputFlow(pipeline).duck_and_arm_timeout()
    await asyncio.sleep(0)

    assert armed is True
    ducking.cancel_timeout.assert_called_once_with()
    ducking.duck.assert_called_once()
    assert "interrupt_started_at" in timeline.timestamps


@pytest.mark.asyncio
async def test_output_flow_arms_duck_deadline_and_records_timeline() -> None:
    async def _deadline_run(timeout_sec: float) -> None:
        deadline.timeout_sec = timeout_sec

    cfg = _duck_cfg()
    ducking = SimpleNamespace(
        installed=True,
        last_unduck_time=0.0,
        timeout_task=None,
        cancel_timeout=MagicMock(),
        duck=MagicMock(return_value=True),
    )
    deadline = SimpleNamespace(run=_deadline_run)
    timeline = TurnTimeline("turn-duck")
    pipeline = SimpleNamespace(
        _ducking=ducking,
        _user_turns=UserTurnCoordinator(speech_merge_grace_sec=.8),
        _state=PipelineState.SPEAKING,
        _get_eot_model=lambda: SimpleNamespace(_config=cfg),
        _filler=None,
        _timeline=timeline,
        _user_speaking_start_time=time.monotonic() - 0.1,
        _callbacks=SimpleNamespace(on_duck_started=MagicMock()),
        _duck_deadline=deadline,
        _record_full_duplex_transition=MagicMock(),
    )

    armed = FullDuplexOutputFlow(pipeline).duck_and_arm_timeout()
    await asyncio.sleep(0)

    assert armed is True
    ducking.cancel_timeout.assert_called_once_with()
    ducking.duck.assert_called_once()
    pipeline._callbacks.on_duck_started.assert_called_once_with()
    assert ducking.timeout_task is not None
    assert deadline.timeout_sec == cfg.duck_suspend_timeout_sec
    assert "interrupt_started_at" in timeline.timestamps
    assert timeline.attrs["duck_last_event"]["event"] == "duck_started"


def test_output_flow_does_not_record_started_when_output_cannot_suspend() -> None:
    cfg = _duck_cfg()
    ducking = SimpleNamespace(
        installed=True,
        last_unduck_time=0.0,
        timeout_task=None,
        mixer=SimpleNamespace(state="CANCELLED"),
        cancel_timeout=MagicMock(),
        duck=MagicMock(return_value=False),
    )
    timeline = TurnTimeline("turn-duck-skipped")
    pipeline = SimpleNamespace(
        _ducking=ducking,
        _user_turns=UserTurnCoordinator(speech_merge_grace_sec=.8),
        _state=PipelineState.SPEAKING,
        _get_eot_model=lambda: SimpleNamespace(_config=cfg),
        _filler=None,
        _timeline=timeline,
        _user_speaking_start_time=time.monotonic() - 0.1,
        _callbacks=SimpleNamespace(on_duck_started=MagicMock()),
        _duck_deadline=SimpleNamespace(run=MagicMock()),
    )

    armed = FullDuplexOutputFlow(pipeline).duck_and_arm_timeout()

    assert armed is False
    pipeline._callbacks.on_duck_started.assert_not_called()
    assert "interrupt_started_at" not in timeline.timestamps
    assert timeline.attrs["duck_last_event"] == {
        "event": "duck_skipped",
        "reason": "output_not_suspendable",
        "output_state": "CANCELLED",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_state", ["open", "empty", "committed", "rejected"])
@pytest.mark.parametrize("speaking", [True, False])
async def test_pending_transcript_continuation_can_pause_inside_output_cooldown(candidate_state, speaking):
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=.8)
    timeline = TurnTimeline("continuation")
    coordinator.start_speech(timeline=timeline)
    if candidate_state != "empty":
        coordinator.add_transcript("让我想想。", is_final=True)
    if candidate_state == "committed":
        coordinator.mark_framework_completed(transcript=coordinator.selected_text,
                                             timeline=timeline, reason="test")
    elif candidate_state == "rejected":
        coordinator.reject_active("test")
    cfg = _duck_cfg()
    cfg.duck_cooldown_sec = .8
    ducking = SimpleNamespace(installed=True, last_unduck_time=time.monotonic(),
        cancel_timeout=MagicMock(), duck=MagicMock(return_value=True), timeout_task=None)
    pipeline = SimpleNamespace(_ducking=ducking, _user_turns=coordinator,
        _get_eot_model=lambda: SimpleNamespace(_config=cfg), _filler=None,
        _timeline=timeline, _user_speaking_start_time=time.monotonic() if speaking else None,
        _callbacks=SimpleNamespace(on_duck_started=MagicMock()),
        _duck_deadline=SimpleNamespace(run=AsyncMock()),
        _record_full_duplex_transition=MagicMock())
    armed = FullDuplexOutputFlow(pipeline).duck_and_arm_timeout()
    await asyncio.sleep(0)
    assert armed is (candidate_state == "open" and speaking)
    assert ducking.duck.call_count == int(armed)
    assert pipeline._duck_deadline.run.await_count == int(armed)
