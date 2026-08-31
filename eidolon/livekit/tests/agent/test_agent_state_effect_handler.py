"""AgentStateEffectHandler boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.agent_state import (
    AgentStateEffectHandler,
    AgentStateTransition,
)
from eidolon.livekit.agent.session.agent_output_coordinator import AgentOutputCoordinator


class _Ducking:
    def __init__(self) -> None:
        self.on_agent_started_speaking = MagicMock()
        self.reset_if_cancelled = MagicMock(return_value=False)


def _handler(
    *,
    timeline=None,
    soft_interrupt_active=False,
    ducking=None,
    filler=None,
    should_flush_on_playback_done=None,
    on_playback_started=None,
    on_playback_finished=None,
):
    mark_activity = MagicMock()
    cancel_soft_interrupt = MagicMock()
    flush = MagicMock()
    ducking = ducking or _Ducking()
    handler = AgentStateEffectHandler(
        get_timeline=lambda: timeline,
        mark_activity=mark_activity,
        cancel_soft_interrupt=cancel_soft_interrupt,
        soft_interrupt_active=lambda: soft_interrupt_active,
        ducking=ducking,
        get_filler=lambda: filler,
        flush_timeline_debug=flush,
        should_flush_on_playback_done=should_flush_on_playback_done,
        on_playback_started=on_playback_started,
        on_playback_finished=on_playback_finished,
    )
    return handler, mark_activity, cancel_soft_interrupt, flush, ducking


def test_thinking_marks_activity_cancels_filler_and_resets_ducking() -> None:
    filler = SimpleNamespace(cancel=MagicMock())
    ducking = _Ducking()
    ducking.reset_if_cancelled.return_value = True
    handler, mark_activity, cancel_soft_interrupt, _, _ = _handler(
        soft_interrupt_active=True,
        ducking=ducking,
        filler=filler,
    )

    handler.handle(SimpleNamespace(old_state="listening", new_state="thinking"))

    mark_activity.assert_called_once_with()
    filler.cancel.assert_called_once_with()
    cancel_soft_interrupt.assert_called_once_with()
    ducking.reset_if_cancelled.assert_called_once_with()


def test_agent_state_transition_normalizes_livekit_event_shape() -> None:
    transition = AgentStateTransition.from_event(
        SimpleNamespace(old_state="thinking", new_state="speaking")
    )

    assert transition.old_state == "thinking"
    assert transition.new_state == "speaking"
    assert transition.starts_output_activity is True
    assert transition.starts_playback is True
    assert transition.starts_generation is False
    assert transition.completes_playback is False


def test_agent_state_transition_handles_playback_done() -> None:
    transition = AgentStateTransition.from_event(
        SimpleNamespace(old_state="speaking", new_state="listening")
    )

    assert transition.starts_output_activity is False
    assert transition.starts_playback is False
    assert transition.completes_playback is True


def test_agent_state_transition_handles_missing_fields() -> None:
    transition = AgentStateTransition.from_event(SimpleNamespace())

    assert transition.old_state == ""
    assert transition.new_state == ""
    assert transition.starts_output_activity is False
    assert transition.completes_playback is False


def test_agent_state_transition_from_event_is_idempotent() -> None:
    transition = AgentStateTransition(old_state="idle", new_state="thinking")

    assert AgentStateTransition.from_event(transition) is transition


def test_speaking_resets_played_counter() -> None:
    handler, mark_activity, _, _, ducking = _handler()

    handler.handle(SimpleNamespace(old_state="thinking", new_state="speaking"))

    mark_activity.assert_called_once_with()
    ducking.on_agent_started_speaking.assert_called_once_with()


def test_playback_done_marks_and_flushes_timeline() -> None:
    timeline = TurnTimeline("turn-agent-state")
    handler, _, _, flush, _ = _handler(timeline=timeline)

    handler.handle(SimpleNamespace(old_state="speaking", new_state="listening"))

    assert "agent_audio_playback_done_at" in timeline.timestamps
    flush.assert_called_once_with("agent_audio_playback_done", True)


def test_playback_lifecycle_callbacks_follow_agent_state() -> None:
    started = MagicMock()
    finished = MagicMock()
    handler, *_ = _handler(
        on_playback_started=started,
        on_playback_finished=finished,
    )

    handler.handle(SimpleNamespace(old_state="thinking", new_state="speaking"))
    handler.handle(SimpleNamespace(old_state="speaking", new_state="listening"))

    started.assert_called_once_with()
    finished.assert_called_once_with()


def test_playback_done_does_not_flush_non_terminal_user_turn() -> None:
    timeline = TurnTimeline("turn-agent-state")
    handler, _, _, flush, _ = _handler(
        timeline=timeline,
        should_flush_on_playback_done=lambda: False,
    )

    handler.handle(SimpleNamespace(old_state="speaking", new_state="listening"))

    assert "agent_audio_playback_done_at" in timeline.timestamps
    flush.assert_not_called()


def test_thinking_and_speaking_mark_timeline() -> None:
    timeline = TurnTimeline("turn-agent-state")
    handler, *_ = _handler(timeline=timeline)

    handler.handle(SimpleNamespace(old_state="listening", new_state="thinking"))
    handler.handle(SimpleNamespace(old_state="thinking", new_state="speaking"))

    assert "llm_started_at" in timeline.timestamps
    assert "tts_first_audio_at" in timeline.timestamps


def test_agent_output_coordinator_keeps_response_identity_across_new_candidate() -> None:
    coordinator = AgentOutputCoordinator()
    response = TurnTimeline("response-turn")
    candidate = TurnTimeline("interrupt-candidate")

    assert coordinator.claim(response) is None
    assert coordinator.active_timeline is response
    # Merely opening a new speech candidate does not transfer output ownership.
    assert candidate is not coordinator.active_timeline
    assert coordinator.release(candidate) is False
    assert coordinator.active_timeline is response
    assert coordinator.release(response) is True
    assert coordinator.active_timeline is None
