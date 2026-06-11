"""AgentStateEffectHandler boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import AgentStateEffectHandler


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
