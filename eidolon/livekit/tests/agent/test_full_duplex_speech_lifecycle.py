from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.speech_lifecycle import (
    FullDuplexSpeechLifecycle,
)
from eidolon.livekit.agent.observability import TurnTimeline


def _owner() -> SimpleNamespace:
    owner = SimpleNamespace()
    turn_completion = SimpleNamespace(
        cancel_deferred_low_eot_commit=MagicMock(),
        cancel_pending_voiceprint_commits=MagicMock(),
        reset_candidate_voiceprint_tasks=MagicMock(),
        remember_candidate_voiceprint_task=MagicMock(),
        playback_low_evidence_reject_reason=MagicMock(return_value=""),
        should_defer_low_eot_commit=MagicMock(return_value=False),
        candidate_voiceprint_gate_task=MagicMock(return_value=None),
        schedule_deferred_low_eot_commit=MagicMock(),
        schedule_voiceprint_gated_commit=MagicMock(),
        clear_session_user_turn=MagicMock(),
    )
    owner._turn_completion = turn_completion
    owner._ensure_turn_completion = MagicMock(return_value=turn_completion)
    owner._ensure_user_turn_coordinator = MagicMock()
    owner._user_turns = MagicMock()
    owner._user_turns.can_merge_new_speech.return_value = False
    owner._skip_commit_after_interrupt_cancel = True
    owner._suppress_transcripts_until_next_speech = True
    owner._completed_turn_voiceprint_task = object()
    owner._completed_turn_voiceprint_result = object()
    owner._completed_turn_voiceprint_timeline = TurnTimeline("old-turn")
    owner._callbacks = MagicMock()
    owner._user_speaking_start_time = None
    owner._timeline = None
    owner._timeline_debug_flushed = False
    owner._room = SimpleNamespace(name="room-a")
    owner._apply_pending_explicit_client_preempt = MagicMock()
    owner._apply_pending_client_control_events = MagicMock()
    owner._voiceprint_turns = MagicMock()
    owner._ensure_provider_event_observer = MagicMock()
    owner._provider_events = MagicMock()
    owner._latest_asr_text = "stale"
    owner._get_eot_model = MagicMock(return_value=MagicMock())
    owner._uses_livekit_native_adaptive_interruption = MagicMock(return_value=False)
    owner._attention_effects = MagicMock()
    owner._ducking = SimpleNamespace(is_suspended=True)
    owner._interruption_orchestrator = MagicMock()
    return owner


def test_speech_lifecycle_start_opens_clean_full_duplex_segment() -> None:
    owner = _owner()

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._callbacks.on_user_started_speaking.assert_called_once_with()
    assert owner._skip_commit_after_interrupt_cancel is False
    assert owner._suppress_transcripts_until_next_speech is False
    assert owner._latest_asr_text == ""
    assert owner._timeline is not None
    assert owner._timeline.attrs["room_name"] == "room-a"
    owner._user_turns.start_speech.assert_called_once_with(timeline=owner._timeline)
    owner._voiceprint_turns.start_turn.assert_called_once_with(timeline=owner._timeline)
    owner._ensure_provider_event_observer.assert_called_once_with()
    owner._provider_events.apply_pending_stt_provider_events.assert_called_once_with()
    owner._provider_events.observe_stt_turn_audio.assert_called_once_with()
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(True)
    owner._attention_effects.handle_speaking_started.assert_called_once_with()
    owner._interruption_orchestrator.start_candidate.assert_called_once_with(
        timeline=owner._timeline,
    )


def test_speech_lifecycle_stop_defers_when_interruption_owner_waits_for_stt() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-wait")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._interruption_orchestrator.defer_false_resume_after_speech_end.return_value = True
    owner._session = MagicMock()

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._callbacks.on_user_ended_speaking.assert_called_once_with()
    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task,
    )
    owner._user_turns.finish_speech.assert_not_called()
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(False)


def test_speech_lifecycle_stop_schedules_voiceprint_gated_commit() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-commit")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    owner._ducking = SimpleNamespace(is_suspended=False)
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._session = MagicMock()
    owner._user_turns.selected_text = "你好"
    owner._user_turns.active = object()
    owner._user_turns.finish_speech.return_value = SimpleNamespace(
        action="commit",
        reason="complete",
        transcript="你好",
        delay_sec=0.0,
    )
    owner._turn_completion.candidate_voiceprint_gate_task.return_value = "verify-task"

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task,
    )
    owner._user_turns.finish_speech.assert_called_once()
    owner._turn_completion.schedule_voiceprint_gated_commit.assert_called_once_with(
        verify_task="verify-task",
        eot_model=owner._get_eot_model.return_value,
        transcript="你好",
        timeline=owner._timeline,
    )
    assert owner._latest_asr_text == ""
