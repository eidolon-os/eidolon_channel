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
        clear_completed_voiceprint_turn=MagicMock(),
        remember_completed_voiceprint_turn=MagicMock(),
        reset_candidate_voiceprint_tasks=MagicMock(),
        remember_candidate_voiceprint_task=MagicMock(),
        attention_admission_reject_reason=MagicMock(return_value=""),
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
    owner._suppress_commit_after_interrupt_until = 123.0
    owner._suppress_transcripts_until_next_speech = True
    owner._completed_turn_voiceprint_task = object()
    owner._completed_turn_voiceprint_result = object()
    owner._completed_turn_voiceprint_timeline = TurnTimeline("old-turn")
    owner._callbacks = MagicMock()
    owner._user_speaking_start_time = None
    owner._timeline = None
    owner._timeline_debug_flushed = False
    owner._room = SimpleNamespace(name="room-a")
    owner._append_turn_timeline_snapshot = MagicMock()
    owner._apply_pending_explicit_client_preempt = MagicMock()
    owner._apply_pending_client_control_events = MagicMock()
    owner._voiceprint_turns = MagicMock()
    owner._ensure_provider_event_observer = MagicMock()
    owner._provider_events = MagicMock()
    owner._latest_asr_text = "stale"
    owner._get_eot_model = MagicMock(return_value=MagicMock())
    owner._uses_livekit_native_adaptive_interruption = MagicMock(return_value=False)
    owner._attention_effects = MagicMock()
    owner._attention_effects.handle_speaking_started.return_value = True
    owner._attach_transcript_ingress_recent_events = MagicMock()
    owner._ducking = SimpleNamespace(is_suspended=True)
    owner._interruption_orchestrator = MagicMock()
    owner._record_full_duplex_transition = MagicMock()

    def set_interrupt_cancel_suppression(active: bool, until: float) -> None:
        owner._skip_commit_after_interrupt_cancel = active
        owner._suppress_commit_after_interrupt_until = until

    owner._set_interrupt_cancel_suppression = MagicMock(
        side_effect=set_interrupt_cancel_suppression
    )
    return owner


def test_speech_lifecycle_start_opens_clean_full_duplex_segment() -> None:
    owner = _owner()

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._callbacks.on_user_started_speaking.assert_called_once_with()
    assert owner._skip_commit_after_interrupt_cancel is False
    assert owner._suppress_commit_after_interrupt_until == 0.0
    assert owner._suppress_transcripts_until_next_speech is False
    owner._turn_completion.clear_completed_voiceprint_turn.assert_called_once_with()
    assert owner._latest_asr_text == ""
    assert owner._timeline is not None
    assert owner._timeline.attrs["room_name"] == "room-a"
    owner._user_turns.start_speech.assert_called_once_with(timeline=owner._timeline)
    owner._append_turn_timeline_snapshot.assert_not_called()
    owner._attach_transcript_ingress_recent_events.assert_called_once_with(
        "speech_started"
    )
    owner._voiceprint_turns.start_turn.assert_called_once_with(timeline=owner._timeline)
    owner._ensure_provider_event_observer.assert_called_once_with()
    owner._provider_events.apply_pending_stt_provider_events.assert_called_once_with()
    owner._provider_events.observe_stt_turn_audio.assert_called_once_with()
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(True)
    owner._set_interrupt_cancel_suppression.assert_called_once_with(False, 0.0)
    owner._attention_effects.handle_speaking_started.assert_called_once_with()
    owner._interruption_orchestrator.start_candidate.assert_called_once_with(
        timeline=owner._timeline,
    )
    owner._record_full_duplex_transition.assert_called_once()
    assert owner._record_full_duplex_transition.call_args.args[0].value == (
        "user_speech_open"
    )
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "speech_started"
    )


def test_speech_lifecycle_start_does_not_open_candidate_without_interrupt_window() -> None:
    owner = _owner()
    owner._attention_effects.handle_speaking_started.return_value = False

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._attention_effects.handle_speaking_started.assert_called_once_with()
    owner._interruption_orchestrator.start_candidate.assert_not_called()


def test_speech_lifecycle_snapshots_replaced_unmerged_timeline() -> None:
    owner = _owner()
    previous = TurnTimeline("previous-turn")
    owner._timeline = previous

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._append_turn_timeline_snapshot.assert_called_once_with(
        previous,
        "speech_started_replaced_unmerged_timeline",
    )
    assert owner._timeline is not previous


def test_speech_lifecycle_keeps_merge_continuation_timeline() -> None:
    owner = _owner()
    previous = TurnTimeline("previous-turn")
    owner._timeline = previous
    owner._user_turns.can_merge_new_speech.return_value = True

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._append_turn_timeline_snapshot.assert_not_called()
    assert owner._timeline is previous


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


def test_speech_lifecycle_stop_commits_confirmed_cancel_candidate() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-confirmed-cancel")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._interruption_orchestrator.finish_confirmed_cancel_speech.return_value = True
    owner._interruption_orchestrator.resolve = MagicMock()
    owner._turn_completion.commit_post_speech_interruption_candidate = MagicMock(
        return_value=True
    )
    owner._user_turns.selected_text = "不是，我刚才说错了"
    owner._session = MagicMock()

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._interruption_orchestrator.finish_confirmed_cancel_speech.assert_called_once_with(
        "不是，我刚才说错了"
    )
    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task,
    )
    owner._turn_completion.commit_post_speech_interruption_candidate.assert_called_once_with(
        "confirmed_cancel_speech_end",
        transcript_override="不是，我刚才说错了",
    )
    owner._interruption_orchestrator.resolve.assert_called_once_with(
        action="cancel",
        reason="confirmed_cancel_turn_committed",
    )
    owner._set_interrupt_cancel_suppression.assert_called_once_with(False, 0.0)
    owner._callbacks.on_user_ended_speaking.assert_called_once_with()
    owner._user_turns.finish_speech.assert_not_called()
    assert owner._skip_commit_after_interrupt_cancel is False
    assert owner._latest_asr_text == ""


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
    assert owner._record_full_duplex_transition.call_args.args[0].value == (
        "user_turn_pending"
    )
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "user_turn_voiceprint_pending"
    )
    assert owner._latest_asr_text == ""


def test_speech_lifecycle_stop_rejects_attention_ignored_transcript() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-muted")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    owner._ducking = SimpleNamespace(is_suspended=False)
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._session = MagicMock()
    owner._user_turns.selected_text = "停一下"
    owner._user_turns.active = object()
    owner._turn_completion.attention_admission_reject_reason.return_value = (
        "attention_ignored:client_mic_muted"
    )

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._user_turns.reject_active.assert_called_once_with(
        "attention_ignored:client_mic_muted"
    )
    owner._turn_completion.clear_session_user_turn.assert_called_once_with(
        "attention_ignored:client_mic_muted"
    )
    owner._turn_completion.remember_candidate_voiceprint_task.assert_not_called()
    owner._turn_completion.schedule_voiceprint_gated_commit.assert_not_called()
    owner._turn_completion.schedule_deferred_low_eot_commit.assert_not_called()
    owner._get_eot_model.return_value.reset.assert_called_once_with()
    assert owner._record_full_duplex_transition.call_args.args[0].value == (
        "user_turn_rejected"
    )
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "attention_admission_rejected"
    )
    assert owner._latest_asr_text == ""
