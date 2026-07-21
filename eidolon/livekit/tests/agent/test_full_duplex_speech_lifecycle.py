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
        cancel_completed_voiceprint_turn=MagicMock(),
        remember_completed_voiceprint_turn=MagicMock(),
        reset_candidate_voiceprint_tasks=MagicMock(),
        remember_candidate_voiceprint_task=MagicMock(),
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
    owner._runtime_participant_identity = "device-a"
    owner._append_turn_timeline_snapshot = MagicMock()
    owner._apply_pending_explicit_client_preempt = MagicMock()
    owner._apply_pending_client_control_events = MagicMock()
    owner._voiceprint_turns = MagicMock()
    owner._ensure_provider_event_observer = MagicMock()
    owner._provider_events = MagicMock()
    owner._latest_asr_text = "stale"
    owner._get_eot_model = MagicMock(return_value=MagicMock())
    owner._uses_livekit_native_adaptive_interruption = MagicMock(return_value=False)
    # full_duplex: barge-in is ON (the single authority the lifecycle now consults
    # before arming/resolving an interruption candidate).
    owner._barge_in_enabled = True
    owner._attention_effects = MagicMock()
    owner._attention_effects.handle_speaking_started.return_value = True
    owner._attach_transcript_ingress_recent_events = MagicMock()
    owner._ducking = SimpleNamespace(is_suspended=True)
    owner._interruption_orchestrator = MagicMock()
    owner._record_full_duplex_transition = MagicMock()
    owner._agent_output = MagicMock()
    owner._ensure_agent_output_coordinator = MagicMock(return_value=owner._agent_output)

    def set_interrupt_cancel_suppression(
        active: bool, until: float | None = None, *, reason: str = ""
    ) -> None:
        owner._skip_commit_after_interrupt_cancel = active
        if until is not None:
            owner._suppress_commit_after_interrupt_until = until

    owner._set_interrupt_cancel_suppression = MagicMock(
        side_effect=set_interrupt_cancel_suppression
    )

    def set_suppress_transcripts_until_next_speech(value: bool, *, reason: str = "") -> None:
        owner._suppress_transcripts_until_next_speech = value

    owner._set_suppress_transcripts_until_next_speech = MagicMock(
        side_effect=set_suppress_transcripts_until_next_speech
    )
    return owner


def test_speech_lifecycle_start_opens_clean_full_duplex_segment() -> None:
    owner = _owner()

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._callbacks.on_user_started_speaking.assert_called_once_with()
    assert owner._skip_commit_after_interrupt_cancel is False
    assert owner._suppress_commit_after_interrupt_until == 0.0
    assert owner._suppress_transcripts_until_next_speech is False
    owner._turn_completion.cancel_completed_voiceprint_turn.assert_called_once_with()
    assert owner._latest_asr_text == ""
    assert owner._timeline is not None
    assert owner._timeline.attrs["room_name"] == "room-a"
    assert owner._timeline.attrs["participant_identity"] == "device-a"
    owner._user_turns.start_speech.assert_called_once_with(timeline=owner._timeline)
    owner._agent_output.link_interruption_candidate.assert_called_once_with(owner._timeline)
    owner._append_turn_timeline_snapshot.assert_not_called()
    owner._attach_transcript_ingress_recent_events.assert_called_once_with("speech_started")
    owner._voiceprint_turns.start_turn.assert_called_once_with(timeline=owner._timeline)
    owner._ensure_provider_event_observer.assert_called_once_with()
    owner._provider_events.apply_pending_stt_provider_events.assert_called_once_with()
    owner._provider_events.observe_stt_turn_audio.assert_called_once_with()
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(True)
    owner._set_interrupt_cancel_suppression.assert_called_once_with(
        False, 0.0, reason="new_speech_started"
    )
    owner._set_suppress_transcripts_until_next_speech.assert_called_once_with(
        False, reason="new_speech_started"
    )
    owner._attention_effects.handle_speaking_started.assert_called_once_with()
    owner._interruption_orchestrator.start_candidate.assert_called_once_with(
        timeline=owner._timeline,
    )
    owner._record_full_duplex_transition.assert_called_once()
    assert owner._record_full_duplex_transition.call_args.args[0].value == ("user_speech_open")
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == ("speech_started")


def test_speech_lifecycle_start_does_not_open_candidate_without_interrupt_window() -> None:
    owner = _owner()
    owner._attention_effects.handle_speaking_started.return_value = False

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._attention_effects.handle_speaking_started.assert_called_once_with()
    owner._interruption_orchestrator.start_candidate.assert_not_called()


def test_speech_lifecycle_start_skips_barge_in_when_half_duplex() -> None:
    # Regression: half_duplex has no barge-in, so VAD-start must NOT arm a duck
    # window or an interruption candidate. The pre-fix behaviour routed every
    # idle-turn utterance into the interruption evidence window, which rolled
    # back on timeout and never committed ("一直收音中, agent 从不回复").
    owner = _owner()
    owner._barge_in_enabled = False

    FullDuplexSpeechLifecycle(owner).handle_started()

    # No barge-in machinery is touched on VAD-start.
    owner._attention_effects.handle_speaking_started.assert_not_called()
    owner._interruption_orchestrator.start_candidate.assert_not_called()
    assert owner._timeline.attrs["interruption_owner"] == "disabled_no_barge_in"
    # Shared turn setup still runs — the user turn must proceed to the framework's
    # normal endpointing/commit path.
    owner._callbacks.on_user_started_speaking.assert_called_once_with()
    owner._user_turns.start_speech.assert_called_once_with(timeline=owner._timeline)
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(True)


def test_speech_lifecycle_stop_skips_interruption_resolve_when_half_duplex() -> None:
    # half_duplex stop-side symmetry: no confirmed-cancel / candidate resolution.
    owner = _owner()
    owner._barge_in_enabled = False

    lifecycle = FullDuplexSpeechLifecycle(owner)
    lifecycle.handle_started()
    lifecycle.handle_stopped()

    owner._interruption_orchestrator.finish_confirmed_cancel_speech.assert_not_called()
    owner._get_eot_model.return_value.update_vad.assert_called_with(False)


def test_speech_lifecycle_snapshots_replaced_unmerged_timeline() -> None:
    owner = _owner()
    previous = TurnTimeline("previous-turn")
    owner._timeline = previous

    FullDuplexSpeechLifecycle(owner).handle_started()

    owner._append_turn_timeline_snapshot.assert_called_once_with(
        previous,
        "speech_started_replaced_unmerged_timeline",
    )
    first_transition = owner._record_full_duplex_transition.call_args_list[0]
    assert first_transition.args[0].value == "user_turn_rejected"
    assert first_transition.kwargs["event"] == "user_turn_superseded_by_new_speech"
    assert first_transition.kwargs["timeline"] is previous
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
    owner._get_eot_model.return_value.update_vad.assert_called_once_with(False)


def test_speech_lifecycle_stop_leaves_confirmed_cancel_for_framework_completion() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-confirmed-cancel")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._interruption_orchestrator.finish_confirmed_cancel_speech.return_value = True
    owner._user_turns.selected_text = "不是，我刚才说错了"
    owner._session = MagicMock()

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._interruption_orchestrator.finish_confirmed_cancel_speech.assert_called_once_with(
        "不是，我刚才说错了"
    )
    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task,
    )
    owner._user_turns.note_speech_stopped.assert_called_once()
    owner._callbacks.on_user_ended_speaking.assert_called_once_with()
    assert owner._skip_commit_after_interrupt_cancel is False
    assert owner._latest_asr_text == "stale"


def test_speech_lifecycle_stop_waits_for_framework_completed_turn() -> None:
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
    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task,
    )
    owner._user_turns.note_speech_stopped.assert_called_once()
    assert owner._record_full_duplex_transition.call_args.args[0].value == ("user_turn_pending")
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "speech_stopped_waiting_framework"
    )
    assert owner._latest_asr_text == "stale"


def test_speech_lifecycle_stop_does_not_apply_terminal_reject_policy() -> None:
    owner = _owner()
    owner._timeline = TurnTimeline("turn-replacement")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    owner._ducking = SimpleNamespace(is_suspended=False)
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._session = MagicMock()
    owner._user_turns.selected_text = "那我再说了。"
    owner._user_turns.active = SimpleNamespace(timeline=owner._timeline)
    FullDuplexSpeechLifecycle(owner).handle_stopped()


def test_speech_lifecycle_stop_leaves_non_semantic_reject_to_completed_turn_gate() -> None:
    # VAD-stop records the cough but does not reject it. The completed-turn gate
    # owns the one terminal non-semantic decision.
    owner = _owner()
    owner._timeline = TurnTimeline("turn-cough")
    voiceprint_task = object()
    owner._voiceprint_turns.finish_turn.return_value = voiceprint_task
    owner._ducking = SimpleNamespace(is_suspended=False)
    effects = MagicMock()
    effects.soft_interrupt_active.return_value = False
    owner._ensure_interruption_effects = MagicMock(return_value=effects)
    owner._session = MagicMock()
    owner._user_turns.selected_text = "咳咳。"
    owner._user_turns.active = object()

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._user_turns.reject_active.assert_not_called()
    assert owner._record_full_duplex_transition.call_args.args[0].value == ("user_turn_pending")
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "speech_stopped_waiting_framework"
    )
    assert owner._latest_asr_text == "stale"


def test_speech_lifecycle_stop_leaves_attention_reject_to_completed_turn_gate() -> None:
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

    FullDuplexSpeechLifecycle(owner).handle_stopped()

    owner._user_turns.reject_active.assert_not_called()
    owner._turn_completion.remember_candidate_voiceprint_task.assert_called_once_with(
        voiceprint_task
    )
    assert owner._record_full_duplex_transition.call_args.args[0].value == ("user_turn_pending")
    assert owner._record_full_duplex_transition.call_args.kwargs["event"] == (
        "speech_stopped_waiting_framework"
    )
    assert owner._latest_asr_text == "stale"
