from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.turn_policy import TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def _policy(owner: str) -> TurnPolicyConfig:
    return replace(TurnPolicyConfig(), interruption_owner=owner)


def _speech_start_pipeline(owner: str) -> StreamingPipeline:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = _policy(owner)
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    pipeline._allow_interruptions = True
    pipeline._ensure_runtime_defaults = MagicMock()
    pipeline._publish_companion_ui_state = MagicMock()
    pipeline._ensure_user_turn_coordinator = MagicMock()
    pipeline._cancel_deferred_low_eot_commit = MagicMock()
    pipeline._cancel_pending_voiceprint_commits = MagicMock()
    pipeline._reset_candidate_voiceprint_tasks = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._room = None
    pipeline._timeline = None
    pipeline._timeline_debug_flushed = False
    pipeline._latest_asr_text = "stale"
    pipeline._user_speaking_start_time = None
    pipeline._skip_commit_after_interrupt_cancel = False
    pipeline._suppress_transcripts_until_next_speech = False
    pipeline._completed_turn_voiceprint_task = None
    pipeline._completed_turn_voiceprint_result = None
    pipeline._completed_turn_voiceprint_timeline = None
    pipeline._session_signals = MagicMock()
    pipeline._voiceprint_turns = MagicMock()
    pipeline._apply_pending_stt_provider_events = MagicMock()
    pipeline._observe_stt_turn_audio = MagicMock()
    pipeline._user_turns = MagicMock()
    pipeline._user_turns.can_merge_new_speech.return_value = False
    pipeline._ducking = SimpleNamespace(is_suspended=True)
    pipeline._attention_effects = MagicMock()
    pipeline._interruption_orchestrator = MagicMock()
    eot_model = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot_model)
    return pipeline


def _transcript_pipeline(owner: str) -> StreamingPipeline:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = _policy(owner)
    pipeline._turn_runtime = TurnPolicyRuntime(pipeline._turn_policy)
    pipeline._allow_interruptions = True
    pipeline._ensure_runtime_defaults = MagicMock()
    pipeline._callbacks = MagicMock()
    pipeline._suppress_transcripts_until_next_speech = False
    pipeline._latest_asr_text = ""
    pipeline._timeline = TurnTimeline("test-turn")
    pipeline._mark_activity = MagicMock()
    pipeline._agent_output_active_for_interrupts = MagicMock(return_value=True)
    pipeline._transcript_echo_gate = MagicMock()
    pipeline._transcript_echo_gate.is_echo.return_value = False
    pipeline._interrupt_window_active = MagicMock(return_value=False)
    pipeline._interrupt_decision_suppressed = MagicMock(return_value=False)
    pipeline._ensure_user_turn_coordinator = MagicMock()
    pipeline._user_turns = MagicMock()
    pipeline._interruption_orchestrator = MagicMock()
    pipeline._attention_effects = MagicMock()
    pipeline._attention_effects.allows_eot_check.return_value = True
    pipeline._semantic_interrupts = MagicMock()
    eot_model = MagicMock()
    pipeline._get_eot_model = MagicMock(return_value=eot_model)
    return pipeline


def _state_event(old: str, new: str) -> SimpleNamespace:
    return SimpleNamespace(old_state=old, new_state=new)


def _transcript_event(text: str, *, final: bool = False) -> SimpleNamespace:
    return SimpleNamespace(transcript=text, is_final=final, speaker_id="device")


def test_livekit_native_profile_sets_adaptive_turn_handling() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = _policy("livekit_native_adaptive")
    pipeline._allow_interruptions = True
    pipeline._false_interruption_timeout = 6.0

    interruption = pipeline._build_turn_handling()["interruption"]

    assert interruption["enabled"] is True
    assert interruption["mode"] == "adaptive"
    assert interruption["resume_false_interruption"] is True
    assert interruption["false_interruption_timeout"] == 6.0


def test_native_profile_not_enabled_when_interruptions_disabled() -> None:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = _policy("livekit_native_adaptive")
    pipeline._allow_interruptions = False
    pipeline._false_interruption_timeout = 6.0

    interruption = pipeline._build_turn_handling()["interruption"]

    assert "mode" not in interruption
    assert interruption["enabled"] is False


def test_livekit_native_profile_does_not_start_channel_duck_candidate() -> None:
    pipeline = _speech_start_pipeline("livekit_native_adaptive")

    pipeline._on_user_state_changed(_state_event("listening", "speaking"))

    pipeline._attention_effects.handle_speaking_started.assert_not_called()
    pipeline._interruption_orchestrator.start_candidate.assert_not_called()
    assert pipeline._timeline is not None
    assert pipeline._timeline.attrs["interruption_owner"] == "livekit_native_adaptive"


def test_channel_owner_still_soft_ducks_and_starts_candidate() -> None:
    pipeline = _speech_start_pipeline("channel")

    pipeline._on_user_state_changed(_state_event("listening", "speaking"))

    pipeline._attention_effects.handle_speaking_started.assert_called_once()
    pipeline._interruption_orchestrator.start_candidate.assert_called_once()


def test_livekit_native_profile_skips_channel_semantic_interrupt_side_effects() -> None:
    pipeline = _transcript_pipeline("livekit_native_adaptive")

    pipeline._on_user_transcribed(_transcript_event("停一下"))

    pipeline._interruption_orchestrator.note_transcript.assert_not_called()
    pipeline._semantic_interrupts.run.assert_not_called()
    pipeline._user_turns.add_transcript.assert_called_once_with("停一下", is_final=False)


def test_channel_owner_keeps_channel_semantic_interrupt_side_effects() -> None:
    pipeline = _transcript_pipeline("channel")

    pipeline._on_user_transcribed(_transcript_event("停一下"))

    pipeline._interruption_orchestrator.note_transcript.assert_called_once_with(
        "停一下",
        is_final=False,
    )
    pipeline._semantic_interrupts.run.assert_called_once_with("停一下", is_final=False)
