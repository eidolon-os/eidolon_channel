from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.transcript_event import FullDuplexTranscriptEvent
from eidolon.livekit.agent.full_duplex.transcript_recorder import (
    FullDuplexTranscriptRecorder,
)
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator


def test_recorder_returns_provider_neutral_canonical_turn_to_eot() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("segmented-provider-final")
    coordinator.start_speech(timeline=timeline, now=0.0)
    eot = MagicMock()
    pipeline = SimpleNamespace(
        _mark_activity=MagicMock(),
        _latest_asr_text="",
        _interruption_orchestrator=None,
        _barge_in_enabled=True,
        _uses_livekit_native_adaptive_interruption=lambda: False,
        _ensure_user_turn_coordinator=lambda: None,
        _ensure_transcript_evidence_buffer=lambda: SimpleNamespace(
            take=MagicMock(return_value=None)
        ),
        _user_turns=coordinator,
        _timeline=timeline,
        _get_eot_model=lambda: eot,
    )
    recorder = FullDuplexTranscriptRecorder(
        pipeline,
        transcript_revision_min_normalized_chars=4,
    )

    first = recorder.record(
        FullDuplexTranscriptEvent(transcript="换个。", is_final=True)
    )
    second = recorder.record(
        FullDuplexTranscriptEvent(transcript="话题", is_final=False)
    )

    assert first == "换个。"
    assert second == "换个话题"
    assert coordinator.selected_text == "换个。话题"
    assert coordinator.selected_policy_text == "换个话题"
    assert eot.update_asr.call_args_list == [
        (("换个。",), {"is_final": True}),
        (("换个话题",), {"is_final": False}),
    ]


def test_recorder_does_not_reuse_terminal_candidate_policy_text() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("terminal-candidate")
    coordinator.start_speech(timeline=timeline, now=0.0)
    coordinator.add_transcript("前一轮完整内容。", is_final=True, now=0.1)
    coordinator.mark_framework_completed(
        transcript="前一轮完整内容。",
        reason="framework_completed_turn",
        timeline=timeline,
        now=0.2,
    )
    eot = MagicMock()
    pipeline = SimpleNamespace(
        _mark_activity=MagicMock(),
        _latest_asr_text="",
        _interruption_orchestrator=None,
        _barge_in_enabled=True,
        _uses_livekit_native_adaptive_interruption=lambda: False,
        _ensure_user_turn_coordinator=lambda: None,
        _ensure_transcript_evidence_buffer=lambda: SimpleNamespace(
            take=MagicMock(return_value=None)
        ),
        _user_turns=coordinator,
        _timeline=timeline,
        _get_eot_model=lambda: eot,
    )

    policy_text = FullDuplexTranscriptRecorder(
        pipeline,
        transcript_revision_min_normalized_chars=4,
    ).record(FullDuplexTranscriptEvent(transcript="好", is_final=False))

    assert policy_text == "好"
    eot.update_asr.assert_called_once_with("好", is_final=False)
