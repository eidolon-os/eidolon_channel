"""InterruptionOrchestrator lifecycle tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import (
    InterruptionDecisionAction,
    InterruptionOrchestrator,
    InterruptionState,
)
from eidolon.livekit.agent.turn_policy import Action, Decision, InterruptIntent


def test_vad_end_without_transcript_waits_for_evidence_when_ducked() -> None:
    now = 10.0

    def clock() -> float:
        return now

    timeline = TurnTimeline("turn-1")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )

    owner.start_candidate(timeline=timeline)
    now = 10.55

    deferred = owner.defer_false_resume_after_speech_end(
        transcript="",
        duck_suspended=True,
    )

    assert deferred is True
    assert owner.should_hold_deadline() is True
    assert owner.max_suspend_sec() == 6.0
    assert owner.state is InterruptionState.SUSPENDED_POST_SPEECH_WAIT
    assert timeline.attrs["interruption_orchestrator_last_event"]["event"] == (
        "post_speech_evidence_wait"
    )


def test_short_vad_blip_keeps_fast_false_resume_path() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )

    owner.start_candidate(timeline=TurnTimeline("turn-1"))
    now = 10.10

    deferred = owner.defer_false_resume_after_speech_end(
        transcript="",
        duck_suspended=True,
    )

    assert deferred is False
    assert owner.should_hold_deadline() is False


def test_resolve_closes_evidence_window() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-1"))
    now = 10.55
    assert owner.defer_false_resume_after_speech_end(
        transcript="",
        duck_suspended=True,
    )

    owner.resolve(action="cancel", reason="hard_stop")

    assert owner.active is False
    assert owner.should_hold_deadline() is False


def test_turn_policy_hold_keeps_weak_transcript_in_evidence_window() -> None:
    now = 10.0

    def clock() -> float:
        return now

    timeline = TurnTimeline("turn-weak")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )

    start = owner.start_candidate(timeline=timeline)
    assert start.action is InterruptionDecisionAction.SOFT_SUSPEND_OUTPUT
    owner.note_transcript("不是", is_final=False)
    decision = owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="semantic_score_wait score=0.00 evidence=interim_substantive",
            intent=InterruptIntent.UNCERTAIN,
        ),
        source="turn_policy",
        transcript="不是",
        vad_active=True,
        eot_score=0.0,
    )

    assert decision.action is InterruptionDecisionAction.HOLD_FOR_EVIDENCE
    assert owner.state is InterruptionState.SUSPENDED_WAITING_EVIDENCE

    now = 10.55
    deferred = owner.defer_false_resume_after_speech_end(
        transcript="不是",
        duck_suspended=True,
    )

    assert deferred is True
    assert owner.awaiting_post_speech_evidence is True
    last = timeline.attrs["interruption_orchestrator_last_event"]
    assert last["event"] == "post_speech_evidence_wait"
    assert last["last_policy_action"] == "hold"


def test_turn_policy_rollback_allows_fast_false_resume_after_speech_end() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-backchannel"))
    owner.note_turn_policy_decision(
        Decision(
            action=Action.ROLLBACK,
            reason="intent:backchannel",
            rollback_drop_buffered=False,
            intent=InterruptIntent.BACKCHANNEL,
        ),
        transcript="嗯嗯",
        vad_active=True,
    )

    now = 10.55
    deferred = owner.defer_false_resume_after_speech_end(
        transcript="嗯嗯",
        duck_suspended=True,
    )

    assert deferred is False
    assert owner.should_hold_deadline() is False


def test_turn_policy_cancel_emits_confirm_cancel_decision() -> None:
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-cancel"))

    decision = owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="intent:hard_stop",
            intent=InterruptIntent.HARD_STOP,
        ),
        transcript="停一下",
        vad_active=True,
        eot_score=0.9,
    )

    assert decision.action is InterruptionDecisionAction.CONFIRM_CANCEL
    assert owner.state is InterruptionState.CONFIRMED_CANCELLED


def test_owner_decides_from_transcript_and_records_policy_once() -> None:
    timeline = TurnTimeline("turn-owner-decide")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=timeline)
    runtime = MagicMock()
    runtime.decide_from_transcript.return_value = Decision(
        action=Action.HOLD,
        reason="semantic_score_wait score=0.00 evidence=interim_substantive",
        intent=InterruptIntent.UNCERTAIN,
    )

    decision = owner.decide_from_transcript(
        runtime,
        "我想问一下",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=False,
    )

    assert decision.action is Action.HOLD
    runtime.decide_from_transcript.assert_called_once_with(
        "我想问一下",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=False,
        event_time_ms=None,
    )
    events = timeline.attrs["interruption_orchestrator_events"]
    assert [event["event"] for event in events].count("turn_policy_decision") == 1
    assert owner.blocks_framework_completed_turn() is True


def test_owner_blocks_framework_completed_turn_during_post_speech_wait() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-framework-block"))
    now = 10.5
    assert owner.defer_false_resume_after_speech_end(
        transcript="我想问一下",
        duck_suspended=True,
    )

    assert owner.blocks_framework_completed_turn() is True


def test_owner_only_commits_semantic_post_speech_cancel() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-normal-cancel"))
    now = 10.5
    assert owner.defer_false_resume_after_speech_end(
        transcript="我想问一下",
        duck_suspended=True,
    )
    owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="final_eot_score_high score=0.80",
            intent=InterruptIntent.NORMAL_INTERRUPT,
        ),
        transcript="我想问一下",
        vad_active=False,
        eot_score=0.8,
    )

    assert owner.should_commit_after_confirmed_cancel() is True

    hard_stop_owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    hard_stop_owner.start_candidate(timeline=TurnTimeline("turn-hard-stop"))
    now = 11.0
    assert hard_stop_owner.defer_false_resume_after_speech_end(
        transcript="不要讲了",
        duck_suspended=True,
    )
    hard_stop_owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="intent:hard_stop_speech_control",
            intent=InterruptIntent.HARD_STOP,
        ),
        transcript="不要讲了",
        vad_active=False,
        eot_score=0.0,
    )

    assert hard_stop_owner.should_commit_after_confirmed_cancel() is False
