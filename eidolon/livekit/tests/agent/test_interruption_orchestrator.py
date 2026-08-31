"""InterruptionOrchestrator lifecycle tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.interruption_orchestrator import (
    InterruptionDecisionAction,
    InterruptionOrchestrator,
    InterruptionState,
    InterruptionVerdictAction,
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


def test_new_acoustic_generation_supersedes_stale_active_candidate() -> None:
    first_timeline = TurnTimeline("turn-first-generation")
    second_timeline = TurnTimeline("turn-second-generation")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=first_timeline, generation_id=1)

    decision = owner.start_candidate(timeline=second_timeline, generation_id=2)

    stale = owner.verdict_for(first_timeline.turn_id, generation_id=1)
    assert stale is not None
    assert stale.action is InterruptionVerdictAction.REJECTED_CANDIDATE
    assert decision.action is InterruptionDecisionAction.SOFT_SUSPEND_OUTPUT
    assert owner.active is True
    assert second_timeline.attrs["interruption_orchestrator_last_event"]["generation_id"] == 2


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


def test_provisional_short_latin_artifact_uses_bounded_evidence_window() -> None:
    now = 10.0

    def clock() -> float:
        return now

    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        no_evidence_timeout_sec=0.8,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-latin-artifact"))
    owner.note_transcript("Okay", is_final=False)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="transcript_evidence_hold:short_latin_artifact cjk=0 latin=4",
            intent=InterruptIntent.UNCERTAIN,
        ),
        transcript="Okay",
        vad_active=True,
    )

    now = 10.55
    assert owner.defer_false_resume_after_speech_end(
        transcript="Okay",
        duck_suspended=True,
    )
    assert owner.max_suspend_sec() == 0.8
    assert owner.should_hold_deadline() is True

    now = 11.36
    assert owner.should_hold_deadline() is False


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


def test_single_char_backchannel_fast_resumes_on_speech_end() -> None:
    now = 10.0

    def clock() -> float:
        return now

    timeline = TurnTimeline("turn-short-backchannel")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=timeline)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="intent:backchannel_await_more_speech",
            rollback_drop_buffered=False,
            intent=InterruptIntent.BACKCHANNEL,
        ),
        transcript="好",
        vad_active=True,
    )

    now = 10.55
    deferred = owner.defer_false_resume_after_speech_end(
        transcript="好",
        duck_suspended=True,
    )

    assert deferred is False
    assert owner.should_hold_deadline() is False
    assert (
        timeline.attrs["interruption_orchestrator_last_event"]["event"]
        == "short_false_interruption_fast_resume"
    )
    last = timeline.attrs["interruption_orchestrator_last_event"]
    assert round(last["elapsed_ms"]) == 550


def test_single_char_backchannel_cannot_end_candidate_while_speech_is_active() -> None:
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-active-backchannel"))
    owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="intent:backchannel_await_more_speech",
            intent=InterruptIntent.BACKCHANNEL,
        ),
        transcript="好",
        vad_active=True,
    )

    assert owner.should_hold_deadline() is True
    assert owner.max_suspend_sec() == 6.0


def test_interruption_owner_events_record_elapsed_and_delta_ms() -> None:
    now = 10.0

    def clock() -> float:
        return now

    timeline = TurnTimeline("turn-event-timing")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )

    owner.start_candidate(timeline=timeline)
    now = 10.12
    owner.note_transcript("好", is_final=False)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="intent:backchannel_await_more_speech",
            rollback_drop_buffered=False,
            intent=InterruptIntent.BACKCHANNEL,
        ),
        transcript="好",
        vad_active=True,
    )
    now = 10.55
    owner.defer_false_resume_after_speech_end(
        transcript="好",
        duck_suspended=True,
    )

    events = timeline.attrs["interruption_orchestrator_events"]
    assert events[0]["event"] == "candidate_started"
    assert round(events[0]["elapsed_ms"]) == 0
    assert "since_last_event_ms" not in events[0]
    assert events[1]["event"] == "transcript_evidence"
    assert round(events[1]["elapsed_ms"]) == 120
    assert round(events[1]["since_last_event_ms"]) == 120
    assert events[2]["event"] == "turn_policy_decision"
    assert round(events[2]["elapsed_ms"]) == 120
    assert round(events[2]["since_last_event_ms"]) == 0
    assert events[-1]["event"] == "short_false_interruption_fast_resume"
    assert round(events[-1]["elapsed_ms"]) == 550
    assert round(events[-1]["since_last_event_ms"]) == 430


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


def test_owner_routes_transcript_then_effect_records_policy_once() -> None:
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
    owner.note_turn_policy_decision(
        decision,
        source="turn_policy",
        transcript="我想问一下",
        vad_active=True,
        eot_score=0.0,
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


def test_semantic_cancel_collects_until_speech_end_before_commit() -> None:
    now = 10.0

    def clock() -> float:
        return now

    timeline = TurnTimeline("turn-active-correction")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=timeline)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="intent:correction",
            intent=InterruptIntent.NORMAL_INTERRUPT,
        ),
        transcript="不是",
        vad_active=True,
        eot_score=0.0,
    )

    assert owner.should_collect_after_confirmed_cancel() is True
    assert owner.should_commit_after_confirmed_cancel() is False
    owner.mark_confirmed_cancel_collecting_turn()
    assert owner.blocks_framework_completed_turn() is True
    assert owner.state is InterruptionState.CONFIRMED_CANCEL_COLLECTING_TURN

    now = 10.9
    assert owner.finish_confirmed_cancel_speech("不是，我刚才说错了") is True

    verdict = owner.verdict_for(timeline.turn_id, generation_id=1)
    assert verdict is not None
    assert verdict.action is InterruptionVerdictAction.CONFIRMED_CANCEL
    assert verdict.continue_to_llm is True
    assert verdict.transcript == "不是，我刚才说错了"
    assert owner.active is False
    assert owner.state is InterruptionState.IDLE
    assert (
        timeline.attrs["interruption_verdict"]["reason"]
        == "confirmed_cancel_speech_completed"
    )


def test_hard_stop_cancel_does_not_collect_user_turn() -> None:
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-active-hard-stop"))
    owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="intent:hard_stop",
            intent=InterruptIntent.HARD_STOP,
        ),
        transcript="停一下",
        vad_active=True,
        eot_score=0.0,
    )

    assert owner.should_collect_after_confirmed_cancel() is False
    assert owner.finish_confirmed_cancel_speech("停一下") is False


def test_resolved_normal_interrupt_exposes_committable_verdict() -> None:
    timeline = TurnTimeline("turn-normal-interrupt")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=timeline)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.CANCEL,
            reason="eot_score_high",
            intent=InterruptIntent.NORMAL_INTERRUPT,
        ),
        transcript="换一个方向继续讲",
        vad_active=False,
        eot_score=1.0,
    )

    owner.resolve(action="cancel", reason="eot_cancel")

    verdict = owner.verdict_for(timeline.turn_id, generation_id=1)
    assert verdict is not None
    assert verdict.action is InterruptionVerdictAction.CONFIRMED_CANCEL
    assert verdict.continue_to_llm is True
    assert verdict.transcript == "换一个方向继续讲"
    assert timeline.attrs["interruption_verdict"]["continue_to_llm"] is True


def test_resolved_backchannel_exposes_non_committable_verdict() -> None:
    timeline = TurnTimeline("turn-backchannel-verdict")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=timeline)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.ROLLBACK,
            reason="intent:backchannel",
            intent=InterruptIntent.BACKCHANNEL,
        ),
        transcript="right",
        vad_active=False,
        eot_score=0.0,
    )

    owner.resolve(action="rollback", reason="backchannel")

    verdict = owner.verdict_for(timeline.turn_id, generation_id=1)
    assert verdict is not None
    assert verdict.action is InterruptionVerdictAction.REJECTED_RESUME
    assert verdict.continue_to_llm is False


def test_verdict_is_not_visible_to_a_later_acoustic_generation() -> None:
    timeline = TurnTimeline("turn-generation-isolation")
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        min_speech_sec=0.25,
    )
    owner.start_candidate(timeline=timeline, generation_id=1)
    owner.resolve(action="rollback", reason="timeout")

    assert owner.verdict_for(timeline.turn_id, generation_id=1) is not None
    assert owner.verdict_for(timeline.turn_id, generation_id=2) is None
