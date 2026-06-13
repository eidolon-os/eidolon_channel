from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import (
    UserTurnCoordinator,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_low_eot_candidate_merges_short_continuation() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-merge")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("看你能不能", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)

    assert decision.action == "defer"
    assert decision.transcript == "看你能不能"

    clock.advance(0.4)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("帮我", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "看你能不能帮我"
    assert timeline.attrs["user_turn_coordinator"]["segments"] == 2


def test_late_final_revises_previous_segment_during_continuation() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-late-final-continuation")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("对不是陪伴了是给一个嗯", is_final=False)
    decision = coordinator.finish_speech(eot_score=0.1, should_defer=True)
    assert decision.action == "defer"

    clock.advance(0.3)
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院", is_final=False)
    coordinator.add_transcript("对，不是陪伴了，是给一个嗯。", is_final=True)
    coordinator.add_transcript("私立医院的。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "对，不是陪伴了，是给一个嗯。私立医院的。"
    assert coordinator.active is not None
    assert [revision.segment_index for revision in coordinator.active.revisions] == [
        0,
        1,
        0,
        1,
    ]


def test_multiple_stt_finals_inside_one_speech_are_preserved() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-multi-final")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("换个话题", is_final=False)
    coordinator.add_transcript("换个话题。", is_final=True)
    coordinator.add_transcript("我们聊一下", is_final=False)
    coordinator.add_transcript("我们聊一下定价。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "换个话题。我们聊一下定价。"
    assert timeline.attrs["user_turn_coordinator"]["segments"] == 2


def test_deferred_candidate_commits_once_after_grace() -> None:
    coordinator = UserTurnCoordinator(merge_grace_sec=0.8, low_eot_delay_sec=0.8)
    timeline = TurnTimeline("turn-deferred")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("比较调皮", is_final=False)
    coordinator.add_transcript("比较调皮，特别贪吃", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)

    assert decision.action == "defer"
    decision = coordinator.deferred_ready()

    assert decision.action == "commit"
    assert decision.transcript == "比较调皮，特别贪吃"
    coordinator.mark_committed(
        transcript=decision.transcript,
        reason="framework_commit_user_turn",
    )
    assert coordinator.snapshot()["state"] == "committed"


def test_late_final_after_commit_does_not_replace_committed_candidate() -> None:
    coordinator = UserTurnCoordinator()
    coordinator.start_speech(timeline=TurnTimeline("turn-late-final"))
    coordinator.add_transcript("你好", is_final=True)
    decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)
    coordinator.mark_committed(
        transcript=decision.transcript,
        reason="framework_commit_user_turn",
    )

    coordinator.add_transcript("你好世界", is_final=True)

    assert coordinator.selected_text == "你好"
    assert coordinator.snapshot()["state"] == "committed"


def test_framework_completed_after_commit_starts_fresh_candidate() -> None:
    coordinator = UserTurnCoordinator()
    first_timeline = TurnTimeline("turn-first")
    second_timeline = TurnTimeline("turn-second")

    coordinator.start_speech(timeline=first_timeline)
    coordinator.add_transcript("我想知道今天北京的天气怎么样。", is_final=True)
    first_decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)
    coordinator.mark_committed(
        transcript=first_decision.transcript,
        reason="framework_commit_user_turn",
    )

    second_decision = coordinator.mark_framework_completed(
        transcript="原来你认识铁。",
        reason="framework_completed_turn",
        timeline=second_timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert second_decision.action == "commit"
    assert second_decision.transcript == "原来你认识铁。"
    assert coordinator.snapshot()["candidate_id"] == "turn-second"
    assert coordinator.snapshot()["selected_text_preview"] == "原来你认识铁。"


def test_framework_deferred_after_commit_starts_fresh_candidate() -> None:
    coordinator = UserTurnCoordinator()
    first_timeline = TurnTimeline("turn-first")
    second_timeline = TurnTimeline("turn-second")

    coordinator.start_speech(timeline=first_timeline)
    coordinator.add_transcript("我想知道今天北京的天气怎么样。", is_final=True)
    first_decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)
    coordinator.mark_committed(
        transcript=first_decision.transcript,
        reason="framework_commit_user_turn",
    )

    second_decision = coordinator.defer_framework_completed(
        transcript="原来你认识铁。",
        reason="framework_completed_wait_for_continuation",
        timeline=second_timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert second_decision.action == "defer"
    assert second_decision.transcript == "原来你认识铁。"
    assert coordinator.snapshot()["candidate_id"] == "turn-second"
    assert coordinator.snapshot()["selected_text_preview"] == "原来你认识铁。"


def test_voiceprint_reject_has_explicit_reason() -> None:
    coordinator = UserTurnCoordinator()
    coordinator.start_speech(timeline=TurnTimeline("turn-non-owner"))
    coordinator.add_transcript("视频里的声音", is_final=True)
    coordinator.finish_speech(eot_score=1.0, should_defer=False)

    decision = coordinator.apply_voiceprint_result(
        SimpleNamespace(commit_allowed=False, commit_reason="speaker_not_owner")
    )

    assert decision.action == "reject"
    assert decision.reason == "voiceprint_blocked:speaker_not_owner"
    assert coordinator.snapshot()["reject_reason"] == "voiceprint_blocked:speaker_not_owner"


def test_voiceprint_accept_keeps_selected_transcript() -> None:
    coordinator = UserTurnCoordinator()
    coordinator.start_speech(timeline=TurnTimeline("turn-owner"))
    coordinator.add_transcript("我要继续测试", is_final=True)
    coordinator.finish_speech(eot_score=1.0, should_defer=False)

    decision = coordinator.apply_voiceprint_result(
        SimpleNamespace(commit_allowed=True, commit_reason="owner_high_confidence")
    )

    assert decision.action == "commit"
    assert decision.transcript == "我要继续测试"
    assert decision.reason == "voiceprint_allowed:owner_high_confidence"


def test_framework_completed_turn_marks_waiting_candidate_committed() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-completed")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("换个话题。", is_final=True)
    coordinator.add_transcript("我们聊一下定价。", is_final=True)
    coordinator.finish_speech(eot_score=0.1, should_defer=True)

    decision = coordinator.mark_framework_completed(
        transcript="我们聊一下定价。",
        reason="framework_completed_turn",
        timeline=timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert decision.action == "commit"
    assert decision.transcript == "换个话题。我们聊一下定价。"
    assert coordinator.snapshot()["state"] == "committed"
    assert coordinator.snapshot()["voiceprint_reason"] == "cached_owner_context"
    assert timeline.attrs["user_turn_coordinator"]["event"] == "framework_completed"


def test_framework_completed_turn_prefers_longer_framework_transcript() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-longer")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("不是给患者的。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)
    decision = coordinator.mark_framework_completed(
        transcript="给医生做的系统。 不是给患者的。",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.transcript == "给医生做的系统。 不是给患者的。"
    assert (
        timeline.attrs["user_turn_coordinator"]["selected_text_preview"]
        == "给医生做的系统。 不是给患者的。"
    )


def test_framework_completed_turn_can_wait_for_continuation() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-deferred")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院的。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)
    decision = coordinator.defer_framework_completed(
        transcript="私立医院的。 给医生做的系统。",
        reason="framework_completed_wait_for_continuation",
        timeline=timeline,
        voiceprint_reason="owner_high_confidence",
    )

    assert decision.action == "defer"
    assert decision.transcript == "私立医院的。 给医生做的系统。"
    assert coordinator.snapshot()["state"] == "waiting_merge"
    assert coordinator.snapshot()["voiceprint_reason"] == "owner_high_confidence"
    assert timeline.attrs["user_turn_coordinator"]["event"] == (
        "framework_completed_deferred"
    )


def test_inconclusive_voiceprint_can_wait_for_continuation() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        voiceprint_deferred_merge_grace_sec=4.0,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-voiceprint-deferred")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院的。", is_final=True)
    coordinator.finish_speech(eot_score=0.9, should_defer=False)
    decision = coordinator.defer_voiceprint_inconclusive(
        transcript="私立医院的。",
        reason="voiceprint_inconclusive:audio_too_short",
        timeline=timeline,
    )

    assert decision.action == "defer"
    assert decision.transcript == "私立医院的。"
    assert decision.delay_sec == 4.0
    assert coordinator.snapshot()["state"] == "waiting_merge"
    assert coordinator.snapshot()["voiceprint_reason"] == (
        "voiceprint_inconclusive:audio_too_short"
    )
    assert timeline.attrs["user_turn_coordinator"]["event"] == "voiceprint_deferred"

    clock.advance(1.5)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("给医生做的系统。", is_final=True)
    coordinator.add_transcript("不是给患者的。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "私立医院的。给医生做的系统。不是给患者的。"


def test_framework_completed_respects_inconclusive_voiceprint_merge_window() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        voiceprint_deferred_merge_grace_sec=4.0,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-framework-voiceprint-merge")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院的。", is_final=True)
    coordinator.finish_speech(eot_score=0.9, should_defer=False)
    coordinator.defer_voiceprint_inconclusive(
        transcript="私立医院的。",
        reason="voiceprint_inconclusive:audio_too_short",
        timeline=timeline,
    )

    clock.advance(1.5)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("主要给医生做的系统。", is_final=True)
    coordinator.finish_speech(eot_score=0.9, should_defer=True)

    assert coordinator.should_wait_for_deferred_voiceprint_merge()
    decision = coordinator.defer_framework_completed(
        transcript="私立医院的。主要给医生做的系统。",
        reason="framework_completed_wait_for_continuation",
        timeline=timeline,
        voiceprint_reason="owner_high_confidence",
    )

    assert decision.action == "defer"
    assert decision.delay_sec == 4.0
    assert coordinator.snapshot()["voiceprint_reason"] == (
        "voiceprint_inconclusive:audio_too_short"
    )

    clock.advance(2.5)
    assert coordinator.should_wait_for_deferred_voiceprint_merge()
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("不是给患者的。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert (
        decision.transcript
        == "私立医院的。主要给医生做的系统。不是给患者的。"
    )


def test_statement_sequence_can_extend_framework_completed_merge_window() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        statement_deferred_merge_grace_sec=3.5,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-statement-sequence")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院的。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)

    clock.advance(0.7)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("主要给医生做的系统。", is_final=True)
    coordinator.finish_speech(eot_score=0.55, should_defer=True)

    assert coordinator.should_wait_for_statement_sequence_merge()
    decision = coordinator.defer_framework_completed(
        transcript="私立医院的。主要给医生做的系统。",
        reason="framework_completed_wait_for_continuation",
        timeline=timeline,
        voiceprint_reason="owner_high_confidence",
    )

    assert decision.action == "defer"
    assert decision.delay_sec == 3.5
    assert coordinator.snapshot()["merge_reason"] == "low_eot_wait_for_continuation"

    clock.advance(2.6)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("不是给患者的。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert (
        decision.transcript
        == "私立医院的。主要给医生做的系统。不是给患者的。"
    )
