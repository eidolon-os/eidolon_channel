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


def _owner_events(timeline: TurnTimeline) -> list[tuple[str, str, str]]:
    return [
        (entry["owner"], entry["event"], entry["reason"])
        for entry in timeline.attrs["user_turn_owner_ledger"]["transitions"]
    ]


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
    assert _owner_events(timeline)[-1] == (
        "accepted_user_turn",
        "committed",
        "framework_commit_user_turn",
    )


def test_unmerged_pending_candidate_gets_superseded_owner() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    old_timeline = TurnTimeline("turn-old")
    new_timeline = TurnTimeline("turn-new")

    coordinator.start_speech(timeline=old_timeline)
    coordinator.add_transcript("你给我查查今天的天气吧。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)
    assert decision.action == "defer"

    clock.advance(1.0)
    assert not coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=new_timeline)

    assert coordinator.snapshot()["candidate_id"] == "turn-new"
    assert old_timeline.attrs["user_turn_coordinator"]["state"] == "rejected"
    assert old_timeline.attrs["user_turn_coordinator"]["reject_reason"] == (
        "superseded_by_new_speech"
    )
    assert old_timeline.attrs["user_turn_superseded_by"] == {
        "replacement_candidate_id": "turn-new",
        "reason": "superseded_by_new_speech",
    }
    assert _owner_events(old_timeline)[-1] == (
        "superseded_user_turn",
        "superseded",
        "superseded_by_new_speech",
    )


def test_rejected_replacement_restores_superseded_pending_candidate() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    old_timeline = TurnTimeline("turn-old-restore")
    new_timeline = TurnTimeline("turn-new-rejected")

    coordinator.start_speech(timeline=old_timeline)
    coordinator.add_transcript("你给我查查今天的天气吧。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)
    assert decision.action == "defer"

    clock.advance(1.0)
    coordinator.start_speech(timeline=new_timeline)
    coordinator.add_transcript("那我再说了。", is_final=True)
    replacement = coordinator.finish_speech(eot_score=1.0, should_defer=False)
    assert replacement.action == "reject"

    restored = coordinator.restore_superseded_candidate_if_replacement_rejected(
        replacement.reason
    )

    assert restored.action == "commit"
    assert restored.reason == "superseded_candidate_restored"
    assert restored.candidate_id == "turn-old-restore"
    assert restored.transcript == "你给我查查今天的天气吧。"
    assert coordinator.snapshot()["candidate_id"] == "turn-old-restore"
    assert coordinator.snapshot()["state"] == "waiting_merge"
    assert _owner_events(old_timeline)[-1] == (
        "provisional_user_turn",
        "superseded_restored",
        "replacement_rejected:non_actionable_meta_turn",
    )


def test_accepted_replacement_finalizes_superseded_candidate() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    old_timeline = TurnTimeline("turn-old-finalized")
    new_timeline = TurnTimeline("turn-new-accepted")

    coordinator.start_speech(timeline=old_timeline)
    coordinator.add_transcript("你给我查查今天的天气吧。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)

    clock.advance(1.0)
    coordinator.start_speech(timeline=new_timeline)
    coordinator.add_transcript("那我们换个话题。", is_final=True)
    decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)

    assert decision.action == "commit"
    assert _owner_events(old_timeline)[-1] == (
        "superseded_user_turn",
        "superseded_finalized",
        "replacement_accepted:speech_finished",
    )
    assert old_timeline.attrs["user_turn_superseded_finalized"] == {
        "replacement_candidate_id": "turn-new-accepted",
        "reason": "replacement_accepted:speech_finished",
    }
    restored = coordinator.restore_superseded_candidate_if_replacement_rejected(
        "late_reject"
    )
    assert restored.action == "none"
    assert restored.reason == "no_superseded_candidate"


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


def test_non_actionable_meta_turn_rejected_before_commit() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-meta")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("那我再说了。", is_final=True)
    decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)

    assert decision.action == "reject"
    assert decision.reason == "non_actionable_meta_turn"
    assert decision.transcript == "那我再说了。"
    assert coordinator.snapshot()["state"] == "rejected"
    assert coordinator.snapshot()["last_owner_transition"]["owner"] == (
        "rejected_user_turn"
    )
    assert timeline.attrs["user_turn_coordinator"]["event"] == "rejected"
    assert _owner_events(timeline)[-1] == (
        "rejected_user_turn",
        "rejected",
        "non_actionable_meta_turn",
    )


def test_non_actionable_meta_turn_rejected_before_deferred_commit() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-meta-deferred")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("我再说一下。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)

    assert decision.action == "reject"
    assert decision.reason == "non_actionable_meta_turn"
    assert coordinator.snapshot()["state"] == "rejected"
    assert timeline.attrs["user_turn_coordinator"]["merge_reason"] == ""


def test_non_actionable_meta_tail_does_not_attach_to_prior_owner() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=clock,
    )
    timeline = TurnTimeline("turn-meta-tail")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("你给我查查今天的天气吧。", is_final=True)
    decision = coordinator.finish_speech(eot_score=0.01, should_defer=True)
    assert decision.action == "defer"

    clock.advance(0.4)
    assert coordinator.can_merge_new_speech()
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("那我再说了。", is_final=True)
    decision = coordinator.finish_speech(eot_score=1.0, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "你给我查查今天的天气吧。"
    assert coordinator.snapshot()["state"] == "waiting_voiceprint"
    assert coordinator.snapshot()["last_owner_transition"]["event"] == (
        "waiting_voiceprint"
    )
    assert (
        "dropped_fragment",
        "non_actionable_meta_tail_dropped",
        "non_actionable_meta_turn",
    ) in _owner_events(timeline)
    assert (
        timeline.attrs["user_turn_coordinator"]["selected_text_preview"]
        == "你给我查查今天的天气吧。"
    )


def test_framework_completed_non_actionable_meta_turn_is_not_accepted_owner() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-meta")

    decision = coordinator.mark_framework_completed(
        transcript="那我再说了。",
        reason="framework_completed_turn",
        timeline=timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert decision.action == "reject"
    assert decision.reason == "non_actionable_meta_turn"
    assert coordinator.snapshot()["state"] == "rejected"
    assert coordinator.snapshot()["commit_reason"] == ""
    assert timeline.attrs["user_turn_coordinator"]["event"] == "rejected"


def test_framework_completed_meta_tail_does_not_replace_waiting_owner() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-meta-tail")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("你给我查查今天的天气吧。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)
    decision = coordinator.mark_framework_completed(
        transcript="那我再说了。",
        reason="framework_completed_turn",
        timeline=timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert decision.action == "commit"
    assert decision.transcript == "你给我查查今天的天气吧。"
    assert coordinator.snapshot()["state"] == "committed"
    assert coordinator.snapshot()["commit_reason"] == "framework_completed_turn"
    assert (
        "dropped_fragment",
        "non_actionable_meta_tail_dropped",
        "non_actionable_meta_turn",
    ) in _owner_events(timeline)
    assert _owner_events(timeline)[-1] == (
        "accepted_user_turn",
        "framework_completed",
        "framework_completed_turn",
    )


def test_framework_deferred_non_actionable_meta_turn_is_not_waiting_owner() -> None:
    coordinator = UserTurnCoordinator()
    timeline = TurnTimeline("turn-framework-meta-deferred")

    decision = coordinator.defer_framework_completed(
        transcript="我重新说一下。",
        reason="framework_completed_wait_for_continuation",
        timeline=timeline,
        voiceprint_reason="cached_owner_context",
    )

    assert decision.action == "reject"
    assert decision.reason == "non_actionable_meta_turn"
    assert coordinator.snapshot()["state"] == "rejected"
    assert coordinator.snapshot()["merge_reason"] == ""


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
    assert timeline.attrs["user_turn_coordinator"]["event"] == ("framework_completed_deferred")


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
    assert decision.transcript == "私立医院的。主要给医生做的系统。不是给患者的。"


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
    assert decision.transcript == "私立医院的。主要给医生做的系统。不是给患者的。"


def test_statement_sequence_merge_max_cjk_chars_is_configurable() -> None:
    clock = _Clock()
    coordinator = UserTurnCoordinator(
        merge_grace_sec=0.8,
        statement_deferred_merge_grace_sec=3.5,
        low_eot_delay_sec=0.8,
        statement_sequence_merge_max_cjk_chars=4,
        clock=clock,
    )
    timeline = TurnTimeline("turn-statement-sequence-threshold")

    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院的。", is_final=True)
    coordinator.finish_speech(eot_score=0.01, should_defer=True)

    clock.advance(0.7)
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("主要给医生做的系统。", is_final=True)
    coordinator.finish_speech(eot_score=0.55, should_defer=True)

    assert not coordinator.should_wait_for_statement_sequence_merge()
    clock.advance(0.9)
    assert not coordinator.can_merge_new_speech()
