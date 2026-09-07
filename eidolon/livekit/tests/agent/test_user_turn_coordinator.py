from __future__ import annotations

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator


def _owner_events(timeline: TurnTimeline) -> list[tuple[str, str, str]]:
    return [
        (entry["owner"], entry["event"], entry["reason"])
        for entry in timeline.attrs["user_turn_owner_ledger"]["transitions"]
    ]


@pytest.mark.parametrize('query,expected', [
    ('我刚才说错了。', '不是我刚才说错了。'),
    ('不是。 我刚才说错了。', '不是我刚才说错了。'),
    ('不是。', '不是。'),
    ('我刚才说', '我刚才说'),
    ('完全不同的话', '完全不同的话'),
    ('', ''),
])
def test_endpointing_resolves_only_current_complete_candidate(query, expected):
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=.8)
    coordinator.start_speech(timeline=TurnTimeline('candidate'))
    coordinator.add_transcript('不是。', is_final=True)
    coordinator.add_transcript('我刚才说', is_final=False)
    coordinator.add_transcript('我刚才说错了。', is_final=True)
    version = coordinator.transcript_change_version
    assert coordinator.endpointing_text(query) == expected
    assert coordinator.transcript_change_version == version
    assert coordinator.active.state == 'open'


@pytest.mark.parametrize('state', ['pending_interim', 'new_speech', 'rejected', 'committed'])
def test_endpointing_does_not_borrow_pending_or_terminal_context(state):
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=.8)
    timeline = TurnTimeline('candidate')
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript('不是。', is_final=True)
    coordinator.add_transcript('我刚才说错了。', is_final=False)
    coordinator.add_transcript('我刚才说错了。', is_final=True)
    if state == 'pending_interim':
        coordinator.add_transcript('还有', is_final=False)
    elif state == 'new_speech':
        coordinator.note_speech_stopped(eot_score=.6)
        coordinator.start_speech(timeline=timeline)
    elif state == 'rejected':
        coordinator.reject_active('test rejection')
    else:
        coordinator.mark_framework_completed(
            transcript=coordinator.selected_text, timeline=timeline, reason='test completion',
        )
    assert coordinator.endpointing_text('我刚才说错了。') == '我刚才说错了。'


def test_interim_after_final_starts_a_new_sentence_segment() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=TurnTimeline("turn-final-then-interim"))

    coordinator.add_transcript("好啊。", is_final=True)
    coordinator.add_transcript("那你记下来吧，这是我们约定。", is_final=False)

    assert coordinator.selected_text == "好啊。那你记下来吧，这是我们约定。"
    assert coordinator.active is not None
    assert [segment.selected_text for segment in coordinator.active.segments] == [
        "好啊。",
        "那你记下来吧，这是我们约定。",
    ]
    assert [revision.segment_index for revision in coordinator.active.revisions] == [0, 1]


def test_policy_text_folds_provider_segment_boundary_without_changing_chat_text() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=TurnTimeline("turn-provider-segments"))

    coordinator.add_transcript("换个。", is_final=True)
    coordinator.add_transcript("话题", is_final=False)

    assert coordinator.selected_text == "换个。话题"
    assert coordinator.selected_policy_text == "换个话题"


def test_vad_stop_records_boundary_without_terminal_decision() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    coordinator.start_speech(timeline=TurnTimeline("turn-vad-stop"))
    coordinator.add_transcript("那你记下来吧", is_final=False)

    coordinator.note_speech_stopped(eot_score=0.1, now=10.0)

    assert coordinator.active is not None
    assert coordinator.active.state == "open"
    assert coordinator.active.current_segment is not None
    assert coordinator.active.current_segment.ended_at == 10.0
    assert coordinator.selected_text == "那你记下来吧"


def test_new_speech_before_framework_completion_stays_in_same_product_turn() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-continuation")
    first = coordinator.start_speech(timeline=timeline, now=0.0)
    coordinator.add_transcript("看你能不能", is_final=True, now=0.1)
    coordinator.note_speech_stopped(eot_score=0.01, now=0.2)

    assert coordinator.can_merge_new_speech(now=0.7)
    second = coordinator.start_speech(timeline=timeline, now=0.7)
    coordinator.add_transcript("帮我查天气", is_final=True, now=0.8)

    assert second is first
    assert coordinator.selected_text == "看你能不能帮我查天气"
    assert timeline.attrs["user_turn_coordinator"]["segments"] == 2


def test_new_speech_outside_merge_grace_starts_new_candidate() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    first = coordinator.start_speech(timeline=TurnTimeline("turn-old"), now=0.0)
    coordinator.note_speech_stopped(eot_score=0.0, now=0.2)

    assert coordinator.can_merge_new_speech(now=1.1) is False
    second = coordinator.start_speech(timeline=TurnTimeline("turn-new"), now=1.1)

    assert second is not first
    assert second.candidate_id == "turn-new"
    assert first.state == "rejected"
    assert first.reject_reason == "superseded_by_new_speech"
    assert second.latest_generation_id == 2


def test_explicit_ingress_resolution_covers_pending_generation() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-repeated-hypothesis")
    coordinator.start_speech(timeline=timeline, now=0.0)
    first = coordinator.add_transcript("铁锤三二五", is_final=False, now=0.1)
    coordinator.note_speech_stopped(eot_score=0.0, now=0.2)
    coordinator.start_speech(timeline=timeline, now=0.5)
    coordinator.add_transcript("铁锤三二五", is_final=False, now=0.6)
    final = coordinator.add_transcript("铁锤三二五。", is_final=True, now=0.7)
    assert first is not None
    assert final is not None
    assert coordinator.cover_pending_transcript_segments(
        candidate_id=final.candidate_id,
        segment_indexes=(first.segment_index,),
        covered_by_generation_id=final.generation_id,
        covered_by_segment_index=final.segment_index,
        reason="provider_final_equivalent_hypothesis",
        now=0.7,
    ) == 1

    readiness = coordinator.framework_completion_readiness("铁锤三二五。")

    assert readiness.ready is True
    assert coordinator.selected_text == "铁锤三二五。"
    assert coordinator.active is not None
    assert [segment.final_text for segment in coordinator.active.segments] == [
        "",
        "铁锤三二五。",
    ]
    assert coordinator.active.segments[0].covered_by_generation_id == 2
    assert coordinator.active.segments[0].covered_by_segment_index == 1
    assert coordinator.active.segments[0].coverage_reason == (
        "provider_final_equivalent_hypothesis"
    )


@pytest.mark.parametrize('framework,pending,covered', [
    ('帮我介绍一下这个方案。', '帮我介绍一下这个方案的优点和缺点', False),
    ('Book a table', 'Book a table for seven people', False),
    ('号码是1234', '号码是12345678', False),
    ('帮我介绍一下这个方案的优点和缺点。', '帮我介绍一下这个方案', True),
    ('BOOK a table for seven people.', 'Book a table for seven people', True),
])
def test_framework_coverage_is_directional(framework, pending, covered):
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=.8)
    coordinator.start_speech(timeline=None)
    coordinator.add_transcript('好的。', is_final=True)
    coordinator.add_transcript(pending, is_final=False)
    readiness = coordinator.framework_completion_readiness(framework)
    assert readiness.ready is covered


def test_framework_completed_is_the_only_normal_commit_boundary() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-framework")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("帮我查一下天气", is_final=True)
    coordinator.note_speech_stopped(eot_score=0.9)

    assert coordinator.active is not None
    assert coordinator.active.state == "open"

    decision = coordinator.mark_framework_completed(
        transcript="帮我查一下天气。",
        reason="framework_completed_turn",
        timeline=timeline,
        now=42.0,
    )

    assert decision.action == "commit"
    assert decision.transcript == "帮我查一下天气。"
    assert coordinator.active.state == "committed"
    assert coordinator.active.committed_at == 42.0
    assert timeline.timestamps["turn_committed_at"] == 42.0
    assert _owner_events(timeline)[-1] == (
        "accepted_user_turn",
        "framework_completed",
        "framework_completed_turn",
    )


def test_framework_completion_keeps_final_then_interim_canonical_text() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-canonical")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("好啊。", is_final=True)
    coordinator.add_transcript("那你记下来吧，这是我们约定。", is_final=False)

    decision = coordinator.mark_framework_completed(
        transcript="那你记下来吧，这是我们约定。",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "好啊。那你记下来吧，这是我们约定。"


def test_prepare_framework_completion_exposes_canonical_before_terminal_decision() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-prepare-canonical")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("好啊。", is_final=True)
    coordinator.note_speech_stopped(eot_score=0.0)
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("那你记下来吧，这是我们约定。", is_final=False)

    canonical = coordinator.prepare_framework_completed(
        transcript="好啊。",
        timeline=timeline,
    )

    assert canonical == "好啊。那你记下来吧，这是我们约定。"
    assert coordinator.active is not None
    assert coordinator.active.state == "open"


def test_late_final_revises_previous_segment_during_continuation() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-late-final")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("对不是陪伴了是给一个嗯", is_final=False)
    coordinator.note_speech_stopped(eot_score=0.1)
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("私立医院", is_final=False)
    coordinator.add_transcript("对，不是陪伴了，是给一个嗯。", is_final=True)
    coordinator.add_transcript("私立医院的。", is_final=True)

    assert coordinator.selected_text == "对，不是陪伴了，是给一个嗯。私立医院的。"
    assert coordinator.active is not None
    assert [revision.segment_index for revision in coordinator.active.revisions] == [
        0,
        1,
        0,
        1,
    ]


def test_repeated_trailing_final_is_idempotent() -> None:
    """Box-3 turn 5 emitted the same trailing backchannel FINAL twice."""

    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-repeated-trailing-final")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("我觉得你有机会可以去一下。", is_final=True)
    coordinator.add_transcript("嗯。", is_final=True)
    coordinator.add_transcript("嗯。", is_final=True)

    decision = coordinator.mark_framework_completed(
        transcript="我觉得你有机会可以去一下。",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "我觉得你有机会可以去一下。嗯。"
    assert coordinator.active is not None
    assert [segment.selected_text for segment in coordinator.active.segments] == [
        "我觉得你有机会可以去一下。",
        "嗯。",
    ]


def test_duplicate_framework_completion_is_not_committed_twice() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-duplicate")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("你好", is_final=True)
    first = coordinator.mark_framework_completed(
        transcript="你好。",
        reason="framework_completed_turn",
        timeline=timeline,
    )
    second = coordinator.mark_framework_completed(
        transcript="你好",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert first.action == "commit"
    assert second.action == "none"
    assert second.reason == "committed_turn_revision"


def test_materially_different_duplicate_completion_on_same_timeline_is_noop() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-duplicate-different")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("第一版", is_final=True)
    coordinator.mark_framework_completed(
        transcript="第一版",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    duplicate = coordinator.mark_framework_completed(
        transcript="完全不同的第二版",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert duplicate.action == "none"
    assert duplicate.reason == "framework_completed_duplicate:committed"
    assert coordinator.selected_text == "第一版"


def test_duplicate_speech_start_does_not_replace_open_candidate() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-duplicate-start")
    first = coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("还在说", is_final=False)

    duplicate = coordinator.start_speech(timeline=timeline)

    assert duplicate is first
    assert len(duplicate.segments) == 1
    assert coordinator.selected_text == "还在说"


def test_framework_boundary_does_not_classify_meta_language() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-meta")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("我再说一下", is_final=True)

    decision = coordinator.mark_framework_completed(
        transcript="我再说一下",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "我再说一下"


def test_framework_boundary_preserves_all_accepted_segments() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    timeline = TurnTimeline("turn-meta-tail")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("帮我查天气。", is_final=True)
    coordinator.add_transcript("那我再说一下。", is_final=True)

    decision = coordinator.mark_framework_completed(
        transcript="那我再说一下。",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "帮我查天气。那我再说一下。"


def test_empty_framework_turn_is_rejected() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)

    decision = coordinator.mark_framework_completed(
        transcript="",
        reason="framework_completed_turn",
    )

    assert decision.action == "reject"
    assert decision.reason == "empty_transcript"


def test_new_speech_after_terminal_turn_gets_new_candidate() -> None:
    coordinator = UserTurnCoordinator(speech_merge_grace_sec=0.8)
    first_timeline = TurnTimeline("turn-first")
    coordinator.start_speech(timeline=first_timeline)
    coordinator.add_transcript("第一轮", is_final=True)
    coordinator.mark_framework_completed(
        transcript="第一轮",
        reason="framework_completed_turn",
        timeline=first_timeline,
    )

    second = coordinator.start_speech(timeline=TurnTimeline("turn-second"))

    assert second.candidate_id == "turn-second"
    assert second.state == "open"
