"""TurnPolicyRuntime orchestration tests."""

from __future__ import annotations

from eidolon_sdk.biz.dialogue_control import TurnCommitBoundary

from eidolon.livekit.agent.turn_policy import Action, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def test_runtime_exposes_decision_timeout_from_config() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    assert runtime.decision_timeout_sec == 0.45


def test_runtime_builds_transcript_bound_committed_turn_decision() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())
    decision = runtime.committed_turn_decision(
        "换个话题",
        boundary=TurnCommitBoundary.FRAMEWORK_COMPLETED,
        eot_score=0.8,
    )

    metadata = decision.as_metadata()

    assert metadata["decision"] == "commit"
    assert "intent" not in metadata
    assert metadata["evidence"]["boundary"] == "framework_completed_turn"
    assert decision.matches_text("换个话题") is True


def test_runtime_fixed_phrases_have_no_control_authority() -> None:
    outcomes = []
    for text in ("别说了", "换个话题", "不是，我说错了", "普通用户内容"):
        runtime = TurnPolicyRuntime(TurnPolicyConfig())
        decision = runtime.decide_from_transcript(
            text,
            0.0,
            vad_active=True,
            agent_speaking=True,
            event_time_ms=100.0,
        )
        outcomes.append((decision.action, decision.intent.value, decision.tier))
        assert decision.topic_switch_hint is False
        assert decision.correction_hint is False

    assert outcomes == [(Action.HOLD, "uncertain", "tier2_interruption")] * 4


def test_runtime_high_eot_is_authoritative_independent_of_text() -> None:
    for text in ("换个话题", "普通用户内容"):
        decision = TurnPolicyRuntime(TurnPolicyConfig()).decide_from_transcript(
            text,
            0.82,
            vad_active=True,
            agent_speaking=True,
            event_time_ms=100.0,
        )

        assert decision.action is Action.CANCEL
        assert decision.intent.value == "normal_interrupt"
        assert decision.intent_source == "eot"
        assert decision.tier == "tier2_interruption"


def test_runtime_deadline_uses_evidence_not_lexical_content() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    low = runtime.deadline_decision(
        True,
        has_transcript=True,
        transcript="换个话题",
        eot_score=0.0,
    )
    high = runtime.deadline_decision(
        True,
        has_transcript=True,
        transcript="普通用户内容",
        eot_score=0.82,
    )

    assert low.action is Action.HOLD
    assert low.intent.value == "uncertain"
    assert low.topic_switch_hint is False
    assert high.action is Action.CANCEL
    assert high.intent_source == "eot"


def test_runtime_short_transcript_stays_reversible_without_semantic_evidence() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    deadline = runtime.deadline_decision(
        True,
        has_transcript=True,
        transcript="是",
        eot_score=0.0,
    )

    assert first.action is Action.HOLD
    assert first.intent.value == "uncertain"
    assert deadline.action is Action.HOLD
    assert deadline.reason.startswith("deadline_wait_for_more_transcript")


def test_runtime_annotates_normal_interrupt_as_tier2() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    decision = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=470.0,
    )

    assert decision.tier == "tier2_interruption"
    assert decision.reason.startswith("stable_normal_interrupt")


def test_runtime_low_score_normal_interrupt_stays_hold() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    decision = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=470.0,
    )

    assert decision.action is Action.HOLD
    assert decision.reason.startswith("semantic_score_wait")


def test_runtime_annotates_short_weak_signal_as_tier3() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "嗯",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.tier == "tier3_backchannel_noise"
    assert decision.tier_reason.startswith("weak_signal_short_transcript")


def test_runtime_stop_like_text_does_not_create_a_hard_stop_hint() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "别说了",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.HOLD
    assert decision.intent.value == "uncertain"


def test_runtime_short_correction_text_waits_for_more_evidence() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "不是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    second = runtime.decide_from_transcript(
        "不是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=180.0,
    )
    third = runtime.decide_from_transcript(
        "不是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=230.0,
    )

    assert first.action is Action.HOLD
    assert "insufficient_transcript_evidence" in first.reason
    assert first.hold_recheck_ms is None
    assert second.action is Action.HOLD
    assert third.action is Action.HOLD
    assert third.correction_hint is False


def test_runtime_hard_stop_prefix_stays_a_hint() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "别说",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.HOLD
    assert decision.intent.value == "uncertain"
    assert decision.tier == "tier3_backchannel_noise"


def test_runtime_long_topic_text_uses_normal_stability_window() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "换个话",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    too_soon = runtime.decide_from_transcript(
        "换个话",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=180.0,
    )
    stable = runtime.decide_from_transcript(
        "换个话",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=470.0,
    )

    assert first.action is Action.HOLD
    assert first.reason.startswith("semantic_score_wait")
    assert first.hold_recheck_ms == 350
    assert first.topic_switch_hint is False
    assert too_soon.action is Action.HOLD
    assert too_soon.hold_recheck_ms == 270
    assert stable.action is Action.CANCEL
    assert stable.intent.value == "normal_interrupt"
    assert stable.intent_source == "stable_signal"
    assert stable.topic_switch_hint is False


def test_runtime_short_correction_final_still_needs_semantic_score() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "不是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.HOLD
    assert decision.correction_hint is False


def test_runtime_late_correction_text_after_vad_end_waits_for_semantics() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "我刚才说错了",
        0.0,
        vad_active=False,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.HOLD
    assert decision.correction_hint is False


def test_runtime_normal_interrupt_requires_stable_substantive_text() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    too_soon = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=260.0,
    )
    stable = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=470.0,
    )

    assert first.action is Action.HOLD
    assert first.hold_recheck_ms == 350
    assert too_soon.action is Action.HOLD
    assert too_soon.hold_recheck_ms == 190
    assert stable.action is Action.CANCEL
    assert stable.reason.startswith("stable_normal_interrupt")


def test_runtime_short_latin_final_does_not_become_stable_normal_interrupt() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "If",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=True,
        event_time_ms=100.0,
    )
    second = runtime.decide_from_transcript(
        "If",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=True,
        event_time_ms=600.0,
    )

    assert first.action is Action.HOLD
    assert second.action is Action.HOLD
    assert not second.reason.startswith("stable_normal_interrupt")


def test_runtime_holds_normal_followup_shortly_after_weak_noise() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    noise = runtime.decide_from_transcript(
        "咳",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    followup = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=700.0,
    )

    assert noise.action is Action.HOLD
    assert followup.action is Action.HOLD
    assert followup.reason.startswith("semantic_score_wait")


def test_runtime_holds_normal_followup_shortly_after_short_artifact() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    artifact = runtime.decide_from_transcript(
        "是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    followup = runtime.decide_from_transcript(
        "那他的主要风险",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=700.0,
    )

    assert artifact.action is Action.HOLD
    assert followup.action is Action.HOLD
    assert followup.reason.startswith("semantic_score_wait")


def test_runtime_weak_followup_does_not_promote_text_to_control_intent() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    runtime.decide_from_transcript(
        "咳",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    decision = runtime.decide_from_transcript(
        "别说了",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=700.0,
    )

    assert decision.action is Action.HOLD
    assert decision.intent.value == "uncertain"


def test_runtime_mid_score_hold_does_not_mark_weak_signal() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    mid = runtime.decide_from_transcript(
        "",
        0.5,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    followup = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=700.0,
    )

    assert mid.action is Action.HOLD
    assert followup.action is Action.HOLD
    assert not followup.reason.startswith("weak_signal_followup_hold")


def test_runtime_high_semantic_score_bypasses_weak_followup_hold() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    runtime.decide_from_transcript(
        "咳",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    followup = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.82,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=700.0,
    )

    assert followup.action is Action.CANCEL
    assert followup.reason.startswith("eot_score_high")


def test_runtime_stable_normal_interrupt_does_not_bypass_weak_followup_hold() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    runtime.decide_from_transcript(
        "咳",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=500.0,
    )
    followup = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.25,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=900.0,
    )

    assert followup.action is Action.HOLD
    assert followup.reason.startswith("weak_signal_followup_hold")
