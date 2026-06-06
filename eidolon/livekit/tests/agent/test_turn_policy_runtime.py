"""TurnPolicyRuntime orchestration tests."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy import Action, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def test_runtime_exposes_decision_timeout_from_config() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    assert runtime.decision_timeout_sec == 0.5


def test_runtime_decision_becomes_turn_control_metadata() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())
    runtime.decide_from_transcript(
        "换个话题",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    decision = runtime.decide_from_transcript(
        "换个话题",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=240.0,
    )

    signal = runtime.control_signal_from_decision(
        decision,
        interrupted_text_excerpt="被打断的回复",
        played_seconds=1.25,
        latency_ms=120.0,
    )
    metadata = signal.as_metadata()

    assert decision.action is Action.CANCEL
    assert metadata["intent"] == "topic_switch"
    assert metadata["topic_switch_hint"] is True
    assert metadata["correction_hint"] is False
    assert metadata["interrupted_text_excerpt"] == "被打断的回复"
    assert metadata["played_seconds"] == 1.25
    assert metadata["latency_ms"] == 120.0


def test_runtime_hard_stop_bypasses_stable_signal_window() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "别说了",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.CANCEL
    assert decision.reason.startswith("intent:hard_stop")


def test_runtime_correction_waits_for_short_stability_window() -> None:
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
    assert first.reason.startswith("stable_signal_wait")
    assert second.action is Action.HOLD
    assert third.action is Action.CANCEL
    assert third.correction_hint is True


def test_runtime_correction_final_can_confirm_without_waiting_window() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "不是",
        0.0,
        vad_active=True,
        agent_speaking=True,
        is_final=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.CANCEL
    assert decision.correction_hint is True


def test_runtime_late_correction_after_vad_end_bypasses_stable_window() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    decision = runtime.decide_from_transcript(
        "我刚才说错了",
        0.0,
        vad_active=False,
        agent_speaking=True,
        event_time_ms=100.0,
    )

    assert decision.action is Action.CANCEL
    assert decision.correction_hint is True


def test_runtime_normal_interrupt_requires_stable_substantive_text() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    first = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=100.0,
    )
    too_soon = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=260.0,
    )
    stable = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=470.0,
    )

    assert first.action is Action.HOLD
    assert too_soon.action is Action.HOLD
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


def test_runtime_noise_followup_does_not_block_hard_stop() -> None:
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

    assert decision.action is Action.CANCEL


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
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=500.0,
    )
    followup = runtime.decide_from_transcript(
        "那它的主要风险是什么",
        0.0,
        vad_active=True,
        agent_speaking=True,
        event_time_ms=900.0,
    )

    assert followup.action is Action.HOLD
    assert followup.reason.startswith("weak_signal_followup_hold")
