"""TurnPolicyRuntime orchestration tests."""

from __future__ import annotations

from eidolon.livekit.agent.turn_policy import Action, TurnPolicyRuntime
from eidolon.livekit.common.config import TurnPolicyConfig


def test_runtime_exposes_decision_timeout_from_config() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())

    assert runtime.decision_timeout_sec == 0.5


def test_runtime_decision_becomes_turn_control_metadata() -> None:
    runtime = TurnPolicyRuntime(TurnPolicyConfig())
    decision = runtime.decide_from_transcript(
        "换个话题",
        0.0,
        vad_active=True,
        agent_speaking=True,
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
    assert followup.reason.startswith("weak_signal_followup_hold")


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
    assert followup.reason.startswith("weak_signal_followup_hold")


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
    assert followup.action is Action.CANCEL
    assert not followup.reason.startswith("weak_signal_followup_hold")
