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
