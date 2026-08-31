"""AgentSession turn_handling config for the full-duplex StreamingPipeline."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.contracts import INTERACTION_MODE_HALF_DUPLEX

from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.full_duplex.agent_builder import build_full_duplex_agent
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import EotPolicyConfig, TurnPolicyConfig


def _pipe(*, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._allow_interruptions = allow_interruptions
    p._false_interruption_timeout = 6.0
    p._turn_policy = TurnPolicyConfig()
    p._avatar_enabled = False
    return p


def test_channel_owned_turn_handling_disables_framework_auto_interrupt() -> None:
    th = _pipe(allow_interruptions=True)._build_turn_handling()
    intr = th["interruption"]
    assert intr["enabled"] is False
    # Channel policy still needs overlapping speech to reach STT.
    assert intr["discard_audio_if_uninterruptible"] is False
    assert intr["false_interruption_timeout"] == 6.0


def test_preemptive_passthrough() -> None:
    p = _pipe(allow_interruptions=True)
    th = p._build_turn_handling()
    assert th["preemptive_generation"]["enabled"] == p._turn_policy.preemptive.enabled
    assert (
        th["preemptive_generation"]["preemptive_tts"]
        == p._turn_policy.preemptive.preemptive_tts
    )


def test_livekit_17_agent_uses_structured_turn_handling_and_public_stt_node() -> None:
    pipeline = SimpleNamespace(
        _instructions="test",
        _factory=SimpleNamespace(
            stt=SimpleNamespace(stt=None),
            llm=SimpleNamespace(llm=None),
            tts=SimpleNamespace(tts=None),
            vad=None,
        ),
        _turn_detection=lambda: "vad",
    )

    agent = build_full_duplex_agent(pipeline)

    assert agent.turn_detection == "vad"
    assert inspect.isasyncgenfunction(agent.stt_node)


def test_turn_policy_speech_merge_grace_reaches_coordinator() -> None:
    p = _pipe(allow_interruptions=True)
    p._turn_policy = TurnPolicyConfig(
        eot=EotPolicyConfig(speech_merge_grace_ms=650),
    )
    coordinator = p._build_user_turn_coordinator()
    coordinator.start_speech(timeline=None, now=0.0)
    coordinator.note_speech_stopped(eot_score=0.0, now=0.1)

    assert coordinator.can_merge_new_speech(now=0.75) is True
    assert coordinator.can_merge_new_speech(now=0.751) is False


def test_streaming_pipeline_accepts_half_duplex_rejects_ptt() -> None:
    from eidolon_sdk.biz.contracts import INTERACTION_MODE_PTT

    factory = SharedStageFactory.__new__(SharedStageFactory)
    # ptt is served by the button-driven segment pipeline, not StreamingPipeline.
    with pytest.raises(ValueError, match="HalfDuplexPttPipeline"):
        StreamingPipeline(factory, interaction_mode=INTERACTION_MODE_PTT)
    # half_duplex IS served by StreamingPipeline (streaming EOT commit, no
    # barge-in). The mode guard must not reject it; deeper __init__ may fail on
    # the bare test factory, but never with the guard's ValueError.
    try:
        StreamingPipeline(factory, interaction_mode=INTERACTION_MODE_HALF_DUPLEX)
    except ValueError as exc:
        assert "HalfDuplexPttPipeline" not in str(exc)
    except Exception:
        pass
