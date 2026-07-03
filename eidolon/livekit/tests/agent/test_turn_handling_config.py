"""AgentSession turn_handling config for the full-duplex StreamingPipeline."""

from __future__ import annotations

import pytest
from eidolon_sdk.biz.contracts import INTERACTION_MODE_HALF_DUPLEX

from eidolon.livekit.agent.streaming import StreamingPipeline
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import TurnPolicyConfig


def _pipe(*, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._allow_interruptions = allow_interruptions
    p._false_interruption_timeout = 6.0
    p._turn_policy = TurnPolicyConfig()
    return p


def test_full_duplex_turn_handling_uses_open_mic_defaults() -> None:
    th = _pipe(allow_interruptions=True)._build_turn_handling()
    intr = th["interruption"]
    assert intr["enabled"] is True
    assert intr["discard_audio_if_uninterruptible"] is True
    assert intr["false_interruption_timeout"] == 6.0


def test_preemptive_passthrough() -> None:
    p = _pipe(allow_interruptions=True)
    th = p._build_turn_handling()
    assert th["preemptive_generation"]["enabled"] == p._turn_policy.preemptive.enabled
    assert (
        th["preemptive_generation"]["preemptive_tts"]
        == p._turn_policy.preemptive.preemptive_tts
    )


def test_streaming_pipeline_rejects_half_duplex_mode() -> None:
    factory = SharedStageFactory.__new__(SharedStageFactory)
    with pytest.raises(ValueError, match="full-duplex only"):
        StreamingPipeline(factory, interaction_mode=INTERACTION_MODE_HALF_DUPLEX)
