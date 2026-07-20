"""AgentSession turn_handling config for the full-duplex StreamingPipeline."""

from __future__ import annotations

import pytest
from eidolon_sdk.biz.contracts import INTERACTION_MODE_HALF_DUPLEX

from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import TurnPolicyConfig


def _pipe(*, allow_interruptions: bool) -> StreamingPipeline:
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._allow_interruptions = allow_interruptions
    p._false_interruption_timeout = 6.0
    p._turn_policy = TurnPolicyConfig()
    p._avatar_enabled = False
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
