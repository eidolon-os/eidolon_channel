"""StreamingPipeline construction regression tests."""

from __future__ import annotations

from types import SimpleNamespace


def test_streaming_pipeline_initializes_ducking_before_effect_handlers() -> None:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline
    from eidolon.livekit.agent.output import OutputDuckingController

    factory = SimpleNamespace(
        stt=SimpleNamespace(stt=SimpleNamespace()),
        tts=SimpleNamespace(tts=SimpleNamespace()),
        llm=SimpleNamespace(llm=SimpleNamespace()),
        vad=None,
    )

    pipeline = StreamingPipeline(factory, instructions="test")

    assert isinstance(pipeline._ducking, OutputDuckingController)
    assert pipeline._agent_state_effects._ducking is pipeline._ducking
