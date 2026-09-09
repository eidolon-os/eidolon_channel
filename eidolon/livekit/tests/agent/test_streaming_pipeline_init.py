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


def test_close_stt_utterance_reaches_the_plugin_through_the_stage() -> None:
    """The path from pipeline to plugin, which is easy to get silently wrong.

    ``_close_stt_utterance`` walks ``_factory.stt.stt.end_utterance``. Every
    step is a ``getattr`` that returns None rather than raising, so a wrong
    attribute name anywhere along it means the boundary is never sent and
    the only symptom is a device that answers with its fallback line — the
    failure this call exists to end. So the walk is pinned.
    """

    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    calls: list[int] = []

    def end_utterance() -> bool:
        calls.append(1)
        return True

    owner = SimpleNamespace(
        _factory=SimpleNamespace(
            stt=SimpleNamespace(stt=SimpleNamespace(end_utterance=end_utterance)),
        ),
    )

    assert StreamingPipeline._close_stt_utterance(owner) is True
    assert calls == [1]


def test_close_stt_utterance_is_silent_on_a_plugin_that_owns_its_boundaries() -> None:
    """A cloud recognizer does not implement it, and that is not a failure."""

    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    owner = SimpleNamespace(
        _factory=SimpleNamespace(stt=SimpleNamespace(stt=SimpleNamespace())),
    )
    assert StreamingPipeline._close_stt_utterance(owner) is False

    # And before the factory has an STT stage at all.
    assert StreamingPipeline._close_stt_utterance(SimpleNamespace()) is False
