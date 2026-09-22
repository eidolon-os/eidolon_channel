from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from eidolon_sdk.biz.presentation import OutputSelection, SessionOutputPlan
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.shared.pipeline import BasePipeline
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.common.config.schema import EffectiveAgentConfig, ProvidersConfig


def silent_plan():
    return SessionOutputPlan(
        session_id="session-1",
        policy_revision=1,
        outputs=OutputSelection(expression=True),
        expression_profile="eidolon.face.v1",
    )


@pytest.mark.parametrize("cue", [False, True])
def test_production_factory_never_constructs_tts_for_silent_session(monkeypatch, cue):
    tts_builder = Mock(side_effect=AssertionError("TTS constructor reached"))
    monkeypatch.setattr(SharedStageFactory, "_build_llm", lambda _: object())
    monkeypatch.setattr(SharedStageFactory, "_build_stt", lambda _: object())
    monkeypatch.setattr(SharedStageFactory, "_build_vad", lambda _: None)
    monkeypatch.setattr(SharedStageFactory, "_build_tts", tts_builder)
    monkeypatch.setattr(SharedStageFactory, "build_interrupt_classifier", lambda _: None)
    cfg = replace(
        EffectiveAgentConfig(),
        providers=ProvidersConfig(tts_provider=None, brain_provider="eidolon_agent"),
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.factory._build_runtime_services",
        lambda _: SimpleNamespace(resolve_room=AsyncMock()),
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.factory._build_device_token_source", lambda **_: lambda: "test"
    )
    monkeypatch.setattr(
        "eidolon.livekit.agent.eidolon_agent_rpc.EidolonAgentGrpcLlm", lambda **_: object()
    )
    factory = SharedStageFactory.from_config(
        cfg, runtime_session_id="session-1", output_plan=silent_plan().model_copy(
            update={"outputs": OutputSelection(expression=True, audio_cue=cue)})
    )
    assert factory.tts is None
    assert factory.outputs.expression and not factory.outputs.speech
    tts_builder.assert_not_called()


@pytest.mark.parametrize(
    "tts,plan,session",
    [
        (object(), silent_plan(), "session-1"),
        (None, None, "session-1"),
        (None, silent_plan(), "session-other"),
    ],
)
def test_stage_and_session_mismatch_are_rejected(tts, plan, session):
    with pytest.raises(ValueError):
        SharedStageFactory(
            llm=object(), stt=object(), tts=tts, runtime_session_id=session, output_plan=plan
        )


@pytest.mark.asyncio
async def test_silent_warmup_only_visits_input_stages():
    stt = SimpleNamespace(warmup=AsyncMock())
    factory = SharedStageFactory(
        llm=object(), stt=stt, tts=None, runtime_session_id="session-1", output_plan=silent_plan()
    )

    class InputOnlyPipeline(BasePipeline):
        async def run(self, room):
            pass

    pipeline = InputOnlyPipeline.__new__(InputOnlyPipeline)
    pipeline._factory = factory
    assert pipeline._lifecycle_stages() == [stt]
    await pipeline._warmup_stages()
    stt.warmup.assert_awaited_once()


def test_silent_session_suppresses_fixed_welcome_for_every_turn_mode():
    from eidolon.livekit.agent.half_duplex import HalfDuplexPttPipeline

    for kind in (StreamingPipeline, HalfDuplexPttPipeline):
        pipeline = kind.__new__(kind)
        pipeline._factory = SimpleNamespace(outputs=silent_plan().outputs)
        pipeline._session_intent = "user_initiated"
        pipeline._welcome_message = "Never synthesize me"
        assert pipeline._welcome_on_enter() is None


@pytest.mark.asyncio
async def test_silent_agent_has_no_transcription_node_side_effects():
    from eidolon.livekit.agent.session.policy_bound_agent import PolicyBoundAgent

    agent = PolicyBoundAgent(instructions="", outputs=silent_plan().outputs)

    async def text():
        raise AssertionError("forbidden transcript must not be consumed or published")
        yield "private answer"

    assert await agent.transcription_node(text(), {}) is None


def test_dispatch_output_plan_cannot_be_bound_to_another_session():
    import json
    from eidolon.livekit.agent.server import _resolve_output_plan

    ctx = SimpleNamespace(
        job=SimpleNamespace(metadata=json.dumps({"output_plan": silent_plan().model_dump()}))
    )
    assert _resolve_output_plan(ctx, "session-1") == silent_plan()
    with pytest.raises(ValueError, match="OUTPUT_PLAN_SESSION_MISMATCH"):
        _resolve_output_plan(ctx, "session-2")


def test_silent_session_opens_no_audio_output_track():
    """The speech selection gates the track, not just what is written to it.

    A silent Companion that still opened an audio output would publish a track
    it is not permitted to speak on, so assert the room options themselves.
    """

    from eidolon.livekit.agent.half_duplex import HalfDuplexPttPipeline

    pipeline = HalfDuplexPttPipeline.__new__(HalfDuplexPttPipeline)
    pipeline._audio_sample_rate = 16000

    pipeline._factory = SimpleNamespace(outputs=silent_plan().outputs)
    silent = pipeline._build_room_options()
    assert silent.audio_output is False
    assert silent.text_output is False

    pipeline._factory = SimpleNamespace(
        outputs=OutputSelection(speech=True, dialogue_text=True)
    )
    speaking = pipeline._build_room_options()
    assert speaking.audio_output is not False
    assert speaking.audio_output.sample_rate == 16000
    assert speaking.text_output is True

    # PTT hands captured audio in itself; neither selection opens a room input.
    assert silent.audio_input is False and speaking.audio_input is False


@pytest.mark.parametrize('mask', range(64))
def test_every_input_output_combination_builds_only_selected_model_stages(monkeypatch, mask):
    from eidolon_sdk.biz.presentation import InputSelection, FACE_PROFILE
    outputs = OutputSelection(**{name: bool(mask & (1 << index))
        for index, name in enumerate(OutputSelection.model_fields)})
    microphone = bool(mask & 32)
    plan = SessionOutputPlan(session_id='matrix', policy_revision=1, outputs=outputs,
        inputs=InputSelection(microphone=microphone),
        expression_profile=FACE_PROFILE if outputs.expression else None)
    stt, tts, vad, interrupt = [Mock(return_value=object()) for _ in range(4)]
    monkeypatch.setattr(SharedStageFactory, '_build_stt', stt)
    monkeypatch.setattr(SharedStageFactory, '_build_tts', tts)
    monkeypatch.setattr(SharedStageFactory, '_build_vad', vad)
    monkeypatch.setattr(SharedStageFactory, 'build_interrupt_classifier', interrupt)
    monkeypatch.setattr('eidolon.livekit.agent.factory._build_runtime_services',
        lambda _: SimpleNamespace(resolve_room=AsyncMock()))
    monkeypatch.setattr('eidolon.livekit.agent.factory._build_device_token_source', lambda **_: lambda: 'test')
    monkeypatch.setattr('eidolon.livekit.agent.eidolon_agent_rpc.EidolonAgentGrpcLlm', lambda **_: object())
    cfg = replace(EffectiveAgentConfig(), providers=ProvidersConfig(brain_provider='eidolon_agent'))
    factory = SharedStageFactory.from_config(cfg, runtime_session_id='matrix', output_plan=plan)
    assert stt.call_count == int(microphone)
    assert vad.call_count == int(microphone)
    assert interrupt.call_count == int(microphone)
    assert tts.call_count == int(outputs.speech)
    assert (factory.stt is not None) == microphone
    assert (factory.tts is not None) == outputs.speech
    assert factory.outputs == outputs


def test_input_disabled_manual_pipeline_keeps_text_and_speech_without_ptt_capture():
    from eidolon_sdk.biz.presentation import InputSelection
    from eidolon.livekit.agent.half_duplex import HalfDuplexPttPipeline
    plan = SessionOutputPlan(session_id='text', policy_revision=1,
        inputs=InputSelection(microphone=False), outputs=OutputSelection(speech=True, dialogue_text=True))
    factory = SharedStageFactory(llm=object(), stt=None, tts=object(),
        output_plan=plan, runtime_session_id='text')
    pipeline = HalfDuplexPttPipeline(factory)
    assert pipeline._ptt_controller is None
    assert pipeline._lifecycle_stages() == [factory.tts]
    pipeline._maybe_start_audio_stream(object(), object())
    assert not pipeline._track_tasks
    options = pipeline._build_room_options()
    assert options.audio_input is False
    assert options.audio_output is not False
    assert options.text_output is True


@pytest.mark.asyncio
@pytest.mark.parametrize('display', [False, True])
@pytest.mark.parametrize('speech', [False, True])
async def test_text_turn_completes_with_independent_text_and_speech_in_real_agent_session(display, speech):
    import asyncio
    from livekit.agents.voice import AgentSession
    from livekit.agents.voice.io import TextOutput
    from eidolon.livekit.agent.session.policy_bound_agent import PolicyBoundAgent
    from .._harness.mocks.mock_llm import MockLLM
    from .._harness.mocks.mock_tts import MockTTS
    from .._harness.headless import RecordingAudioOutput

    class CaptureText(TextOutput):
        def __init__(self):
            super().__init__(label='test', next_in_chain=None)
            self.parts = []
        async def capture_text(self, text):
            self.parts.append(text)
        def flush(self):
            pass

    sink = CaptureText()
    session = AgentSession(turn_handling={'turn_detection': 'manual', 'interruption': {'enabled': False}})
    session.output.transcription = sink
    audio = RecordingAudioOutput()
    if speech:
        session.output.audio = audio
    agent = PolicyBoundAgent(instructions='', llm=MockLLM.echo(),
        outputs=OutputSelection(dialogue_text=display, speech=speech), stt=None, tts=MockTTS() if speech else None)
    try:
        await session.start(agent)
        handle = session.generate_reply(user_input='你好')
        await asyncio.wait_for(handle.wait_for_playout(), timeout=3)
        assert (''.join(sink.parts) == '你好') if display else not sink.parts, sink.parts
        assert bool(audio.collected_pcm) == speech
    finally:
        await session.aclose()
