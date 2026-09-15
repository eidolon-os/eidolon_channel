"""Exercise audio-only welcomes through the real SDK and production agents."""

import asyncio
from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.contracts import SESSION_INTENT_PROACTIVE
from eidolon_sdk.biz.presentation import OutputSelection

from eidolon.livekit.agent.half_duplex.pipeline import HalfDuplexPttPipeline
from eidolon.livekit.agent.session.welcome import _audio_frames
from eidolon.livekit.common.welcome import WelcomeAudio, prepare_welcome_audio
from eidolon.livekit.tests._harness.headless import headless_session
from eidolon.livekit.tests._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD
from eidolon.livekit.tests._harness.production import production_session


async def _assert_sound_only(handle, tts):
    await handle.audio_out.wait_for_first_audio()
    await handle.events.wait_for(
        lambda e: e.type == "agent_state_changed"
        and e.payload.old_state == "speaking" and e.payload.new_state == "listening",
    )
    assert tts.synth_count == 0
    assert handle.events.agent_messages() == []
    pcm = prepare_welcome_audio(WelcomeAudio(), sample_rate=16000)
    assert handle.audio_out.collected_pcm == pcm


@pytest.mark.parametrize("mode", ["full_duplex", "half_duplex"])
async def test_streaming_welcome_audio_bypasses_tts_and_text(mode):
    tts = MockTTS.errors_with(AssertionError("welcome must not synthesize"))
    async with production_session(
        mode=mode, welcome=WelcomeAudio(),
        llm=MockLLM.scripted([]), stt=MockSTT.scripted([]), tts=tts, vad=MockVAD.silent(),
    ) as (pipeline, handle):
        await _assert_sound_only(handle, tts)
        assert pipeline._assistant_speech.latest is None


async def test_ptt_welcome_audio_bypasses_tts_and_text():
    llm, stt, tts, vad = (
        MockLLM.scripted([]), MockSTT.scripted([]),
        MockTTS.errors_with(AssertionError("welcome must not synthesize")), MockVAD.silent(),
    )
    pipeline = HalfDuplexPttPipeline(SimpleNamespace(
        outputs=OutputSelection(speech=True, dialogue_text=True),
        llm=SimpleNamespace(llm=llm), tts=SimpleNamespace(tts=tts), stt=stt, vad=None,
        interrupt_classifier=None,
    ), welcome_message=WelcomeAudio())
    try:
        async with headless_session(
            llm=llm, stt=stt, tts=tts, vad=vad, agent=pipeline._build_agent(),
            configure_session=pipeline._bind_session, strict_cleanup=True,
            extra_session_kwargs={"turn_handling": pipeline._build_turn_handling()},
        ) as handle:
            await _assert_sound_only(handle, tts)
    finally:
        await pipeline.shutdown()


@pytest.mark.parametrize("intent,speech", [(SESSION_INTENT_PROACTIVE, True), ("user_initiated", False)])
def test_sound_respects_proactive_and_silent_output_policy(intent, speech):
    from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline

    for pipeline_cls in (StreamingPipeline, HalfDuplexPttPipeline):
        pipeline = pipeline_cls.__new__(pipeline_cls)
        pipeline._factory = SimpleNamespace(outputs=OutputSelection(speech=speech, dialogue_text=True))
        pipeline._session_intent = intent
        pipeline._welcome_message = WelcomeAudio()
        assert pipeline._welcome_on_enter() is None


async def test_concurrent_playbacks_have_independent_frames_and_cursors():
    pcm = prepare_welcome_audio(WelcomeAudio(), sample_rate=16000)
    interrupted = _audio_frames(pcm, 16000)
    first = await anext(interrupted)
    first.data[0] = 123  # A mixer may mutate a frame; cached PCM must stay pristine.
    await interrupted.aclose()

    async def collect():
        return b"".join([bytes(frame.data) async for frame in _audio_frames(pcm, 16000)])

    assert await asyncio.gather(collect(), collect()) == [pcm, pcm]
