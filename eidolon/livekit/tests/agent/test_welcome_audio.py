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
    if tts is not None:
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


@pytest.mark.parametrize("has_tts", [False, True])
async def test_ptt_welcome_audio_bypasses_tts_and_text(has_tts):
    llm, stt, tts, vad = (
        MockLLM.scripted([]), MockSTT.scripted([]),
        MockTTS.errors_with(AssertionError("welcome must not synthesize")), MockVAD.silent(),
    )
    if not has_tts:
        tts = None
    pipeline = HalfDuplexPttPipeline(SimpleNamespace(
        outputs=OutputSelection(speech=has_tts, dialogue_text=True, audio_cue=True),
        llm=SimpleNamespace(llm=llm), tts=SimpleNamespace(tts=tts) if has_tts else None, stt=stt, vad=None,
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


@pytest.mark.parametrize("speech", [False, True])
@pytest.mark.parametrize("cue", [False, True])
@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("intent", ["user_initiated", "presence_initiated", SESSION_INTENT_PROACTIVE])
def test_welcome_output_matrix_for_both_pipeline_types(speech, cue, audio, intent):
    from eidolon.livekit.agent.full_duplex.pipeline import StreamingPipeline
    from eidolon.livekit.agent.session.welcome import play_welcome
    from unittest.mock import Mock
    outputs = OutputSelection(speech=speech, audio_cue=cue, dialogue_text=True)
    welcome = WelcomeAudio() if audio else "Hello"
    for pipeline_cls in (StreamingPipeline, HalfDuplexPttPipeline):
        pipeline = pipeline_cls.__new__(pipeline_cls)
        pipeline._factory = SimpleNamespace(outputs=outputs)
        pipeline._session_intent = intent
        pipeline._welcome_message = welcome
        allowed = (cue if audio else speech) and intent != SESSION_INTENT_PROACTIVE
        assert pipeline._welcome_on_enter() == (welcome if allowed else None)
    # The last playback boundary must also enforce selection before consuming
    # PCM or publishing the greeting to the assistant text ledger.
    session, queue = Mock(), Mock()
    if not (cue if audio else speech):
        play_welcome(session, welcome, outputs=outputs, pcm=None, sample_rate=16000, queue_text=queue)
        session.say.assert_not_called()
        queue.assert_not_called()


@pytest.mark.parametrize("mode", ["full_duplex", "half_duplex"])
async def test_cue_only_plays_without_a_tts_stage(mode):
    async with production_session(
        mode=mode, welcome=WelcomeAudio(),
        outputs=OutputSelection(dialogue_text=True, audio_cue=True),
        llm=MockLLM.scripted([]), stt=MockSTT.scripted([]), tts=None, vad=MockVAD.silent(),
    ) as (pipeline, handle):
        await handle.audio_out.wait_for_first_audio()
        await handle.events.wait_for(lambda e: e.type == "agent_state_changed"
            and e.payload.old_state == "speaking" and e.payload.new_state == "listening")
        assert pipeline._factory.tts is None
        assert handle.events.agent_messages() == []
        assert handle.audio_out.collected_pcm == prepare_welcome_audio(WelcomeAudio(), sample_rate=16000)


def test_ptt_audio_track_depends_on_any_selected_audio_not_tts():
    pipeline = HalfDuplexPttPipeline.__new__(HalfDuplexPttPipeline)
    pipeline._audio_sample_rate = 16000
    for speech in (False, True):
        for cue in (False, True):
            pipeline._factory = SimpleNamespace(outputs=OutputSelection(speech=speech, audio_cue=cue))
            assert (pipeline._build_room_options().audio_output is not False) == (speech or cue)
