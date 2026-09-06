"""Order-independent settlement through the production Agent/SDK callback chain."""
import asyncio

import pytest

from .._harness.audio import synth_voiced
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest.mark.asyncio
@pytest.mark.parametrize('final_at_ms', [700, 1050, 1200], ids=['final_before_stop', 'same_time', 'final_after_stop'])
@pytest.mark.parametrize('repeated_interim', [False, True], ids=['final_only', 'repeated_interim'])
async def test_final_and_silence_resume_same_speech(final_at_ms, repeated_interim):
    welcome = '我先介绍背景，再慢慢说明各个细节，最后还会举一个例子。'
    scripts = []
    if repeated_interim:
        scripts.append(ScriptedTranscript(
            text='嗯嗯你继续', trigger_after_ms=450,
            interims=['嗯嗯你继续', '嗯嗯你继续'], interim_gap_ms=80, final=False,
        ))
    scripts.append(ScriptedTranscript(text='嗯嗯你继续', trigger_after_ms=final_at_ms))
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([]),
        stt=MockSTT.scripted(scripts),
        tts=MockTTS(char_seconds=.08, chunk_delay_ms=30),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.3))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        await asyncio.sleep(1.5)
        assert pipeline._ducking.mixer.state == 'NORMAL'
        assert not pipeline._interruption_orchestrator.active
        assert speech is not None and not speech.interrupted
        await h.events.wait_for_agent_messages(1, timeout=5)
        assert h.events.agent_messages() == [welcome]
        assert h.events.user_messages() == []
        assert not any(s.cleared for s in h.audio_out.segments)
