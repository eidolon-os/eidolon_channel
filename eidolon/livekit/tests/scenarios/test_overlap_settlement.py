"""Order-independent settlement through the production Agent/SDK callback chain."""
import json

import pytest

from .._harness.audio import synth_voiced
from .._harness.headless import wait_until
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest.mark.asyncio
@pytest.mark.parametrize('final_at_ms', [700, 1050, 1200], ids=['final_before_stop', 'same_time', 'final_after_stop'])
@pytest.mark.parametrize('repeated_interim', [False, True], ids=['final_only', 'repeated_interim'])
async def test_final_and_silence_resume_same_speech(final_at_ms, repeated_interim, record_property):
    """A rejected backchannel resumes the same speech, whatever the event order.

    Playback recovers provisionally once the continuation grace after VAD stop
    expires; the candidate keeps its own evidence deadline and later expires
    without cancelling the reply or opening a user turn.
    """
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
        orchestrator = pipeline._interruption_orchestrator
        h.audio_in.feed_pcm(synth_voiced(1.3))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        await wait_until(lambda: orchestrator.state.value == 'resumed_waiting_evidence', timeout=5)
        assert pipeline._ducking.mixer.state == 'NORMAL'
        assert orchestrator.active, 'resume is provisional; the candidate still owns its evidence deadline'
        assert speech is not None and not speech.interrupted

        # The audio hold is bounded by the continuation grace after VAD stop,
        # not by the candidate's longer evidence budget.
        recovery = next(event for event in pipeline._timeline.attrs['duck_events']
                        if event['event'] == 'output_resumed_pending_evidence')
        assert recovery['user_speaking'] is False
        stop_to_resume_ms = (recovery['at'] - recovery['speech_stopped_at']) * 1000.0
        record_property('output_recovery', json.dumps(recovery))
        record_property('stop_to_resume_ms', stop_to_resume_ms)
        assert 0 <= stop_to_resume_ms <= pipeline._turn_policy.eot.speech_merge_grace_ms + 200

        # The candidate then expires on its own budget: no cancel, no user turn.
        await wait_until(lambda: not orchestrator.active, timeout=8)
        verdict = pipeline._timeline.attrs['interruption_verdict']
        record_property('interruption_verdict', json.dumps(verdict, ensure_ascii=False))
        assert verdict['action'] in ('rejected_resume', 'expired_resume')
        assert verdict['continue_to_llm'] is False

        await h.events.wait_for_agent_messages(1, timeout=5)
        assert h.events.agent_messages() == [welcome]
        assert h.events.user_messages() == []
        assert not speech.interrupted
        assert not any(s.cleared for s in h.audio_out.segments)
