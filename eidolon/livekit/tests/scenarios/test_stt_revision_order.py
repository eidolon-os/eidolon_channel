"""Provider revision ordering through the public STT node and product turn gate."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.common.transcript_evidence import TranscriptEvidence
from .._harness.audio import synth_voiced, synth_silence
from .._harness.latency import ReplyLatencyProbe
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
async def test_delayed_old_final_cannot_replace_corrected_destination(mode):
    newest = '明天去上海。'
    stale = '明天去北京。'
    scripts = [ScriptedTranscript(
        text=text, trigger_after_ms=at,
        metadata=TranscriptEvidence(stream_key='stream-1', revision_key='sentence-1', sequence=seq).as_metadata(),
    ) for text, at, seq in [(newest, 250, 2), (stale, 400, 1)]]
    async with production_session(
        mode=mode,
        llm=MockLLM.scripted([(newest, '收到，目的地是上海。')]),
        stt=MockSTT.scripted(scripts), tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted([MockVADEvent('start', 10), MockVADEvent('end', 650, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(.7))
        await h.events.wait_for_agent_messages(1, timeout=8)
        assert h.events.user_messages() == [newest]
        assert h.events.agent_messages() == ['收到，目的地是上海。']
        assert pipeline._latest_asr_text == newest


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
@pytest.mark.parametrize('first,interim,final', [
    ('帮我介绍一下这个方案。', '帮我介绍一下这个方案的优点', '帮我介绍一下这个方案的优点和缺点。'),
    ('Book a table.', 'Book a table for seven', 'Book a table for seven people.'),
    ('号码是1234。', '号码是123456', '号码是12345678。'),
])
async def test_short_framework_boundary_waits_for_growing_transcript(
    mode, first, interim, final, record_property,
):
    """A positive old prediction cannot certify a longer, still-changing input.

    Recognition can finalize an early fragment and then supply a growing
    repetition/extension after VAD end but before the SDK's endpoint delay ends.
    Force the old positive score to isolate commit coverage from model accuracy.
    """
    async with production_session(
        mode=mode, llm=MockLLM.scripted([('', '收到。')]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=first, trigger_after_ms=1200),
            ScriptedTranscript(text=interim, interims=[interim], trigger_after_ms=1600, final=False),
            ScriptedTranscript(text=final, trigger_after_ms=2200),
        ]),
        tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1500, .1)]),
        real_time_audio=True,
    ) as (pipeline, h):
        detector = pipeline._get_eot_model()
        original = detector.predict_end_of_turn
        detector.predict_end_of_turn = AsyncMock(return_value=.99)
        probe = ReplyLatencyProbe(pipeline, h)
        pcm = synth_silence(.35) + synth_voiced(1.15) + synth_silence(1.4)
        try:
            h.audio_in.feed_pcm(pcm)
            await h.events.wait_for(lambda e: e.type == 'user_input_transcribed'
                and e.payload.transcript == interim, timeout=4)
            # Give the old SDK endpoint enough time to fire, but withhold final.
            await asyncio.sleep(.3)
            record_property('premature_users', h.events.user_messages())
            record_property('premature_audio_bytes', h.audio_out.captured_bytes)
            assert not h.events.user_messages(), 'old shorter boundary committed the growing input'
            assert h.audio_out.captured_bytes == 0
            await h.audio_in.wait_drained(timeout=5)
            await h.events.wait_for_agent_messages(1, timeout=3)
            expected = f'{first} {final}' if first.isascii() else first + final
            assert h.events.user_messages() == [expected]
            assert h.events.agent_messages() == ['收到。']
            report = probe.report(pcm)
            record_property('latency', json.dumps(report, ensure_ascii=False))
            assert 0 <= report['stop_to_reply_audio_ms'] <= 1200
            final_event = next(e for e in h.events.of_type('user_input_transcribed')
                if e.payload.transcript == final and e.payload.is_final)
            user = next(e for e in h.events.of_type('conversation_item_added')
                if getattr(e.payload.item, 'role', '') == 'user')
            final_to_commit = user.timestamp - final_event.timestamp
            record_property('final_to_commit_ms', round(final_to_commit * 1000, 1))
            assert 0 <= final_to_commit < .3
        finally:
            probe.close()
            detector.predict_end_of_turn = original
