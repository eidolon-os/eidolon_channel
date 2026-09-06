"""Reply latency and premature-reply guards across streaming modes.

These budgets constrain channel overhead, not real provider or device P95.
The slow/paused/revised speech suite separately constrains premature replies.
"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.agent.turn_policy import InterruptIntent, InterruptIntentResult
from eidolon.livekit.common.config import TurnPolicyConfig
from .._harness.audio import synth_silence, synth_voiced
from .._harness.latency import ReplyLatencyProbe
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
@pytest.mark.parametrize('text', ['好', '7'])
async def test_short_answer_replies_promptly(text, mode, record_property):
    """Protect short answers when changing model inputs or endpoint policy."""
    reply = '收到。'
    async with production_session(
        mode=mode, llm=MockLLM.scripted([(text, reply)]),
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1000, .1)]),
        real_time_audio=True,
    ) as (pipeline, h):
        probe = ReplyLatencyProbe(pipeline, h)
        try:
            pcm = synth_silence(.35) + synth_voiced(.65) + synth_silence(.8)
            h.audio_in.feed_pcm(pcm)
            await h.events.wait_for_agent_messages(1, timeout=6)
            report = probe.report(pcm)
            record_property('text', text)
            record_property('mode', mode)
            record_property('stop_to_reply_audio_ms', report['stop_to_reply_audio_ms'])
            record_property('eot_predictions', report['eot_predictions'])
            assert h.events.user_messages() == [text]
            assert h.events.agent_messages() == [reply]
            assert 0 <= report['stop_to_reply_audio_ms'] <= 1000
        finally:
            probe.close()


@pytest.mark.asyncio
# Include the scripted 500 ms VAD delay and scheduling/audio-buffer headroom.
# These are regression ceilings, not promises for real network providers.
@pytest.mark.parametrize('overlap,intent_delay,budget_ms', [(False, 0, 1000), (True, .25, 1000), (True, .7, 1500)])
async def test_complete_turn_does_not_add_a_second_endpoint_wait(overlap, intent_delay, budget_ms, record_property):
    async def classify(*args, **kwargs):
        await asyncio.sleep(intent_delay)
        return InterruptIntentResult(InterruptIntent.NORMAL_INTERRUPT, 0, 'fault_injection', '')

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    text = '帮我详细介绍一下这个方案。'
    async with production_session(
        welcome='我先介绍一些背景，后面还有几个细节需要慢慢说明。' if overlap else '',
        llm=MockLLM.scripted([(text, '现在介绍这个方案。')]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text='帮我详细', interims=['帮我详细'], trigger_after_ms=650, final=False),
            ScriptedTranscript(text=text, trigger_after_ms=1100),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1500, .1)]),
        turn_policy=policy, interrupt_classifier=classifier, real_time_audio=True,
    ) as (pipeline, h):
        probe = ReplyLatencyProbe(pipeline, h)
        if overlap:
            await h.audio_out.wait_for_first_audio()
            old_speech = h.session.current_speech
            assert old_speech is not None
        pcm = synth_silence(.35) + synth_voiced(.65) + synth_silence(.8)
        h.audio_in.feed_pcm(pcm)
        await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
            and getattr(e.payload.item, 'text_content', '') == '现在介绍这个方案。', timeout=6)
        report = probe.report(pcm)
        probe.close()
        record_property('stop_to_reply_audio_ms', report['stop_to_reply_audio_ms'])
        record_property('complete_to_reply_audio_ms', report['complete_to_reply_audio_ms'])
        record_property('budget_ms', budget_ms)
        assert report['complete_to_reply_audio_ms'] is not None, 'this gate requires positive EOT evidence'
        assert 0 <= report['stop_to_reply_audio_ms'] <= budget_ms
        assert h.events.user_messages() == [text]
        assert classifier.classify.await_count == int(overlap)
        if overlap:
            assert old_speech.interrupted


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,intent_provider', [
    ('full_duplex', 'none'), ('full_duplex', 'llm'), ('half_duplex', 'none'),
])
@pytest.mark.parametrize('prefix,continuation', [
    ('我想问的是', '明天的天气怎么样。'),
    pytest.param(
        '如果预算不够的话', '就选便宜的。',
        marks=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason='learned EOT treats an unfinished conditional clause as complete and replies during the pause',
        ),
    ),
])
async def test_incomplete_turn_survives_long_pause_then_replies_promptly(
    mode, intent_provider, prefix, continuation, record_property,
):
    """A 2.2 s pause must not split one request, even if EOT misreads its prefix."""
    classifier = SimpleNamespace(classify=AsyncMock())
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider=intent_provider))
    reply = '现在回应你的问题。'
    async with production_session(
        mode=mode,
        llm=MockLLM.scripted([('', reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=continuation, trigger_after_ms=3550),
        ]),
        tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1000, .1),
            MockVADEvent('start', 3200), MockVADEvent('end', 3900, .1),
        ]),
        turn_policy=policy, interrupt_classifier=classifier, real_time_audio=True,
    ) as (pipeline, h):
        prefix_score = pipeline._get_eot_model()._context_eot.semantic_completeness_score(prefix)
        record_property('prefix_eot_score', prefix_score)
        probe = ReplyLatencyProbe(pipeline, h)
        try:
            pcm = (synth_silence(.35) + synth_voiced(.65) + synth_silence(2.2)
                + synth_voiced(.7) + synth_silence(.8))
            h.audio_in.feed_pcm(pcm)
            # Observe the actual second speech-start event, rather than racing
            # a wall-clock snapshot against the scheduled continuation.
            await h.events.wait_for(
                lambda e: e.type == 'user_state_changed' and e.payload.new_state == 'speaking'
                and sum(s.payload.new_state == 'speaking'
                    for s in h.events.of_type('user_state_changed')) == 2,
                timeout=5,
            )
            record_property('premature_commits', len(h.events.user_messages()))
            record_property('premature_audio_bytes', h.audio_out.captured_bytes)
            assert h.events.user_messages() == [], 'incomplete prefix was committed before continuation'
            assert h.audio_out.captured_bytes == 0, 'reply began during the within-question pause'
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == reply, timeout=5)
            report = probe.report(pcm)
            record_property('stop_to_reply_audio_ms', report['stop_to_reply_audio_ms'])
            # ASR finals are two fragments; the SDK scores the combined turn.
            record_property('sdk_complete_to_reply_audio_ms', report['sdk_complete_to_reply_audio_ms'])
            assert report['sdk_complete_to_reply_audio_ms'] is not None
            assert 0 <= report['stop_to_reply_audio_ms'] <= 1000
            assert h.events.user_messages() == [f'{prefix} {continuation}']
            assert h.events.agent_messages() == [reply]
            classifier.classify.assert_not_awaited()
        finally:
            probe.close()
