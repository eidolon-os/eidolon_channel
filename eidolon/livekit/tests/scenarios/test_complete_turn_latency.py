"""Reply latency and premature-reply guards across streaming modes.

These budgets constrain channel overhead, not real provider or device P95.
The slow/paused/revised speech suite separately constrains premature replies.
"""
import asyncio
import json
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
@pytest.mark.parametrize('score', [.424, .95], ids=['uncertain', 'complete'])
@pytest.mark.parametrize('vad_silence', [.2, .5])
async def test_endpointing_preserves_complete_text_after_interruption(
    score, vad_silence, monkeypatch, record_property,
):
    """Protect complete text and the fast path; observe uncertain-turn timing.

    Scores are injected at the Channel model boundary. A complete result must
    take the fast path. Uncertain timing is recorded for diagnosis, not fixed
    as a product requirement; paused-turn tests guard premature completion.
    """
    text, reply = '请调整一下参加活动的人数。', '已收到修改。'
    async with production_session(
        welcome='我先介绍原来的安排，后面还有几项细节需要继续说明。',
        llm=MockLLM.scripted([('', reply)]),
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350),
            MockVADEvent('end', 1000, .1, silence_duration=vad_silence),
        ]),
    ) as (pipeline, h):
        monkeypatch.setattr(pipeline._get_eot_model()._context_eot,
            'semantic_completeness_score', lambda text: score)
        h.audio_in.feed_pcm(synth_voiced(1.2))
        await h.audio_out.wait_for_first_audio()
        old_speech = h.session.current_speech
        await h.events.wait_for(lambda e: e.type == 'user_input_transcribed'
            and e.payload.transcript == text and e.payload.is_final, timeout=3)
        timeline = pipeline._timeline
        await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
            and getattr(e.payload.item, 'text_content', '') == reply, timeout=7)
        marks = timeline.timestamps
        stop_to_commit = (marks['turn_committed_at'] - marks['speech_stopped_at']) * 1000
        cancel_to_commit = (marks['turn_committed_at'] - marks['interrupt_cancel_resolved_at']) * 1000
        record_property('stop_to_commit_ms', stop_to_commit)
        record_property('cancel_to_commit_ms', cancel_to_commit)
        record_property('injected_eot_score', score)
        record_property('vad_silence_sec', vad_silence)
        assert old_speech.interrupted
        assert h.events.user_messages() == [text]
        if score >= .5:
            assert 0 <= stop_to_commit <= 500


@pytest.mark.asyncio
@pytest.mark.parametrize('prefix,tail', [
    ('如果参加人数改成两个人', '费用需要重新计算。'),
    ('如果把出发时间改到下午', '就不用安排午饭了。'),
])
async def test_confirmed_interrupt_does_not_finish_a_paused_user_turn(
    prefix, tail, monkeypatch, record_property,
):
    """A user may still continue after the old reply has been interrupted."""
    reply = '收到完整的修改要求。'
    llm = MockLLM.scripted([('', reply)])
    async with production_session(
        welcome='我先介绍原来的安排，接下来还有几项具体内容需要说明。',
        llm=llm,
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=tail, trigger_after_ms=3150),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1000, .1, silence_duration=.5),
            MockVADEvent('start', 2600), MockVADEvent('end', 3500, .1, silence_duration=.5),
        ]),
    ) as (pipeline, h):
        monkeypatch.setattr(pipeline._get_eot_model()._context_eot,
            'semantic_completeness_score', lambda text: .95 if tail in text else .424)
        h.audio_in.feed_pcm(synth_voiced(3.7))
        await h.audio_out.wait_for_first_audio()
        old_speech = h.session.current_speech
        await h.events.wait_for(lambda e: e.type == 'user_state_changed'
            and e.payload.new_state == 'speaking'
            and sum(s.payload.new_state == 'speaking'
                for s in h.events.of_type('user_state_changed')) == 2, timeout=5)
        record_property('premature_messages', json.dumps(h.events.user_messages(), ensure_ascii=False))
        record_property('old_speech_interrupted', old_speech.interrupted)
        assert old_speech.interrupted, 'test must reach an accepted interruption before continuation'
        assert not h.events.user_messages(), 'interruption acceptance must not commit the unfinished turn'
        assert llm.call_count == 0
        await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
            and getattr(e.payload.item, 'text_content', '') == reply, timeout=5)
        assert len(h.events.user_messages()) == 1
        assert prefix in h.events.user_messages()[0] and tail in h.events.user_messages()[0]
        assert llm.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('provider,prefix,tail', [
    ('llm', '不是。', '我刚才说错了。'),
    ('none', '我想改一下。', '不是明天是后天。'),
    ('none', '先别介绍背景。', '告诉我费用。'),
])
async def test_fragmented_correction_scores_whole_turn_without_extra_endpoint_wait(
    provider, prefix, tail, record_property,
):
    async def classify(*args, **kwargs):
        await asyncio.sleep(.7)
        return InterruptIntentResult(InterruptIntent.NORMAL_INTERRUPT, 0, 'fault_injection', '')

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider=provider))
    reply = '收到更正。'
    async with production_session(
        welcome='这里正在播放原本的回答，后面还有几个细节需要继续说明。',
        llm=MockLLM.scripted([(tail.rstrip('。'), reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=tail[:3], interims=[tail[:3]], final=False, trigger_after_ms=850),
            ScriptedTranscript(text=tail, trigger_after_ms=1490),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1450, .1, silence_duration=.5),
        ]),
        interrupt_classifier=classifier, turn_policy=policy, real_time_audio=True,
    ) as (pipeline, h):
        probe = ReplyLatencyProbe(pipeline, h)
        pcm = synth_silence(.35) + synth_voiced(.6) + synth_silence(4)
        try:
            h.audio_in.feed_pcm(pcm)
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == reply, timeout=7)
            report = probe.report(pcm)
            record_property('latency', json.dumps(report, ensure_ascii=False))
            record_property('intent_provider', provider)
            assert len(h.events.user_messages()) == 1
            assert prefix.rstrip('。') in h.events.user_messages()[0]
            assert tail.rstrip('。') in h.events.user_messages()[0]
            assert h.events.agent_messages()[-1] == reply
            if provider == 'none':
                classifier.classify.assert_not_awaited()
            assert 0 <= report['stop_to_reply_audio_ms'] <= 1500
        finally:
            probe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('final_at_ms', [2000, 4250])
async def test_cancelled_reply_waits_for_late_tail_without_second_endpoint_delay(
    final_at_ms, record_property,
):
    """Late ASR final before/after the SDK deadline must preserve one reply.

    The budget starts at actual final delivery, not at speech stop: provider
    delay is injected deliberately and must not be hidden as channel overhead.
    """
    prefix, tail, reply = '先别介绍背景。', '告诉我费用。', '费用取决于用量。'
    async with production_session(
        welcome='我先详细介绍一下方案背景，然后介绍实现步骤和各项细节。',
        llm=MockLLM.scripted([(tail.rstrip('。'), reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=tail, interims=[tail], final=False, trigger_after_ms=1550),
            ScriptedTranscript(text=tail, trigger_after_ms=final_at_ms),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1450, .1, silence_duration=.5),
        ]),
        real_time_audio=True,
    ) as (pipeline, h):
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        probe = ReplyLatencyProbe(pipeline, h)
        pcm = synth_silence(.35) + synth_voiced(.6) + synth_silence(4.5)
        try:
            h.audio_in.feed_pcm(pcm)
            await h.events.wait_for(lambda e: e.type == 'user_input_transcribed'
                and e.payload.transcript == tail and not e.payload.is_final, timeout=3)
            await asyncio.sleep(.15)
            assert speech.interrupted
            assert not h.events.user_messages(), 'pending tail must not be committed as final'
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == reply, timeout=5)
            assert len(h.events.user_messages()) == 1
            assert prefix.rstrip('。') in h.events.user_messages()[0]
            assert tail.rstrip('。') in h.events.user_messages()[0]
            assert h.events.agent_messages()[-1] == reply
            final_event = next(e for e in h.events.of_type('user_input_transcribed')
                if e.payload.transcript == tail and e.payload.is_final)
            user = next(e for e in h.events.of_type('conversation_item_added')
                if getattr(e.payload.item, 'role', '') == 'user')
            report = probe.report(pcm)
            final_to_audio = ((user.timestamp - final_event.timestamp) * 1000
                + report['commit_to_reply_audio_ms'])
            record_property('injected_final_at_ms', final_at_ms)
            record_property('final_to_reply_audio_ms', final_to_audio)
            record_property('latency', json.dumps(report, ensure_ascii=False))
            assert 0 <= final_to_audio <= 500
        finally:
            probe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('tail_final_ms', [3100, 3460], ids=['final_before_stop', 'final_after_stop'])
async def test_correction_across_merged_vad_segments_replies_once(tail_final_ms, record_property):
    """RTC regression: a superseded fragment must not reject the complete correction."""
    prefix, tail, reply = '我想改一下。', '不是明天，是后天。', '收到更正。'
    async with production_session(
        welcome='我先介绍一些背景，接下来会逐项说明时间、地点和需要准备的材料。',
        llm=MockLLM.scripted([(tail.rstrip('。'), reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text='我想改', interims=['我想改'], final=False, trigger_after_ms=800),
            ScriptedTranscript(text=prefix, trigger_after_ms=1780),
            ScriptedTranscript(text='不是明天', interims=['不是明天'], final=False, trigger_after_ms=2200),
            ScriptedTranscript(text=tail.rstrip('。'), interims=[tail.rstrip('。')], final=False, trigger_after_ms=2900),
            ScriptedTranscript(text=tail, trigger_after_ms=tail_final_ms),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1550, .1),
            MockVADEvent('start', 1700), MockVADEvent('end', 3300, .1, silence_duration=.5),
        ]),
        real_time_audio=True,
    ) as (pipeline, h):
        probe = ReplyLatencyProbe(pipeline, h)
        pcm = synth_silence(.35) + synth_voiced(2.45) + synth_silence(2)
        try:
            h.audio_in.feed_pcm(pcm)
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == reply, timeout=7)
            await asyncio.sleep(.5)
            assert len(h.events.user_messages()) == 1
            assert h.events.user_messages()[0].replace(' ', '') == prefix + tail
            assert h.events.agent_messages()[-1] == reply
            report = probe.report(pcm)
            record_property('latency', json.dumps(report, ensure_ascii=False))
            assert 0 <= report['stop_to_reply_audio_ms'] <= 1500
        finally:
            probe.close()


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
