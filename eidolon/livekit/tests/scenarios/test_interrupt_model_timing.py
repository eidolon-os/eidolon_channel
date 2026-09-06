"""Fault injection at the model boundary; production SDK owns audio and replies."""
import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC, WIRE_SCHEMA_VERSION

from eidolon.livekit.agent.turn_policy import InterruptIntent, InterruptIntentResult
from eidolon.livekit.common.config import TurnPolicyConfig
from .._harness.audio import synth_voiced
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest.mark.asyncio
async def test_explicit_preempt_during_question_does_not_wait_for_old_model_request():
    started = asyncio.Event()
    release = asyncio.Event()

    async def classify(*args, **kwargs):
        started.set()
        await release.wait()
        return InterruptIntentResult(InterruptIntent.HARD_STOP, 0, 'fault_injection', '')

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    # Keep the old provider pending beyond the assertion deadline: completing
    # the new reply must be independent of this superseded request.
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm', intent_timeout_ms=10000))
    prefix = '关于这个方案，'
    continuation = '帮我详细介绍一下。'
    text = prefix + continuation
    reply = '现在介绍这个方案。'
    async with production_session(
        welcome='我正在说明原来的内容，还有几个细节需要慢慢介绍。',
        llm=MockLLM.scripted([(text, reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=continuation, trigger_after_ms=2300),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1200, .1),
            MockVADEvent('start', 1900), MockVADEvent('end', 2600, .1),
        ]),
        interrupt_classifier=classifier, turn_policy=policy,
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(2.7))
        await h.audio_out.wait_for_first_audio()
        old_speech = h.session.current_speech
        await asyncio.wait_for(started.wait(), timeout=3)
        old_task = pipeline._semantic_interrupts._intent_task
        packet = SimpleNamespace(
            topic=CLIENT_AUDIO_STATE_TOPIC,
            participant=SimpleNamespace(identity='human-simulator'),
            data=json.dumps({
                'schema_v': WIRE_SCHEMA_VERSION, 'type': 'client.audio_state',
                'seq': 1, 'input_mode': 'ptt', 'playback_state': 'agent_speaking',
                'mic_muted': False, 'ptt': True, 'rms': .1, 'client_ts_ms': 700,
            }).encode(),
        )
        try:
            pipeline._room_data.handle_packet(packet)
            pipeline._client_preempts.on_client_room_packet(packet)
            assert old_speech.interrupted
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == reply, timeout=4)
            assert old_task is not None and not old_task.done()
            assert h.events.user_messages() == [text]
            assert h.events.agent_messages()[-1] == reply
            user_event = await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'role', None) == 'user')
            assert any(s.first_audible_at is not None and s.first_audible_at >= user_event.timestamp
                and not s.cleared for s in h.audio_out.segments)
        finally:
            release.set()
            await old_task
        classifier.classify.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('delay', [.05, .7, 1.2])
@pytest.mark.parametrize('intent', [InterruptIntent.NORMAL_INTERRUPT, InterruptIntent.BACKCHANNEL, InterruptIntent.HARD_STOP])
async def test_model_result_on_either_side_of_sdk_endpoint(intent, delay):
    async def classify(*args, **kwargs):
        await asyncio.sleep(delay)
        return InterruptIntentResult(intent, 0.0, 'fault_injection', '')

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    welcome = '这里正在播放原本的回答，后面还有几个细节需要继续说明。'
    text = '测试中由外部意图证据决定这轮交互'
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([(text, '这是新的回复。')]),
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.1))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        if intent is InterruptIntent.NORMAL_INTERRUPT:
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and '这是新的回复。' in str(e.payload), timeout=6)
            assert h.events.user_messages() == [text]
            assert h.events.agent_messages()[-1] == '这是新的回复。'
            assert speech.interrupted
        elif intent is InterruptIntent.BACKCHANNEL:
            await h.events.wait_for_agent_messages(1, timeout=6)
            assert h.events.agent_messages() == [welcome]
            assert not speech.interrupted
            assert not any(s.cleared for s in h.audio_out.segments)
            assert not h.events.user_messages()
        else:
            await asyncio.sleep(2.5)
            assert speech.interrupted
            assert not h.events.user_messages()
        classifier.classify.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'error'])
async def test_model_unavailable_resumes_original_speech_after_vad_end(failure):
    async def classify(*args, **kwargs):
        if failure == 'error':
            raise ConnectionError('injected failure')
        await asyncio.sleep(10)

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm', intent_timeout_ms=150))
    welcome = '原来的回答仍然在这里播放，模型暂时不可用时需要保留这段内容。'
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([]),
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted([ScriptedTranscript(text='测试输入', trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.1))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        await asyncio.sleep(1.5)
        assert pipeline._ducking.mixer.state == 'NORMAL'
        assert not speech.interrupted
        await h.events.wait_for_agent_messages(1, timeout=6)
        assert h.events.agent_messages() == [welcome]
        assert not h.events.user_messages()
        assert not any(s.cleared for s in h.audio_out.segments)
        assert not pipeline._semantic_interrupts._intent_tasks


@pytest.mark.asyncio
async def test_resume_does_not_leak_overlap_into_next_user_turn():
    classifier = SimpleNamespace(classify=AsyncMock(return_value=InterruptIntentResult(
        InterruptIntent.BACKCHANNEL, 0.0, 'fault_injection', '',
    )))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    welcome = '我们先说明原来的方案，然后再讨论你的其他问题。'
    text = '现在请告诉我明天的天气'
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([(text, '现在回答天气问题。')]),
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted([
            ScriptedTranscript(text='好的你继续', trigger_after_ms=700),
            ScriptedTranscript(text=text, trigger_after_ms=5300),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1050, .1),
            MockVADEvent('start', 5000), MockVADEvent('end', 5700, .1),
        ]),
    ) as (_, h):
        h.audio_in.feed_pcm(synth_voiced(5.8))
        await h.events.wait_for_agent_messages(2, timeout=10)
        assert h.events.agent_messages() == [welcome, '现在回答天气问题。']
        assert h.events.user_messages() == [text]
        classifier.classify.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('profile_name', ['fast', 'slow', 'repeated_interims', 'within_turn_pause', 'late_final'])
async def test_overlap_delivery_waits_for_final_and_replies_once(profile_name):
    from .._harness.human_profiles import HUMAN_PROFILES

    profile = next(p for p in HUMAN_PROFILES if p.name == profile_name)
    classifier = SimpleNamespace(classify=AsyncMock(return_value=InterruptIntentResult(
        InterruptIntent.NORMAL_INTERRUPT, 0.0, 'fault_injection', '',
    )))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    start = 350
    scripts = [ScriptedTranscript(
        text=profile.text, trigger_after_ms=start + 100, interims=list(profile.interims),
        interim_gap_ms=profile.gap_ms, final=not profile.final_delay_ms,
    )]
    if profile.final_delay_ms:
        scripts.append(ScriptedTranscript(text=profile.text,
            trigger_after_ms=start + profile.duration_ms + profile.final_delay_ms))
    vad_events = [MockVADEvent('start', start)]
    for stop, resume in profile.pauses:
        vad_events.extend([MockVADEvent('end', start + stop, .1), MockVADEvent('start', start + resume)])
    vad_events.append(MockVADEvent('end', start + profile.duration_ms, .1))
    async with production_session(
        welcome='我正在说明原来的方案，还有几个细节需要慢慢介绍。',
        llm=MockLLM.scripted([(profile.text, '已收到完整问题。')]),
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted(scripts), tts=MockTTS(char_seconds=.12, chunk_delay_ms=40),
        vad=MockVAD.scripted(vad_events),
    ) as (_, h):
        h.audio_in.feed_pcm(synth_voiced((start + profile.duration_ms) / 1000))
        h.audio_in.feed_silence(.2)
        await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
            and '已收到完整问题。' in str(e.payload), timeout=8)
        assert h.events.user_messages() == [profile.text]
        assert h.events.agent_messages()[-1] == '已收到完整问题。'
        classifier.classify.assert_awaited_once()
        assert classifier.classify.call_args.args[0] == profile.text


@pytest.mark.asyncio
@pytest.mark.parametrize('vad_end_ms', [950, 1400])
@pytest.mark.parametrize('intent', [InterruptIntent.NORMAL_INTERRUPT, InterruptIntent.BACKCHANNEL])
async def test_fragmented_final_intent_survives_deadline_and_vad_stop(intent, vad_end_ms):
    """Provider sentence fragments must retain one canonical interruption candidate."""
    async def classify(*args, **kwargs):
        await asyncio.sleep(.7)
        return InterruptIntentResult(intent, 0.0, 'fault_injection', '')

    classifier = SimpleNamespace(classify=AsyncMock(side_effect=classify))
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    prefix, tail = (
        ('不是。', '我刚才说错了。') if intent is InterruptIntent.NORMAL_INTERRUPT
        else ('别停。', '继续讲。')
    )
    welcome = '这里正在播放原本的回答，后面还有几个细节需要继续说明。'
    llm = MockLLM.scripted([(tail, '这是更正后的回复。')])
    async with production_session(
        welcome=welcome, llm=llm,
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=700),
            ScriptedTranscript(text=tail, interims=[tail[:2]], final=False, trigger_after_ms=850),
            ScriptedTranscript(text=tail, trigger_after_ms=1100),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', vad_end_ms, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.5))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        async with asyncio.timeout(4):
            while pipeline._semantic_interrupts._intent_result is None:
                await asyncio.sleep(.02)
        final_request = classifier.classify.call_args.args[0]
        assert prefix.rstrip('。') in final_request and tail in final_request
        if intent is InterruptIntent.NORMAL_INTERRUPT:
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and '这是更正后的回复。' in str(e.payload), timeout=4)
            assert speech.interrupted
            assert len(h.events.user_messages()) == 1
            assert prefix.rstrip('。') in h.events.user_messages()[0]
            assert tail in h.events.user_messages()[0]
            assert h.events.agent_messages()[-1] == '这是更正后的回复。'
            assert llm.call_count == 1
            assert any(s.first_audible_at is not None and not s.cleared
                for s in h.audio_out.segments)
        else:
            await h.events.wait_for_agent_messages(1, timeout=6)
            assert h.events.agent_messages() == [welcome]
            assert not speech.interrupted
            assert not h.events.user_messages()
            assert not any(s.cleared for s in h.audio_out.segments)
            assert llm.call_count == 0
        assert not pipeline._semantic_interrupts._intent_tasks
