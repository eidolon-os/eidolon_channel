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
    tts = MockTTS(char_seconds=.1, chunk_delay_ms=40)
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([(text, '这是新的回复。')]),
        interrupt_classifier=classifier, turn_policy=policy,
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=tts,
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
            # Text/context equality cannot detect a frame tail replaced with
            # silence. Check the PCM produced by the existing deterministic TTS.
            import numpy as np
            expected = np.frombuffer(b''.join(
                synth_voiced(max(.05, min(60.0, len(t) * .1)),
                    sample_rate=tts.sample_rate, amplitude=.3)
                for t in tts.synth_texts
            ), dtype=np.int16)
            actual = np.frombuffer(h.audio_out.collected_pcm, dtype=np.int16)
            # AudioEmitter may append a 10 ms silent final-segment marker.
            # Permit only that trailing marker, never missing source samples.
            assert len(actual) - len(expected) in (0, tts.sample_rate // 100)
            assert not np.any(actual[len(expected):])
            actual = actual[:len(expected)]
            zeroed = np.count_nonzero((expected != 0) & (actual == 0))
            assert zeroed < tts.sample_rate * .005, 'unheard PCM was silenced instead of resumed'
        else:
            await asyncio.sleep(2.5)
            assert speech.interrupted
            assert not h.events.user_messages()
        classifier.classify.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_intent_timeout_resumes_original_pcm():
    """A real low-completeness final exhausts the existing evidence budget."""
    import numpy as np

    welcome = '这里正在播放原本的回答，后面还有几个细节需要继续说明。'
    tts = MockTTS(char_seconds=.1, chunk_delay_ms=40)
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([]), tts=tts,
        stt=MockSTT.scripted([ScriptedTranscript(text='好。', trigger_after_ms=700)]),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.1))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        await h.events.wait_for_agent_messages(1, timeout=10)
        assert not speech.interrupted
        assert h.events.agent_messages() == [welcome]
        assert not h.events.user_messages()
        assert not any(s.cleared for s in h.audio_out.segments)
        # Outside the fade envelopes, every source sample must survive.
        expected = np.frombuffer(synth_voiced(len(welcome) * .1,
            sample_rate=tts.sample_rate, amplitude=.3), dtype=np.int16)
        actual = np.frombuffer(h.audio_out.collected_pcm, dtype=np.int16)
        assert len(actual) - len(expected) in (0, tts.sample_rate // 100)
        assert not np.any(actual[len(expected):])
        zeroed = np.count_nonzero((expected != 0) & (actual[:len(expected)] == 0))
        assert zeroed < tts.sample_rate * .005


@pytest.mark.asyncio
@pytest.mark.parametrize('final_ms', [700, 1150])
@pytest.mark.parametrize('prefix_score,tail_score', [(.003, .424), (.003, .95), (.4, .95)], ids=['observed_eot', 'low_eot', 'uncertain_eot'])
@pytest.mark.parametrize('prefix,tail', [
    ('交付时间这一项。', '请改到下周三。'),
    ('关于参加的人数。', '改为七个人。'),
    ('For the destination.', 'Please change it to Shanghai.'),
])
async def test_ambiguous_sentence_final_preserves_pause_continuation(
    prefix, tail, prefix_score, tail_score, final_ms, monkeypatch, record_property,
):
    """ASR finality cannot decide whether a paused speaker will continue.

    Inject EOT evidence at the existing model boundary to isolate orchestration
    from wording/model accuracy: uncertain prefix, decisive continuation.
    """
    reply = '已收到完整修改要求。'
    llm = MockLLM.scripted([('', reply)])
    async with production_session(
        welcome='原来的说明还在进行，接下来还有一些内容需要继续向你介绍。',
        llm=llm,
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix, trigger_after_ms=final_ms),
            ScriptedTranscript(text=tail, trigger_after_ms=1800),
        ]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1050, .1),
            MockVADEvent('start', 1450), MockVADEvent('end', 2150, .1),
        ]),
    ) as (pipeline, h):
        monkeypatch.setattr(pipeline._get_eot_model()._context_eot,
            'semantic_completeness_score', lambda text: tail_score if tail in text else prefix_score)
        h.audio_in.feed_pcm(synth_voiced(2.3))
        await h.audio_out.wait_for_first_audio()
        await h.events.wait_for(lambda e: e.type == 'user_input_transcribed'
            and e.payload.transcript == prefix and e.payload.is_final, timeout=3)
        # Both event orderings have reached VAD-stop + a sentence final here,
        # before the next acoustic segment. Neither establishes a false trigger.
        async with asyncio.timeout(3):
            while h.session.user_state == 'speaking':
                await asyncio.sleep(.01)
        pause_state = pipeline._ducking.mixer.state
        pause_users = h.events.user_messages()
        pause_calls = llm.call_count
        record_property('pause_state', pause_state)
        await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
            and getattr(e.payload.item, 'text_content', '') == reply, timeout=6)
        record_property('completed_user_messages', json.dumps(h.events.user_messages(), ensure_ascii=False))
        assert pause_state == 'SUSPENDED'
        assert not pause_users
        assert pause_calls == 0
        assert len(h.events.user_messages()) == 1
        assert prefix.rstrip('。.') in h.events.user_messages()[0]
        assert tail in h.events.user_messages()[0]
        assert llm.call_count == 1


async def _wait_until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.01)


@pytest.mark.asyncio
@pytest.mark.parametrize('continuation', ['late_final', 'new_speech_in_cooldown', 'new_speech_after_cooldown', 'long_active_speech'])
async def test_resumed_output_keeps_receiving_interruption_evidence(continuation):
    prefix = '关于配送时间'
    tail = '请改到下周三。'
    long_speech = continuation == 'long_active_speech'
    is_new_speech = continuation not in {'late_final', 'long_active_speech'}
    start_ms = 2100 if continuation == 'new_speech_in_cooldown' else 2900
    final_ms = start_ms + 400 if is_new_speech else (7800 if long_speech else 2300)
    end_ms = start_ms + 650 if is_new_speech else (7400 if long_speech else 1050)
    events = [MockVADEvent('start', 350), MockVADEvent('end', 7400 if long_speech else 1050, .1)]
    if is_new_speech:
        events.extend([MockVADEvent('start', start_ms), MockVADEvent('end', end_ms, .1)])
    llm = MockLLM.scripted([('', '已收到修改。')])
    expected = tail if is_new_speech else prefix + tail
    async with production_session(
        welcome='原来的说明还有一些内容，我们可以慢慢介绍每个步骤和具体安排。' * 2,
        llm=llm,
        stt=MockSTT.scripted([
            ScriptedTranscript(text=prefix if is_new_speech else '', interims=[] if is_new_speech else [prefix], trigger_after_ms=700),
            ScriptedTranscript(text=expected, trigger_after_ms=final_ms),
        ]),
        tts=MockTTS(char_seconds=.2, chunk_delay_ms=200), vad=MockVAD.scripted(events),
    ) as (pipeline, h):
        # A fresh mobile idle report must not veto a server-owned candidate.
        packet = SimpleNamespace(topic=CLIENT_AUDIO_STATE_TOPIC,
            participant=SimpleNamespace(identity='human-simulator'),
            data=json.dumps({'schema_v': WIRE_SCHEMA_VERSION, 'type': 'client.audio_state',
                'seq': 1, 'input_mode': 'auto', 'playback_state': 'idle',
                'mic_muted': False, 'ptt': False, 'rms': 0.0, 'client_ts_ms': 0}).encode())
        pipeline._room_data.handle_packet(packet)
        model = pipeline._get_eot_model()._context_eot
        original = model.semantic_completeness_score
        model.semantic_completeness_score = lambda text: .95 if tail in text else .003
        try:
            h.audio_in.feed_pcm(synth_voiced(max(3.7, (final_ms + 400) / 1000)))
            await h.audio_out.wait_for_first_audio()
            speech = h.session.current_speech
            await _wait_until(lambda: pipeline._interruption_orchestrator.state.value == 'resumed_waiting_evidence', timeout=8)
            assert pipeline._ducking.mixer.state != 'SUSPENDED'
            assert not speech.interrupted
            assert not h.events.user_messages()
            if continuation == 'long_active_speech':
                recovery = next(event for event in pipeline._timeline.attrs['duck_events']
                    if event['event'] == 'output_resumed_pending_evidence')
                assert recovery['user_speaking'] is True
                assert recovery['speech_stopped_at'] is None
            await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                and getattr(e.payload.item, 'text_content', '') == '已收到修改。', timeout=7)
            assert speech.interrupted
            assert len(h.events.user_messages()) == 1
            assert h.events.user_messages()[0].replace(' ', '') == prefix + tail
            assert llm.call_count == 1
        finally:
            model.semantic_completeness_score = original


@pytest.mark.asyncio
async def test_incomplete_speech_resumes_output_then_expires_without_reply(record_property):
    text = '关于配送时间'
    llm = MockLLM.scripted([('', '不应调用')])
    async with production_session(
        welcome='原来的说明还有一些内容，我们可以慢慢介绍每个步骤和具体安排。' * 2,
        llm=llm, stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.2, chunk_delay_ms=200),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        model = pipeline._get_eot_model()._context_eot
        original = model.semantic_completeness_score
        model.semantic_completeness_score = lambda text: .003
        try:
            h.audio_in.feed_pcm(synth_voiced(1.2))
            await h.audio_out.wait_for_first_audio()
            speech = h.session.current_speech
            await _wait_until(lambda: pipeline._interruption_orchestrator.state.value == 'resumed_waiting_evidence', timeout=8)
            assert pipeline._ducking.mixer.state != 'SUSPENDED'
            assert pipeline._interruption_orchestrator.active
            timeline = pipeline._timeline
            recovery = next(event for event in timeline.attrs['duck_events']
                if event['event'] == 'output_resumed_pending_evidence')
            assert recovery['at'] >= recovery['speech_stopped_at']
            record_property('output_recovery', json.dumps(recovery))
            await _wait_until(lambda: not pipeline._interruption_orchestrator.active, timeout=6)
            events = timeline.attrs['interruption_orchestrator_events']
            resolved = next(event for event in events if event['event'] == 'candidate_resolved')
            resumed = next(event for event in events
                if event['event'] == 'turn_policy_decision' and event['action'] == 'resume')
            record_property('candidate_resolved_elapsed_ms', resolved['elapsed_ms'])
            record_property('output_resume_decision_elapsed_ms', resumed['elapsed_ms'])
            assert resolved['elapsed_ms'] > resumed['elapsed_ms']
            assert not h.events.user_messages()
            assert llm.call_count == 0
            assert not speech.interrupted
        finally:
            model.semantic_completeness_score = original


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
