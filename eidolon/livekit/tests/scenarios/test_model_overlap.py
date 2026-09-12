"""Actual model evidence -> production turn owner -> SDK audio/turn effects."""

import asyncio
import time
from dataclasses import replace

import pytest
import pytest_asyncio

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import load_effective_config
from .._harness.audio import synth_silence, synth_voiced
from .._harness.headless import duck_settled, wait_until
from .._harness.latency import ReplyLatencyProbe
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


@pytest_asyncio.fixture(scope='module', loop_scope='module')
async def model_policy():
    cfg = load_effective_config()
    # 2500 rather than the shipped 1500: measured over 36 classifications of
    # these very texts the provider runs 652/1023/1497/1599 ms for
    # min/median/p90/max, so 1500 sits on its p90 and truncates roughly one
    # call in six once the pipeline is also driving audio. The budget decides
    # how long an answer may take, not what it says; the verdicts asserted
    # below are unchanged.
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interrupt=replace(
        cfg.turn_policy.interrupt, intent_provider='llm', intent_timeout_ms=2500,
    )))
    classifier = SharedStageFactory.build_interrupt_classifier(cfg)
    try:
        yield classifier, cfg.turn_policy
    finally:
        await classifier.aclose()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope='module')
@pytest.mark.parametrize('text,action', [
    ('好的', 'resume'),
    ('不要停止，请继续刚才的解释', 'resume'),
    ('不是，我说的是明天', 'reply'),
    ('停', 'stop'),
    ('嗯嗯你继续', 'resume'),
    ('不用回答我刚刚的问题，继续你原来讲的', 'resume'),
    ('他说不要讲了，但我还想听你继续解释', 'resume'),
    ('先不讲背景，告诉我费用', 'reply'),
    ('我，我想问的是，这个方案每个月要花多少钱？', 'reply'),
    ('呃……嗯，明白了，你接着说', 'resume'),
    ('不是让你停，我是说这个词叫 stop，接着解释吧', 'resume'),
    ('等等，先别说了，我需要想一想', 'stop'),
])
async def test_real_model_overlap_effects(text, action, model_policy, record_property):
    classifier, policy = model_policy
    welcome = '我们接着讨论这个方案，我先介绍一下背景，然后慢慢说明各个细节。'
    brain = MockLLM.scripted([(text, '收到，我按你的要求说明。')])
    async with production_session(
        welcome=welcome, llm=brain,
        interrupt_classifier=classifier, turn_policy=policy, warmup=True, real_time_audio=True,
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        probe = ReplyLatencyProbe(pipeline, h) if action == 'reply' else None
        try:
            pcm = synth_silence(.35) + synth_voiced(.7) + synth_silence(.3)
            h.audio_in.feed_pcm(pcm)
            await h.audio_out.wait_for_first_audio()
            speech = h.session.current_speech
            assert speech is not None
            # Capture the turn's timeline while it is live; the cancel below
            # clears the pipeline's reference in the same tick it records the
            # duck event this waits for.
            await wait_until(lambda: pipeline._timeline is not None, timeout=5)
            timeline = pipeline._timeline
            # The final arrives at 700 ms and the model has its own
            # intent_timeout_ms budget on top of it, so a fixed wait measured
            # from here is both load-sensitive and, at the policy's own worst
            # case, too short -- the 2.5 s sleep this replaced sat below 700 ms
            # plus that budget. Wait for the verdict, then for the duck effect
            # it causes: the two facts every assertion below reads. The effect
            # follows the verdict within a tick, but which terminal event it
            # records depends on the verdict.
            #
            # Time the classification while waiting for it. Giving the budget
            # room to cover this provider is what lets the case below judge its
            # answers, but it also stops the run from failing when the provider
            # is slow, so the latency has to be reported rather than inferred
            # from a red case.
            semantic = pipeline._semantic_interrupts
            classification = {}

            def verdict_arrived():
                if 'started_at' not in classification and semantic._intent_task is not None:
                    classification['started_at'] = time.monotonic()
                return semantic._intent_result is not None

            budget_sec = pipeline._turn_policy.interrupt.intent_timeout_ms / 1000
            await wait_until(verdict_arrived, timeout=budget_sec + 3)
            record_property('intent_budget_ms', pipeline._turn_policy.interrupt.intent_timeout_ms)
            record_property('intent_latency_ms', round(
                (time.monotonic() - classification['started_at']) * 1000, 1))
            await wait_until(lambda: duck_settled(timeline),
                             timeout=pipeline._turn_policy.eot.speech_merge_grace_ms / 1000 + 3)
            record_property('expected_action', action)
            record_property('interrupted', speech.interrupted)
            record_property('output_state', pipeline._ducking.mixer.state)
            record_property('intent', getattr(pipeline._semantic_interrupts._intent_result, 'intent', None))
            assert pipeline._semantic_interrupts._intent_result is not None
            assert pipeline._semantic_interrupts._intent_result.source == 'llm'
            assert speech.interrupted is (action != 'resume')
            if action == 'resume':
                assert pipeline._ducking.mixer.state == 'NORMAL'
                await h.events.wait_for_agent_messages(1, timeout=8)
                assert h.events.agent_messages() == [welcome]
                assert not h.events.user_messages()
                assert not any(s.cleared for s in h.audio_out.segments)
            elif action == 'reply':
                await h.events.wait_for(lambda e: e.type == 'conversation_item_added'
                    and '收到，我按你的要求说明。' in str(e.payload), timeout=8)
                assert h.events.user_messages() == [text]
                assert h.events.agent_messages()[-1] == '收到，我按你的要求说明。'
                report = probe.report(pcm)
                record_property('stop_to_reply_audio_ms', report['stop_to_reply_audio_ms'])
                record_property('eot_predictions', report['eot_predictions'])
            else:
                await asyncio.sleep(.5)
                assert not h.events.user_messages()
        finally:
            if probe is not None:
                probe.close()
