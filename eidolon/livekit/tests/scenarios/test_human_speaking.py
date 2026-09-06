"""Production callback-chain E2E with real SDK/EOT and scripted speech providers.

Known semantic acceptance gaps use strict xfail: an unexpected pass forces review.
Run ``--runxfail -k overlap_intent`` to see their current failures directly.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.common.config import TurnPolicyConfig

from .._harness.audio import synth_voiced
from .._harness.mocks import MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript
from .._harness.production import production_session


from .._harness.human_profiles import HUMAN_PROFILES


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
@pytest.mark.parametrize('profile', HUMAN_PROFILES, ids=lambda p: p.name)
async def test_human_delivery_commits_once(profile, mode, record_property, caplog):
    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider='llm'))
    classifier = SimpleNamespace(classify=AsyncMock(side_effect=AssertionError('no overlap')))
    scripts = [ScriptedTranscript(
        text=profile.text, trigger_after_ms=100,
        interims=list(profile.interims), interim_gap_ms=profile.gap_ms,
        final=not profile.final_delay_ms,
    )]
    if profile.final_delay_ms:
        scripts.append(ScriptedTranscript(
            text=profile.text,
            trigger_after_ms=profile.duration_ms + profile.final_delay_ms,
        ))
    vad_events = [MockVADEvent('start', 10)]
    for stop, resume in profile.pauses:
        vad_events.extend([MockVADEvent('end', stop, .1), MockVADEvent('start', resume)])
    vad_events.append(MockVADEvent('end', profile.duration_ms, .1))
    async with production_session(
        mode=mode, llm=MockLLM.scripted([(profile.text, '好的，我来看看。')]),
        turn_policy=policy, interrupt_classifier=classifier,
        stt=MockSTT.scripted(scripts), tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted(vad_events),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(profile.duration_ms / 1000))
        h.audio_in.feed_silence(.2)
        await h.events.wait_for_agent_messages(1, timeout=8)
        assert h.events.user_messages() == [profile.text]
        assert h.events.agent_messages() == ['好的，我来看看。']
        assert h.audio_out.collected_pcm
        assert not any(s.cleared for s in h.audio_out.segments)
        record_property('mode', mode)
        record_property('profile', profile.name)
        # Real model scores are diagnostic evidence, never canned test scores.
        record_property('eot_score', pipeline._get_eot_model().current_eot_score)
        assert not [r for r in caplog.records if r.levelname == 'ERROR']
        classifier.classify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
async def test_welcome_inherits_interruption_owner(mode):
    async with production_session(
        mode=mode, welcome='欢迎回来，我们可以慢慢聊。',
        llm=MockLLM.scripted([]), stt=MockSTT.scripted([]),
        tts=MockTTS(char_seconds=.05, chunk_delay_ms=20), vad=MockVAD.silent(),
    ) as (_, h):
        await h.audio_out.wait_for_first_audio()
        assert h.session.current_speech is not None
        assert h.session.current_speech.allow_interruptions is False
        await h.events.wait_for_agent_messages(1)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex'])
async def test_brief_noise_during_reply_does_not_destroy_output(mode, caplog):
    welcome = '我们接着讲这个故事，接下来还有几个细节可以慢慢说明。'
    async with production_session(
        mode=mode, welcome=welcome,
        llm=MockLLM.scripted([]), stt=MockSTT.scripted([]),
        tts=MockTTS(char_seconds=.05, chunk_delay_ms=20),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 470, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(.5))
        await h.events.wait_for_agent_messages(1, timeout=6)
        assert h.events.user_messages() == []
        assert h.events.agent_messages() == [welcome]
        assert not any(s.cleared for s in h.audio_out.segments)
        assert not [r for r in caplog.records if r.levelname == 'ERROR']


@pytest.mark.asyncio
@pytest.mark.parametrize('text,should_interrupt', [
    pytest.param('好的', False, marks=pytest.mark.xfail(strict=True, reason='EOT completeness is still used as interruption intent; see execution report FD-INTENT')),
    ('嗯嗯你继续', False),
    pytest.param('不要停止，请继续刚才的解释', False, marks=pytest.mark.xfail(strict=True, reason='Negated commands need intent evidence independent of EOT; see FD-INTENT')),
    pytest.param('不是，我说的是明天', True, marks=pytest.mark.xfail(strict=True, reason='Low EOT corrective interruption is not recognized; see FD-INTENT')),
    ('停', True),
])
async def test_overlap_intent_contract(text, should_interrupt, record_property):
    welcome = '我们接着讨论这个问题，我先介绍一下背景，然后慢慢说明各个细节。'
    async with production_session(
        welcome=welcome, llm=MockLLM.scripted([(text, '收到。')]),
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.1, chunk_delay_ms=40),
        vad=MockVAD.scripted([MockVADEvent('start', 350), MockVADEvent('end', 1050, .1)]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.1))
        await h.audio_out.wait_for_first_audio()
        speech = h.session.current_speech
        assert speech is not None
        # Accepted interruption must settle promptly. Non-interruptions must
        # resume the same response, not regenerate or silently lose its tail.
        import asyncio
        await asyncio.sleep(2)
        record_property('expected_interrupt', should_interrupt)
        record_property('observed_interrupt', speech.interrupted)
        record_property('output_state', pipeline._ducking.mixer.state)
        assert speech.interrupted is should_interrupt
        if not should_interrupt:
            assert pipeline._ducking.mixer.state == 'NORMAL'
            assert h.events.user_messages() == []
