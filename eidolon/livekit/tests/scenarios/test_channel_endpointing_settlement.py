"""Channel-owned barge-in must settle before the real SDK admits a reply."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.common.config import TurnPolicyConfig

from .._harness.audio import synth_voiced
from .._harness.mocks import (
    MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent, ScriptedTranscript,
)
from .._harness.production import production_session
from .._harness.headless import wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("final_ms", [800, 1300])
async def test_default_policy_late_cancel_delivers_one_reply(final_ms):
    """Real SDK + ONNX EOT, final on either side of VAD stop; no intent model."""
    text, reply = "这个故事不好听。", "那我们换一个故事。"
    async with production_session(
        welcome="我正在讲一个很长的故事，后面还有许多内容没有讲完。",
        llm=MockLLM.scripted([(text, reply)]),
        stt=MockSTT.scripted([ScriptedTranscript(text=text, trigger_after_ms=final_ms)]),
        tts=MockTTS(char_seconds=.4, chunk_delay_ms=100),
        vad=MockVAD.scripted([
            MockVADEvent("start", 350), MockVADEvent("end", 1050, .1),
        ]),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.4))
        await h.audio_out.wait_for_first_audio()
        old_speech = h.session.current_speech
        assert not old_speech.allow_interruptions
        await h.events.wait_for(
            lambda e: e.type == "conversation_item_added"
            and getattr(e.payload.item, "text_content", "") == reply,
            timeout=7,
        )
        assert old_speech.interrupted
        assert h.events.user_messages() == [text]
        assert h.events.agent_messages().count(reply) == 1
        user_event = await h.events.wait_for(
            lambda e: e.type == "conversation_item_added"
            and getattr(e.payload.item, "role", None) == "user",
        )
        assert any(segment.first_audible_at is not None
                   and segment.first_audible_at >= user_event.timestamp
                   and not segment.cleared for segment in h.audio_out.segments)


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_provider", ["none", "llm"])
async def test_sdk_close_releases_pending_channel_endpointing(intent_provider):
    started = asyncio.Event()

    async def classify(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(
        policy.interrupt, intent_provider=intent_provider, intent_timeout_ms=6000,
    ))
    async with production_session(
        welcome="我正在讲一个很长的故事，后面还有许多内容没有讲完。",
        llm=MockLLM.scripted([]),
        stt=MockSTT.scripted([ScriptedTranscript(text="好。", trigger_after_ms=700)]),
        tts=MockTTS(char_seconds=.4, chunk_delay_ms=100),
        vad=MockVAD.scripted([
            MockVADEvent("start", 350), MockVADEvent("end", 1050, .1),
        ]),
        turn_policy=policy,
        interrupt_classifier=SimpleNamespace(classify=AsyncMock(side_effect=classify)),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.4))
        await wait_until(lambda: pipeline._interruption_orchestrator.awaiting_post_speech_evidence,
                         timeout=4)
        if intent_provider == "llm":
            await asyncio.wait_for(started.wait(), 1)
        # Closing the SDK directly exercises the public Agent.on_exit boundary,
        # before the outer production harness calls pipeline.shutdown().
        await asyncio.wait_for(h.session.aclose(), 2)
        assert pipeline._interruption_orchestrator.closed
        assert not h.events.user_messages()


@pytest.mark.asyncio
async def test_sdk_exit_rejects_candidate_before_transcript_settlement_deadline(monkeypatch):
    """Channel ends its pending turn before the SDK drains provider input."""
    async with production_session(
        welcome='', llm=MockLLM.scripted([]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text='好啊。', trigger_after_ms=700),
            ScriptedTranscript(text='', interims=['再帮我查一下'], final=False,
                               trigger_after_ms=1100),
        ]),
        tts=MockTTS(char_seconds=.01),
        vad=MockVAD.scripted([
            MockVADEvent('start', 350), MockVADEvent('end', 1000, .1),
        ]),
        real_time_audio=True,
    ) as (pipeline, h):
        monkeypatch.setattr(pipeline._get_eot_model(), 'predict_end_of_turn',
                            AsyncMock(return_value=.99))
        h.audio_in.feed_pcm(synth_voiced(1.5))
        await wait_until(lambda: bool(pipeline._user_turns._transcript_change_waiters), timeout=4)
        candidate = pipeline._user_turns.active
        closing = asyncio.create_task(h.session.aclose())
        try:
            await wait_until(lambda: candidate.state == 'rejected'
                             and not pipeline._user_turns._transcript_change_waiters, timeout=1)
        finally:
            # The SDK independently gives a provider with no FINAL its existing
            # transcription timeout. This test constrains Channel settlement,
            # not that separate SDK drain; do not patch the SDK or its timer.
            await asyncio.wait_for(closing, timeout=6)
        assert candidate.state == 'rejected'
        assert not pipeline._user_turns._transcript_change_waiters
        assert not h.events.user_messages()
        assert not h.events.agent_messages()


@pytest.mark.asyncio
async def test_withdrawn_interim_restores_final_and_delivers_one_sdk_reply():
    """Replay FINAL(A), INTERIM(B), empty FINAL(B) while output is suspended."""
    from eidolon.livekit.agent.turn_policy import InterruptIntent, InterruptIntentResult
    from eidolon.livekit.common.transcript_evidence import TranscriptEvidence

    text, reply = "继续播放。", "好的，继续播放。"
    calls = 0
    cancelled = asyncio.Event()

    async def classify(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return InterruptIntentResult(InterruptIntent.TOPIC_SWITCH, .99, "model", "new request")

    def evidence(revision, sequence):
        return TranscriptEvidence(stream_key="withdrawal-stream", revision_key=revision,
                                  sequence=sequence).as_metadata()

    policy = TurnPolicyConfig()
    policy = replace(policy, interrupt=replace(policy.interrupt, intent_provider="llm"))
    async with production_session(
        welcome="我正在讲一个很长的故事，后面还有许多内容没有讲完。",
        llm=MockLLM.scripted([(text, reply)]),
        stt=MockSTT.scripted([
            ScriptedTranscript(text=text, trigger_after_ms=700, metadata=evidence("a", 1)),
            ScriptedTranscript(text="", interims=["杂音"], final=False,
                               trigger_after_ms=850, metadata=evidence("b", 2)),
            ScriptedTranscript(text="", trigger_after_ms=1200, metadata=evidence("b", 3)),
        ]),
        tts=MockTTS(char_seconds=.2, chunk_delay_ms=50),
        vad=MockVAD.scripted([MockVADEvent("start", 350), MockVADEvent("end", 1050, .1)]),
        turn_policy=policy, real_time_audio=True,
        interrupt_classifier=SimpleNamespace(classify=classify),
    ) as (pipeline, h):
        h.audio_in.feed_pcm(synth_voiced(1.5))
        await h.audio_out.wait_for_first_audio()
        old_speech = h.session.current_speech
        await h.events.wait_for(
            lambda e: e.type == "conversation_item_added"
            and getattr(e.payload.item, "text_content", "") == reply,
            timeout=8,
        )
        assert cancelled.is_set()
        assert calls == 2
        assert old_speech.interrupted
        assert h.events.user_messages() == [text]
        assert h.events.agent_messages().count(reply) == 1
