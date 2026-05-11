# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scenario A1 — happy-path single turn.

The most basic conversational unit:
    User says "你好"
    Agent replies "你好！很高兴见到你"

Asserts:
  * STT final transcript reaches AgentSession (user_input_transcribed).
  * Agent state machine traverses listening → thinking → speaking → idle.
  * LLM is invoked exactly once with the user's text.
  * TTS audio is captured (non-empty PCM).
  * conversation_item_added carries both user and assistant messages.
  * No errors, no false interruptions.
"""

from __future__ import annotations

import asyncio

import pytest

from eidolon.channel.livekit.tests._harness.audio import (
    pcm_rms,
    synth_silence,
    synth_voiced,
)
from eidolon.channel.livekit.tests._harness.headless import headless_session
from eidolon.channel.livekit.tests._harness.mocks import (
    MockLLM,
    MockSTT,
    MockTTS,
    MockVAD,
    MockVADEvent,
    ScriptedReply,
    ScriptedTranscript,
)


@pytest.mark.asyncio
async def test_single_turn_user_question_agent_reply():
    """User asks one thing → agent replies once → state machine settles."""
    async with headless_session(
        llm=MockLLM.scripted(
            [ScriptedReply(when="你好", reply="你好！很高兴见到你")],
            default_reply="?",
        ),
        stt=MockSTT.scripted(
            [
                ScriptedTranscript(
                    text="你好",
                    interims=["你"],
                    interim_gap_ms=10,
                    trigger_after_ms=80,
                )
            ]
        ),
        tts=MockTTS(char_seconds=0.02),
        vad=MockVAD.scripted(
            [
                MockVADEvent("start", at_ms=20, probability=0.9),
                MockVADEvent("end", at_ms=180, probability=0.1),
            ]
        ),
    ) as h:
        # Push 200 ms voiced audio + 200 ms trailing silence.
        h.audio_in.feed_pcm(synth_voiced(0.2))
        h.audio_in.feed_silence(0.2)

        # Wait until agent has spoken AND returned to a non-speaking
        # state — that's when the assistant ChatMessage is committed.
        await h.events.wait_for_state(agent="speaking", timeout=10.0)
        # Then wait for the speaking → listening/idle transition.
        await h.events.wait_for(
            lambda e: e.type == "agent_state_changed"
            and e.payload.old_state == "speaking",
            timeout=10.0,
        )
        # Plus a small grace for ConversationItemAdded to land.
        await asyncio.sleep(0.1)

        # ── Assertions ──────────────────────────────────────────

        # STT: exactly one final transcript with the right text.
        finals = h.events.user_finals()
        assert len(finals) == 1
        assert finals[0].transcript == "你好"

        # LLM: invoked once, saw the user message.
        llm_mock = h.session.llm  # type: ignore
        assert isinstance(llm_mock, MockLLM)
        assert llm_mock.call_count == 1

        # State transitions: user listening→speaking→listening,
        # agent listening→thinking→speaking.
        agent_history = h.events.agent_state_history()
        assert "thinking" in agent_history
        assert "speaking" in agent_history
        # Specifically, thinking precedes speaking.
        assert agent_history.index("thinking") < agent_history.index("speaking")

        user_history = h.events.user_state_history()
        # User went speaking at some point.
        assert "speaking" in user_history

        # TTS: audio was captured.
        assert h.audio_out.captured_bytes > 0
        assert pcm_rms(h.audio_out.collected_pcm) > 0.05

        # Conversation log: both turns recorded.
        user_msgs = h.events.user_messages()
        agent_msgs = h.events.agent_messages()
        assert any("你好" in m for m in user_msgs)
        assert any("你好！很高兴见到你" in m for m in agent_msgs)

        # No errors, no false interruptions.
        assert h.events.of_type("error") == []
        assert h.events.of_type("agent_false_interruption") == []


@pytest.mark.asyncio
async def test_no_input_no_response():
    """If user never speaks, agent never replies."""
    async with headless_session(
        llm=MockLLM.scripted([]),  # default reply, but should never fire
        stt=MockSTT.scripted([]),  # no transcripts scripted
        tts=MockTTS(char_seconds=0.02),
        vad=MockVAD.silent(),
    ) as h:
        # Feed pure silence for 500 ms.
        h.audio_in.feed_silence(0.5)
        # Give the pipeline a beat to (not) react.
        import asyncio

        await asyncio.sleep(0.2)

        # Assertions
        assert h.events.user_finals() == []
        assert h.audio_out.captured_bytes == 0
        # LLM never called (we expect call_count == 0 because there
        # was no user transcript to trigger generation).
        llm_mock = h.session.llm  # type: ignore
        assert isinstance(llm_mock, MockLLM)
        assert llm_mock.call_count == 0
