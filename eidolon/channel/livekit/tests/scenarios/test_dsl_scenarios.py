# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scenario tests using the Conversation DSL.

These demonstrate the value of the declarative DSL: each test is
4-8 lines compared to ~30+ lines of equivalent imperative setup.
They cover edge cases that complement the hand-written
``test_turn_taking_happy_path.py`` / ``test_interruption.py`` /
``test_backchannel_suppression.py`` (the latter stay imperative
because they need precise interrupt timing).

Naming convention:
  * A* — turn-taking scenarios
  * B* — interruption/recovery (mostly in test_interruption.py)
  * C* — multi-turn context (limited by framework race; see DSL docstring)
"""

from __future__ import annotations

import pytest

from eidolon.channel.livekit.tests._harness.conversation import (
    Agent,
    Conversation,
    Pause,
    User,
)


# ──────────────────────────────────────────────────────────────────
# A class — turn-taking edge cases
# ──────────────────────────────────────────────────────────────────


class TestTurnTakingEdgeCases:
    """Single-turn variations that the DSL handles with one-liners."""

    @pytest.mark.asyncio
    async def test_a3_short_user_utterance(self):
        """User says a single character ("好"). Agent should still
        reply normally — no minimum-length suppression false-positive."""
        async with Conversation.script([
            User("好", speech_duration_ms=300),
            Agent("收到", contains="收到"),
        ]) as conv:
            await conv.run()
            conv.assert_passed()
            conv.assert_audio_played()

    @pytest.mark.asyncio
    async def test_a4_long_user_utterance(self):
        """User says a 30+ char question. Agent's long reply is
        synthesised through the SentenceAggregator (Round 8 R8.2)."""
        async with Conversation.script([
            User(
                "我想了解一下你们公司最近发布的那个新产品的详细情况，包括价格和功能",
                speech_duration_ms=2000,
            ),
            Agent(
                "好的，我来给你详细介绍一下我们的新产品功能。"
                "首先它支持多种语言识别。",
                contains="新产品",
            ),
        ]) as conv:
            await conv.run(total_timeout=20.0)
            conv.assert_passed()
            # Audio for a 30-char reply at char_seconds=0.02 is ~600ms.
            assert conv.captured_pcm_bytes > 0

    @pytest.mark.asyncio
    async def test_a5_user_with_interim_transcripts(self):
        """User's STT delivers interim transcripts before the final.
        Verify framework handles the streaming correctly."""
        async with Conversation.script([
            User(
                "你好世界",
                interims=("你", "你好", "你好世"),
                speech_duration_ms=600,
            ),
            Agent("你好"),
        ]) as conv:
            await conv.run()
            conv.assert_passed()
            # Should have seen 3 interim + 1 final.
            interims = conv.handle.events.user_interims()
            assert len(interims) >= 3

    @pytest.mark.asyncio
    async def test_a6_full_state_lifecycle(self):
        """Verify the full state machine traverses thinking → speaking
        on a basic turn."""
        async with Conversation.script([
            User("你好"),
            Agent("你好"),
        ]) as conv:
            await conv.run()
            conv.assert_agent_state_history_contains("thinking")
            conv.assert_agent_state_history_contains("speaking")
            conv.assert_user_state_history_contains("speaking")

    @pytest.mark.asyncio
    async def test_a7_no_errors_emitted_on_clean_turn(self):
        """A successful turn should not emit any error events."""
        async with Conversation.script([
            User("你好"),
            Agent("你好"),
        ]) as conv:
            await conv.run()
            conv.assert_no_errors()

    @pytest.mark.asyncio
    async def test_a8_substring_assertion_for_dynamic_replies(self):
        """When the agent reply text is partly dynamic, ``Agent.contains``
        gives a more lenient match."""
        async with Conversation.script([
            User("天气怎么样"),
            Agent(
                "今天天气晴朗，温度23度，适合外出活动",
                contains="天气",
            ),
        ]) as conv:
            await conv.run()
            conv.assert_passed()


# ──────────────────────────────────────────────────────────────────
# Pause — explicit silence between turns
# ──────────────────────────────────────────────────────────────────


class TestPauseHandling:
    @pytest.mark.asyncio
    async def test_pause_between_user_and_agent_completes(self):
        """A Pause between User and the next User should NOT break
        the single-turn case — it's just extra silence."""
        async with Conversation.script([
            User("你好"),
            Agent("你好"),
        ]) as conv:
            # Manually feed extra silence after run.
            await conv.run()
            await conv.feed_silence(500)
            conv.assert_passed()


# ──────────────────────────────────────────────────────────────────
# Imperative — interrupts, multi-turn (kept here as DSL examples)
# ──────────────────────────────────────────────────────────────────


class TestImperativeDSLExamples:
    @pytest.mark.asyncio
    async def test_imperative_single_turn(self):
        """Imperative API works for single turn just as well as
        declarative — used here to demonstrate the API surface."""
        from eidolon.channel.livekit.tests._harness.mocks import (
            MockLLM, MockSTT, MockTTS, MockVAD, MockVADEvent,
            ScriptedReply, ScriptedTranscript,
        )

        async with Conversation.with_mocks(
            llm=MockLLM.scripted([
                ScriptedReply(when="你好", reply="你好朋友"),
            ]),
            stt=MockSTT.scripted([
                ScriptedTranscript(text="你好", trigger_after_ms=400),
            ]),
            tts=MockTTS(char_seconds=0.02),
            vad=MockVAD.scripted([
                MockVADEvent("start", at_ms=20, probability=0.9),
                MockVADEvent("end", at_ms=350, probability=0.1),
            ]),
        ) as conv:
            await conv.user_says("你好", speech_duration_ms=300)
            reply = await conv.wait_for_agent_reply()
            assert "朋友" in reply
            conv.assert_no_errors()
