# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for tests/_harness/mocks/mock_llm.py."""

from __future__ import annotations

import asyncio
import time

import pytest
from livekit.agents.llm import ChatChunk, ChatContext

from eidolon.channel.livekit.tests._harness.mocks import MockLLM, ScriptedReply


def _build_ctx(*messages: tuple[str, str]) -> ChatContext:
    """Build a ChatContext with (role, text) tuples."""
    from livekit.agents.llm.chat_context import ChatMessage

    items = [ChatMessage(role=role, content=[text]) for role, text in messages]
    return ChatContext(items)


async def _collect(stream) -> list[ChatChunk]:
    chunks: list[ChatChunk] = []
    async for ev in stream:
        chunks.append(ev)
    return chunks


class TestScriptedMode:
    async def test_substring_match_wins(self):
        llm = MockLLM.scripted(
            [("你好", "你好啊"), ("再见", "拜拜")], default_reply="?"
        )
        chunks = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "你好世界"))))
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == "你好啊"

    async def test_no_match_uses_default(self):
        llm = MockLLM.scripted(
            [("你好", "你好啊")], default_reply="default reply"
        )
        chunks = await _collect(
            llm.chat(chat_ctx=_build_ctx(("user", "完全不相关")))
        )
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == "default reply"

    async def test_first_match_wins_in_order(self):
        # Both patterns match — the first scripted entry should win.
        llm = MockLLM.scripted([("你好", "first"), ("好", "second")])
        chunks = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "你好"))))
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == "first"

    async def test_predicate_match(self):
        llm = MockLLM.scripted(
            [
                ScriptedReply(
                    when=lambda t: len(t) > 10, reply="long input received"
                ),
                ScriptedReply(when=None, reply="short"),
            ]
        )
        chunks = await _collect(
            llm.chat(chat_ctx=_build_ctx(("user", "this is a long message")))
        )
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == "long input received"

    async def test_emits_final_usage_chunk(self):
        llm = MockLLM.scripted([("你好", "ok")])
        chunks = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "你好"))))
        # Last chunk should carry usage.
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.completion_tokens == len("ok")

    async def test_chunk_size_one_yields_per_char(self):
        llm = MockLLM.scripted([("你好", "abc")])
        chunks = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "你好"))))
        # 3 content chunks (one per char) + 1 usage = 4 chunks.
        content_chunks = [c for c in chunks if c.delta and c.delta.content]
        assert len(content_chunks) == 3
        assert [c.delta.content for c in content_chunks] == ["a", "b", "c"]

    async def test_chunk_size_larger_than_one(self):
        llm = MockLLM(
            scripts=[ScriptedReply(when="hi", reply="hello world")],
            chunk_size=5,
        )
        chunks = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "hi"))))
        content_chunks = [c for c in chunks if c.delta and c.delta.content]
        # "hello world" has 11 chars / chunk_size=5 → 3 chunks
        assert len(content_chunks) == 3


class TestEchoMode:
    async def test_echoes_user_message(self):
        llm = MockLLM.echo()
        chunks = await _collect(
            llm.chat(chat_ctx=_build_ctx(("user", "echo me")))
        )
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == "echo me"

    async def test_echo_with_prefix(self):
        llm = MockLLM.echo(prefix=">>> ")
        chunks = await _collect(
            llm.chat(chat_ctx=_build_ctx(("user", "test")))
        )
        text = "".join(c.delta.content for c in chunks if c.delta and c.delta.content)
        assert text == ">>> test"


class TestErrorMode:
    async def test_chat_raises_configured_exc(self):
        llm = MockLLM.errors_with(RuntimeError("upstream down"))
        # The exception is raised inside _run() and surfaces via
        # the LLM "error" event + propagates out of __anext__.
        with pytest.raises(RuntimeError, match="upstream down"):
            await _collect(llm.chat(chat_ctx=_build_ctx(("user", "hi"))))

    async def test_per_script_raises(self):
        llm = MockLLM(
            scripts=[
                ScriptedReply(
                    when="boom", reply="x", raises=ValueError("kaboom")
                ),
                ScriptedReply(when=None, reply="ok"),
            ]
        )
        # Non-matching → normal reply.
        out = await _collect(llm.chat(chat_ctx=_build_ctx(("user", "fine"))))
        text = "".join(c.delta.content for c in out if c.delta and c.delta.content)
        assert text == "ok"
        # Matching → raises.
        with pytest.raises(ValueError, match="kaboom"):
            await _collect(llm.chat(chat_ctx=_build_ctx(("user", "boom"))))


class TestRecording:
    async def test_records_prompts_and_count(self):
        llm = MockLLM.scripted([("a", "A"), ("b", "B")])
        await _collect(llm.chat(chat_ctx=_build_ctx(("user", "a"))))
        await _collect(llm.chat(chat_ctx=_build_ctx(("user", "b"))))
        await _collect(llm.chat(chat_ctx=_build_ctx(("user", "c"))))
        assert llm.call_count == 3
        assert len(llm.prompts) == 3


class TestDelays:
    async def test_first_chunk_delay(self):
        llm = MockLLM(
            scripts=[
                ScriptedReply(when=None, reply="hi", delay_first_chunk_ms=50)
            ]
        )
        t0 = time.monotonic()
        await _collect(llm.chat(chat_ctx=_build_ctx(("user", "x"))))
        elapsed = time.monotonic() - t0
        # Allow scheduling slack.
        assert elapsed >= 0.04

    async def test_chunk_delay_per_char(self):
        llm = MockLLM(
            scripts=[ScriptedReply(when=None, reply="abc")],
            chunk_delay_ms=20,
        )
        t0 = time.monotonic()
        await _collect(llm.chat(chat_ctx=_build_ctx(("user", "x"))))
        elapsed = time.monotonic() - t0
        # 3 chars × 20ms ≥ 60ms.
        assert elapsed >= 0.05
