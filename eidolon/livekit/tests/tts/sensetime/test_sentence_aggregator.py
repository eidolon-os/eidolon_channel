# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the SentenceAggregator (Round 8 R8.2.a)."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import pytest

from eidolon.livekit.plugins.tts.sensetime._aggregator import (
    SentenceAggregator,
)


def _make_collector() -> tuple[Callable[[str], Awaitable[None]], list[str]]:
    """Return ``(on_segment, segments_list)`` — the callback appends to the list."""
    segments: list[str] = []

    async def on_segment(text: str) -> None:
        segments.append(text)

    return on_segment, segments


@pytest.mark.asyncio
async def test_hard_punct_flushes_immediately():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(on_segment, idle_ms=500)
    try:
        await agg.feed("Hello")
        await agg.feed(", ")
        await agg.feed("world")
        await agg.feed(".")
        # Hard punct triggers immediate flush.
        assert segments == ["Hello, world."]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_chinese_hard_punct_flushes():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(on_segment, idle_ms=500)
    try:
        await agg.feed("你好")
        await agg.feed("世界")
        await agg.feed("。")
        assert segments == ["你好世界。"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_question_mark_flushes():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(on_segment, idle_ms=500)
    try:
        await agg.feed("What")
        await agg.feed(" is ")
        await agg.feed("this?")
        assert segments == ["What is this?"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_soft_punct_holds_until_min_length():
    """Soft punct alone (short buffer) should NOT flush — wait for more."""
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=12, hard_max_chars=60, idle_ms=2000,
    )
    try:
        await agg.feed("Hi")
        await agg.feed(",")
        # Buffer is "Hi," — only 3 chars, below soft_min_chars=12.
        assert segments == []
        # Add more chars. Once buf_len passes soft_min and last char is
        # soft punct, it flushes.
        await agg.feed(" my name is")  # buf="Hi, my name is" (14 chars, no soft punct at end)
        await agg.feed(",")  # now ends with "," and len>=12 → flush
        assert segments == ["Hi, my name is,"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_hard_max_force_flushes_even_without_punct():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=12, hard_max_chars=20, idle_ms=2000,
    )
    try:
        # No punct at all, just keep feeding.
        await agg.feed("a" * 19)   # 19 chars, below hard_max
        assert segments == []
        await agg.feed("bc")        # 21 chars total, ≥ hard_max → flush
        assert len(segments) == 1
        assert segments[0] == "a" * 19 + "bc"
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_idle_timer_flushes_held_buffer():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=20, hard_max_chars=100, idle_ms=100,
    )
    try:
        await agg.feed("incomplete")  # 10 chars, no punct, below soft_min
        assert segments == []
        # Wait for idle timer.
        await asyncio.sleep(0.2)
        assert segments == ["incomplete"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_idle_timer_cancelled_by_subsequent_feed():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=100, hard_max_chars=100, idle_ms=100,
    )
    try:
        await agg.feed("part1")
        # Wait less than idle_ms, then feed again — timer should reset.
        await asyncio.sleep(0.05)
        await agg.feed("part2")
        await asyncio.sleep(0.05)
        # Combined wait ~100 ms but no full idle window after a feed.
        # Should NOT have flushed yet.
        assert segments == []
        # Now wait the full idle window after the last feed.
        await asyncio.sleep(0.15)
        assert segments == ["part1part2"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_explicit_flush_emits_buffer():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=100, hard_max_chars=100, idle_ms=2000,
    )
    try:
        await agg.feed("partial")
        assert segments == []
        await agg.flush()
        assert segments == ["partial"]
        # Subsequent flush on empty buffer is a no-op.
        await agg.flush()
        assert segments == ["partial"]
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_idle_timer():
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(
        on_segment, soft_min_chars=100, hard_max_chars=100, idle_ms=50,
    )
    await agg.feed("x")
    # Close before idle timer fires.
    await agg.aclose()
    # Wait past the original idle window — should NOT emit (closed).
    await asyncio.sleep(0.15)
    assert segments == []


@pytest.mark.asyncio
async def test_feed_after_aclose_raises():
    on_segment, _ = _make_collector()
    agg = SentenceAggregator(on_segment)
    await agg.aclose()
    with pytest.raises(RuntimeError):
        await agg.feed("oops")


@pytest.mark.asyncio
async def test_realistic_llm_token_stream():
    """End-to-end-ish: simulate the production LLM token stream that
    triggered the bug (12 tiny tokens producing 12 task_continues).
    With the aggregator, the same input should produce ≤ 4 segments.
    """
    on_segment, segments = _make_collector()
    agg = SentenceAggregator(on_segment, idle_ms=200)
    try:
        # Reproduces the production log: "Hey there I'm ready to help.
        # What can I do for you?"
        for tok in ["Hey ", "there ", "I'm ", "ready ", "to ", "help",
                    ".", " What ", "can ", "I ", "do ", "for ", "you", "?"]:
            await agg.feed(tok)
        await agg.flush()

        # Expected: 2 sentences.
        # Sentence 1: "Hey there I'm ready to help." — flushed by "."
        # Sentence 2: "What can I do for you?" — flushed by "?"
        assert len(segments) == 2, f"expected 2 sentences, got {segments}"
        assert "Hey" in segments[0] and "help" in segments[0]
        assert "What" in segments[1] and "you" in segments[1]
        assert segments[0].endswith(".")
        assert segments[1].endswith("?")
    finally:
        await agg.aclose()


@pytest.mark.asyncio
async def test_validates_thresholds():
    on_segment, _ = _make_collector()
    with pytest.raises(ValueError):
        SentenceAggregator(on_segment, soft_min_chars=0)
    with pytest.raises(ValueError):
        SentenceAggregator(on_segment, hard_max_chars=0)
    with pytest.raises(ValueError):
        SentenceAggregator(on_segment, soft_min_chars=50, hard_max_chars=20)
    with pytest.raises(ValueError):
        SentenceAggregator(on_segment, idle_ms=0)


@pytest.mark.asyncio
async def test_callback_exception_does_not_break_aggregator():
    """If the callback raises, the aggregator should log and continue."""
    call_count = 0

    async def on_segment(text: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated callback failure")

    agg = SentenceAggregator(on_segment, idle_ms=2000)
    try:
        await agg.feed("Sentence one.")    # callback raises (logged, swallowed)
        await agg.feed("Sentence two.")    # should still work
        assert call_count == 2
    finally:
        await agg.aclose()
