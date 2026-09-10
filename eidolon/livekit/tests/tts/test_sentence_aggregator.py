"""Streaming text boundaries must survive provider/SDK chunk placement."""
import asyncio

import pytest
from livekit.agents.voice.transcription.filters import filter_markdown

from eidolon.livekit.plugins.tts._aggregator import SentenceAggregator


@pytest.mark.asyncio
@pytest.mark.parametrize('chunks', [
    ['第一句话', '。第二句话'],
    ['First sentence', '. Next sentence'],
    ['最初的说明', '；后续内容'],
    ['First sentence. Next sentence'],
])
async def test_complete_text_is_sent_before_end_of_input(chunks):
    sent = []

    async def emit(text):
        sent.append(text)

    aggregator = SentenceAggregator(emit)
    try:
        for chunk in chunks:
            await aggregator.feed(chunk)
        assert sent, 'an available sentence boundary must not wait for EOF/idle'
        await aggregator.flush()
        assert ''.join(sent) == ''.join(chunks)
    finally:
        await aggregator.aclose()


@pytest.mark.asyncio
async def test_sdk_filtered_stream_starts_before_last_sentence_arrives():
    sent = []
    chunks = ['费用取决于具体实现方式。', '若使用云端服务，', '按用量计费。']

    async def emit(text):
        sent.append(text)

    async def source():
        yield chunks[0]
        yield chunks[1]
        assert sent, 'SDK filter moves punctuation to the start of the next chunk'
        yield chunks[2]

    aggregator = SentenceAggregator(emit, first_sentence_flush_any_punct=True)
    try:
        async for chunk in filter_markdown(source()):
            await aggregator.feed(chunk)
        await aggregator.flush()
        assert ''.join(sent) == ''.join(chunks)
    finally:
        await aggregator.aclose()


@pytest.mark.asyncio
async def test_soft_boundary_keeps_minimum_and_first_segment_scope():
    sent = []

    async def emit(text):
        sent.append(text)

    aggregator = SentenceAggregator(emit, soft_min_chars=12, first_sentence_soft_min_chars=4)
    try:
        await aggregator.feed('你好，')
        await aggregator.feed('接下来继续说明')
        assert sent == [], 'a short prefix comma does not gain length from later text'
        await aggregator.feed('，开始')
        assert len(sent) == 1
        await aggregator.feed('收到，继续')
        assert len(sent) == 1, 'later segments retain the regular minimum'
        await aggregator.flush()
        assert ''.join(sent) == '你好，接下来继续说明，开始收到，继续'
    finally:
        await aggregator.aclose()


@pytest.mark.asyncio
async def test_idle_and_explicit_flush_do_not_repeat_emitted_text():
    sent = []
    emitted = asyncio.Event()

    async def emit(text):
        sent.append(text)
        emitted.set()

    aggregator = SentenceAggregator(emit, idle_ms=20)
    try:
        await aggregator.feed('已完成。余下内容')
        assert sent == ['已完成。余下内容']
        emitted.clear()
        await aggregator.feed('还有尾段')
        await asyncio.wait_for(emitted.wait(), timeout=1)
        await aggregator.flush()
        assert sent == ['已完成。余下内容', '还有尾段']
    finally:
        await aggregator.aclose()


@pytest.mark.asyncio
async def test_idle_flush_finishes_async_send_before_next_segment():
    """A stalled LLM must still deliver its text through a yielding transport."""
    entered, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def emit(text):
        if text == '首段内容':
            entered.set()
            await release.wait()
        sent.append(text)

    aggregator = SentenceAggregator(emit, idle_ms=20)
    next_segment = None
    try:
        await aggregator.feed('首段内容')
        await asyncio.wait_for(entered.wait(), timeout=1)
        next_segment = asyncio.create_task(aggregator.feed('随后一句。'))
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(next_segment, timeout=1)
        await aggregator.flush()
        assert sent == ['首段内容', '随后一句。']
    finally:
        release.set()
        if next_segment is not None:
            await asyncio.gather(next_segment, return_exceptions=True)
        await aggregator.aclose()


@pytest.mark.asyncio
async def test_close_cancels_inflight_idle_send():
    """An interruption must retain ownership of a blocked transport send."""
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def emit(text):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    aggregator = SentenceAggregator(emit, idle_ms=20)
    try:
        await aggregator.feed('未完成发送的内容')
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not cancelled.is_set(), 'idle flush cancelled its own send'
    finally:
        await asyncio.wait_for(aggregator.aclose(), timeout=1)
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('chunk_size', [1, 17, 1000])
@pytest.mark.parametrize('text', [
    '先说明背景，然后给出例子。最后总结这个方法。' * 20,
    '这是一个没有标点的很长回复' * 25,
])
async def test_large_provider_chunks_respect_batch_ceiling_without_losing_tail(text, chunk_size):
    sent = []

    async def emit(part):
        sent.append(part)

    aggregator = SentenceAggregator(emit, hard_max_chars=60)
    try:
        for offset in range(0, len(text), chunk_size):
            await aggregator.feed(text[offset:offset + chunk_size])
        await aggregator.flush()
        assert ''.join(sent) == text
        assert all(0 < len(part) <= 60 for part in sent)
    finally:
        await aggregator.aclose()
