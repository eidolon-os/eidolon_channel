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
