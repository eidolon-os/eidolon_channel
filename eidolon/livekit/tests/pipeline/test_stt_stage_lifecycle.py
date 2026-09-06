"""A finished PTT transcript must not wait for transport teardown."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from livekit.agents import stt

from eidolon.livekit.agent.providers.stt import SttStage


class ControlledStream:
    def __init__(self, text, *, error=None):
        self.text = text
        self.error = error
        self.input_ended = asyncio.Event()
        self.result_ready = asyncio.Event()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.closed = False
        self.close_calls = 0

    def push_frame(self, frame):
        assert not self.input_ended.is_set()

    def end_input(self):
        self.input_ended.set()

    async def __aiter__(self):
        await self.result_ready.wait()
        if self.error:
            raise self.error
        yield stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language='zh', text=self.text)])
        yield stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        await self.allow_close.wait()
        self.closed = True


def stage_for(*streams):
    created = []
    def stream():
        value = streams[len(created)]
        created.append(value)
        return value
    plugin = SimpleNamespace(stream=stream, shutdown=AsyncMock())
    return SttStage(plugin), plugin, created


@pytest.mark.asyncio
async def test_result_precedes_close_but_next_recognition_and_shutdown_drain_it():
    first, second = ControlledStream('第一轮'), ControlledStream('第二轮')
    stage, plugin, created = stage_for(first, second)
    first.result_ready.set()
    second.result_ready.set()
    task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
    next_task = shutdown = None
    try:
        await asyncio.wait_for(first.close_started.wait(), 1)
        # Shield keeps a regression timeout from cancelling the operation.
        assert await asyncio.wait_for(asyncio.shield(task), .2) == '第一轮'
        assert not first.closed
        next_task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
        await asyncio.sleep(.02)
        assert created == [first], 'do not reuse a provider before prior cleanup'
        first.allow_close.set()
        assert await asyncio.wait_for(next_task, 1) == '第二轮'
        await asyncio.wait_for(second.close_started.wait(), 1)
        shutdown = asyncio.create_task(stage.shutdown())
        await asyncio.sleep(.02)
        assert not shutdown.done()
        plugin.shutdown.assert_not_awaited()
        second.allow_close.set()
        await asyncio.wait_for(shutdown, 1)
        assert first.closed and second.closed
        assert first.close_calls == second.close_calls == 1
        plugin.shutdown.assert_awaited_once()
        with pytest.raises(RuntimeError, match='shut down'):
            await stage.recognize_streaming(bytes(3200))
    finally:
        first.allow_close.set()
        second.allow_close.set()
        await asyncio.gather(*(t for t in (task, next_task, shutdown) if t), return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_failed_or_cancelled_recognition_closes_before_propagating(cancel):
    stream = ControlledStream('不能提交', error=ValueError('recognition failed'))
    stage, _, _ = stage_for(stream)
    task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
    try:
        await asyncio.wait_for(stream.input_ended.wait(), 1)
        if cancel:
            task.cancel()
        else:
            stream.result_ready.set()
        await asyncio.wait_for(stream.close_started.wait(), 1)
        assert not task.done()
        stream.allow_close.set()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
        assert stream.closed and stream.close_calls == 1
    finally:
        stream.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
async def test_cancelled_next_request_cannot_cancel_previous_cleanup():
    stream = ControlledStream('第一轮')
    stream.result_ready.set()
    stage, _, created = stage_for(stream)
    first = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
    next_task = None
    try:
        await asyncio.wait_for(stream.close_started.wait(), 1)
        assert await asyncio.wait_for(asyncio.shield(first), .2) == '第一轮'
        next_task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
        await asyncio.sleep(.02)
        next_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_task
        assert created == [stream]
        stream.allow_close.set()
        await stage.shutdown()
        assert stream.closed and stream.close_calls == 1
    finally:
        stream.allow_close.set()
        await asyncio.gather(*(t for t in (first, next_task) if t), return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_recognition_and_its_cleanup():
    stream = ControlledStream('完整结果')
    stage, plugin, _ = stage_for(stream)
    task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
    shutdown = None
    try:
        await asyncio.wait_for(stream.input_ended.wait(), 1)
        shutdown = asyncio.create_task(stage.shutdown())
        await asyncio.sleep(.02)
        plugin.shutdown.assert_not_awaited()
        stream.result_ready.set()
        await asyncio.wait_for(stream.close_started.wait(), 1)
        assert not shutdown.done()
        stream.allow_close.set()
        assert await task == '完整结果'
        await asyncio.wait_for(shutdown, 1)
        assert stream.closed
    finally:
        stream.result_ready.set()
        stream.allow_close.set()
        await asyncio.gather(*(t for t in (task, shutdown) if t), return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
async def test_offline_recognition_also_waits_for_previous_stream_cleanup():
    stream = ControlledStream('流式结果')
    stream.result_ready.set()
    stage, plugin, _ = stage_for(stream)
    plugin.recognize = AsyncMock(return_value=stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language='zh', text='离线结果')]))
    offline = None
    try:
        assert await stage.recognize_streaming(bytes(3200)) == '流式结果'
        await asyncio.wait_for(stream.close_started.wait(), 1)
        offline = asyncio.create_task(stage.recognize(bytes(3200)))
        await asyncio.sleep(.02)
        plugin.recognize.assert_not_awaited()
        stream.allow_close.set()
        assert await asyncio.wait_for(offline, 1) == '离线结果'
    finally:
        stream.allow_close.set()
        if offline:
            await asyncio.gather(offline, return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
async def test_repeated_cancellation_and_cancelled_shutdown_keep_cleanup_owned():
    stream = ControlledStream('不能提交')
    stage, plugin, _ = stage_for(stream)
    task = asyncio.create_task(stage.recognize_streaming(bytes(3200)))
    shutdown = None
    try:
        await asyncio.wait_for(stream.input_ended.wait(), 1)
        task.cancel()
        await asyncio.wait_for(stream.close_started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        shutdown = asyncio.create_task(stage.shutdown())
        await asyncio.sleep(.02)
        shutdown.cancel()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        plugin.shutdown.assert_not_awaited()
        stream.allow_close.set()
        await stage.shutdown()
        assert stream.closed and stream.close_calls == 1
        plugin.shutdown.assert_awaited_once()
    finally:
        stream.allow_close.set()
        await asyncio.gather(*(t for t in (task, shutdown) if t), return_exceptions=True)
        await stage.shutdown()


@pytest.mark.asyncio
async def test_close_failure_is_observed_and_does_not_replace_recognition_error(caplog):
    stream = ControlledStream('不能提交', error=ValueError('recognition failed'))
    stream.result_ready.set()
    stream.aclose = AsyncMock(side_effect=OSError('transport close failed'))
    stage, plugin, _ = stage_for(stream)
    with pytest.raises(ValueError, match='recognition failed'):
        await stage.recognize_streaming(bytes(3200))
    await stage.shutdown()
    stream.aclose.assert_awaited_once()
    plugin.shutdown.assert_awaited_once()
    assert 'STT stream close failed' in caplog.text
