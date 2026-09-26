import asyncio
from unittest.mock import AsyncMock

import pytest
from livekit.agents import Agent
from livekit.agents.voice import AgentSession
from eidolon_sdk.biz.control.coordination_stream import ReplyStart
from eidolon.livekit.agent.coordination.native_output import NativeSpeechPresenter, SpeechEndpoint
from .._harness.mocks.mock_tts import MockTTS
from .._harness.headless import RecordingAudioOutput


def request():
    return ReplyStart(type='reply_start', session_id='scene', stream_id='stream',
        request_id='turn', turn_id='turn', epoch=1, companion_id='a', device_id='device')


async def words():
    yield '你好，'
    yield '我是被选中的角色。'


@pytest.mark.asyncio
async def test_native_session_synthesizes_without_llm_and_waits_for_device():
    confirmed = asyncio.Event()
    host_done = asyncio.Event()

    async def confirm(start):
        assert start.turn_id == 'turn'
        host_done.set()
        await confirmed.wait()
        return True

    session = AgentSession()
    sink = RecordingAudioOutput()
    session.output.audio = sink
    await session.start(Agent(instructions='', tts=MockTTS(), llm=None, stt=None))
    presenter = NativeSpeechPresenter({('a', 'device'): SpeechEndpoint(session, confirm)})
    task = asyncio.create_task(presenter(request(), words(), lambda: None))
    try:
        await asyncio.wait_for(host_done.wait(), 5)
        assert sink.collected_pcm
        assert not task.done()  # Host queue completion is not device completion.
        confirmed.set()
        assert await asyncio.wait_for(task, 2)
        assert not any(item.type == "message" for item in session.history.items)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await session.aclose()


@pytest.mark.asyncio
async def test_cancel_interrupts_native_speech_without_confirming():
    started = asyncio.Event()

    async def blocked_text():
        yield '这段播放应该被打断。'
        started.set()
        await asyncio.Event().wait()

    confirm = AsyncMock(return_value=True)
    session = AgentSession()
    session.output.audio = RecordingAudioOutput()
    await session.start(Agent(instructions='', tts=MockTTS(), llm=None, stt=None))
    presenter = NativeSpeechPresenter({('a', 'device'): SpeechEndpoint(session, confirm)})
    task = asyncio.create_task(presenter(request(), blocked_text(), lambda: None))
    try:
        await asyncio.wait_for(started.wait(), 5)
        handle = session.current_speech
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert handle.interrupted
        confirm.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_unselected_pair_never_calls_session():
    presenter = NativeSpeechPresenter({})
    with pytest.raises(KeyError):
        await presenter(request(), words(), lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize('broken', [False, True])
async def test_empty_or_failed_text_never_confirms_playback(broken):
    async def invalid_text():
        if broken:
            yield '未完成的内容'
            raise RuntimeError('upstream generation failed')
        yield ''

    confirm = AsyncMock(return_value=True)
    session = AgentSession()
    session.output.audio = RecordingAudioOutput()
    await session.start(Agent(instructions='', tts=MockTTS(), llm=None, stt=None))
    presenter = NativeSpeechPresenter({('a', 'device'): SpeechEndpoint(session, confirm)})
    try:
        assert not await asyncio.wait_for(presenter(request(), invalid_text(), lambda: None), 5)
        confirm.assert_not_called()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_native_completion_sequences_without_claiming_device_ack():
    session = AgentSession()
    sink = RecordingAudioOutput()
    session.output.audio = sink
    await session.start(Agent(instructions='', tts=MockTTS(), llm=None, stt=None))
    presenter = NativeSpeechPresenter({('a', 'device'): SpeechEndpoint(session)})
    try:
        assert await asyncio.wait_for(presenter(request(), words(), lambda: None), 5)
        assert sink.collected_pcm
    finally:
        await session.aclose()
