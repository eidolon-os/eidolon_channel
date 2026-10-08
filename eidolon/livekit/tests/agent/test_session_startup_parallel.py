import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.session import room_io


@pytest.mark.asyncio
async def test_media_and_prewarm_overlap_and_both_finish(monkeypatch):
    media_started, warm_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def warmup():
        warm_started.set()
        await media_started.wait()
        await release.wait()

    async def media(**kwargs):
        media_started.set()
        await warm_started.wait()
        return "presentation-io"

    pipeline = SimpleNamespace(_warmup_stages=warmup, _filler=None, session_mark=MagicMock())
    monkeypatch.setattr(room_io, "start_room_session", media)
    task = asyncio.create_task(FullDuplexSessionLifecycle(pipeline)._prepare_session())
    await asyncio.wait_for(media_started.wait(), 1)
    await asyncio.wait_for(warm_started.wait(), 1)
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, 1)
    assert pipeline._presentation_io == "presentation-io"
    assert {c.args[0] for c in pipeline.session_mark.call_args_list} == {"media_started", "warmup_done"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["warmup", "media", "cancel"])
async def test_startup_failure_or_cancellation_joins_sibling(monkeypatch, failure):
    entered = {key: asyncio.Event() for key in ("warmup", "media")}
    finished = set()

    async def stage(name):
        entered[name].set()
        try:
            await entered["media" if name == "warmup" else "warmup"].wait()
            if failure == name:
                raise ValueError(name)
            await asyncio.Future()
        finally:
            finished.add(name)

    async def media(**kwargs):
        return await stage("media")

    pipeline = SimpleNamespace(_warmup_stages=lambda: stage("warmup"), _filler=None, session_mark=MagicMock())
    monkeypatch.setattr(room_io, "start_room_session", media)
    task = asyncio.create_task(FullDuplexSessionLifecycle(pipeline)._prepare_session())
    if failure == "cancel":
        await asyncio.wait_for(entered["media"].wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ExceptionGroup) as error:
            await asyncio.wait_for(task, 1)
        assert isinstance(error.value.exceptions[0], ValueError)
    assert finished == {"warmup", "media"}


@pytest.mark.asyncio
async def test_welcome_waits_for_session_confirmation():
    from eidolon_sdk.biz.presentation import OutputSelection
    from eidolon.livekit.agent.full_duplex.agent_builder import build_full_duplex_agent

    ready = asyncio.Event()
    welcome = MagicMock(return_value=None)
    pipeline = SimpleNamespace(
        _turn_detection=lambda: "manual", _instructions="",
        _factory=SimpleNamespace(outputs=OutputSelection(),
            stt=SimpleNamespace(stt=None), llm=SimpleNamespace(llm=None), tts=None, vad=None),
        _welcome_on_enter=welcome, _is_proactive=False,
    )
    agent = build_full_duplex_agent(pipeline, ready=ready)
    task = asyncio.create_task(agent.on_enter())
    await asyncio.sleep(0)
    assert not task.done()
    welcome.assert_not_called()
    ready.set()
    await asyncio.wait_for(task, 1)
    welcome.assert_called_once()
