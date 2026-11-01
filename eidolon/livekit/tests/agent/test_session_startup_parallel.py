import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.shared.pipeline import BasePipeline
from eidolon.livekit.agent.session import room_io


def _pipeline(warmup):
    pipeline = SimpleNamespace(
        _warmup_stages=warmup, _filler=None, session_mark=MagicMock(), _warmup_task=None,
        _lifecycle_stages=lambda: [],
    )
    for name in ("_begin_stage_warmup", "_end_stage_warmup", "_shutdown_stages"):
        setattr(pipeline, name, MethodType(getattr(BasePipeline, name), pipeline))
    return pipeline


def _marks(pipeline):
    return {c.args[0] for c in pipeline.session_mark.call_args_list}


@pytest.mark.asyncio
async def test_the_session_is_confirmed_once_media_is_up_while_warmup_continues(monkeypatch):
    # session_started waited for every warmup, the remote interrupt classifier
    # last: ~0.4 s median in front of each conversation (11-01).
    warm_started, release = asyncio.Event(), asyncio.Event()

    async def warmup():
        warm_started.set()
        await release.wait()

    async def media(**kwargs):
        await warm_started.wait()
        return "presentation-io"

    pipeline = _pipeline(warmup)
    monkeypatch.setattr(room_io, "start_room_session", media)
    await asyncio.wait_for(FullDuplexSessionLifecycle(pipeline)._prepare_session(), 1)
    assert pipeline._presentation_io == "presentation-io"
    assert _marks(pipeline) == {"media_started"}
    assert not pipeline._warmup_task.done()
    release.set()
    await asyncio.wait_for(pipeline._warmup_task, 1)
    assert _marks(pipeline) == {"media_started", "warmup_done"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["media", "cancel"])
async def test_a_failed_or_cancelled_start_stops_the_warmup(monkeypatch, failure):
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def warmup():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    async def media(**kwargs):
        await entered.wait()
        if failure == "media":
            raise ValueError("media")
        await asyncio.Future()

    pipeline = _pipeline(warmup)
    monkeypatch.setattr(room_io, "start_room_session", media)
    task = asyncio.create_task(FullDuplexSessionLifecycle(pipeline)._prepare_session())
    if failure == "cancel":
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(ValueError):
            await asyncio.wait_for(task, 1)
    assert stopped.is_set()
    assert pipeline._warmup_task is None


@pytest.mark.asyncio
async def test_a_failing_warmup_does_not_fail_the_start(monkeypatch, caplog):
    async def warmup():
        raise RuntimeError("classifier unreachable")

    async def media(**kwargs):
        await asyncio.sleep(0)
        return "presentation-io"

    pipeline = _pipeline(warmup)
    monkeypatch.setattr(room_io, "start_room_session", media)
    await asyncio.wait_for(FullDuplexSessionLifecycle(pipeline)._prepare_session(), 1)
    await asyncio.wait({pipeline._warmup_task})
    await asyncio.sleep(0)
    assert pipeline._presentation_io == "presentation-io"
    assert "stage warmup failed" in caplog.text


@pytest.mark.asyncio
async def test_stage_shutdown_stops_a_warmup_still_running():
    stopped = asyncio.Event()

    async def warmup():
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    pipeline = _pipeline(warmup)
    pipeline._begin_stage_warmup()
    await asyncio.sleep(0)
    await asyncio.wait_for(pipeline._shutdown_stages(), 1)
    assert stopped.is_set() and pipeline._warmup_task is None
    await pipeline._end_stage_warmup()  # idempotent


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
