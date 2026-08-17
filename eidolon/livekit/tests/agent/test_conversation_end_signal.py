"""A conversation ends when the channel drops us, not when the framework agrees.

Measured against a real backend after a session was withdrawn: the room reports
"disconnected" 3.01s later, the framework cancels the job's entrypoint 15.01s
later. The pipeline used to wait only for the framework, so every session paid
for ~12 extra seconds of streaming speech recognition — plus a warm TTS pool and
a held-open brain connection — for a conversation that was already over.

The half-duplex pipeline already raced both signals; these tests pin the same
contract for the streaming pipeline, which is the path every non-PTT device
takes.
"""

from __future__ import annotations

import asyncio

import pytest

from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle


def _lifecycle() -> tuple[FullDuplexSessionLifecycle, StreamingPipeline]:
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session_closed_event = asyncio.Event()
    pipeline._room_disconnected_event = asyncio.Event()
    return FullDuplexSessionLifecycle(pipeline), pipeline


@pytest.mark.asyncio
async def test_a_dropped_channel_ends_the_conversation() -> None:
    lifecycle, pipeline = _lifecycle()

    lifecycle._on_room_disconnected("SERVER_SHUTDOWN")

    assert pipeline._room_disconnected_event.is_set()
    assert await lifecycle._await_conversation_end() is False


@pytest.mark.asyncio
async def test_a_closed_session_ends_the_conversation() -> None:
    lifecycle, pipeline = _lifecycle()

    pipeline._session_closed_event.set()

    assert await lifecycle._await_conversation_end() is True


@pytest.mark.asyncio
async def test_the_channel_dropping_does_not_wait_for_the_framework() -> None:
    """The whole point: neither signal is allowed to gate the other."""
    lifecycle, pipeline = _lifecycle()
    waiting = asyncio.create_task(lifecycle._await_conversation_end())
    await asyncio.sleep(0)
    assert not waiting.done()

    lifecycle._on_room_disconnected(None)

    # The framework's close never arrives, and that must not matter.
    assert await asyncio.wait_for(waiting, timeout=1.0) is False
    assert not pipeline._session_closed_event.is_set()


@pytest.mark.asyncio
async def test_waiting_leaves_no_task_behind() -> None:
    """A losing waiter left running would outlive the job it belonged to."""
    lifecycle, pipeline = _lifecycle()
    before = len(asyncio.all_tasks())

    pipeline._session_closed_event.set()
    await lifecycle._await_conversation_end()
    await asyncio.sleep(0)

    assert len(asyncio.all_tasks()) <= before


@pytest.mark.asyncio
async def test_both_signals_together_prefer_speaking_to_the_device() -> None:
    """When the session closed cleanly the device is still there to be told."""
    lifecycle, pipeline = _lifecycle()

    pipeline._session_closed_event.set()
    pipeline._room_disconnected_event.set()

    assert await lifecycle._await_conversation_end() is True
