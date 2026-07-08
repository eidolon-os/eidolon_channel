import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.full_duplex.deferred_commit_state import (
    FullDuplexDeferredCommitState,
)


def test_deferred_commit_state_replace_and_clear() -> None:
    owner = SimpleNamespace()
    state = FullDuplexDeferredCommitState(owner)
    task = SimpleNamespace()

    assert state.current() is None
    state.replace(task)
    assert state.current() is task

    state.clear()
    assert state.current() is None


@pytest.mark.asyncio
async def test_deferred_commit_state_cancels_pending_task() -> None:
    owner = SimpleNamespace()
    state = FullDuplexDeferredCommitState(owner)
    task = asyncio.create_task(asyncio.sleep(60))

    state.replace(task)
    assert state.cancel() is True

    assert state.current() is None
    assert task.cancelled() or task.cancelling()
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_deferred_commit_state_clears_done_task_without_cancel() -> None:
    owner = SimpleNamespace()
    state = FullDuplexDeferredCommitState(owner)
    task = asyncio.create_task(asyncio.sleep(0))
    await task

    state.replace(task)
    assert state.cancel() is False

    assert state.current() is None
    assert task.cancelled() is False


@pytest.mark.asyncio
async def test_deferred_commit_state_clear_if_current_only_clears_matching_task() -> None:
    owner = SimpleNamespace()
    state = FullDuplexDeferredCommitState(owner)
    first = asyncio.create_task(asyncio.sleep(0))
    second = asyncio.create_task(asyncio.sleep(0))

    state.replace(first)
    state.clear_if_current(second)
    assert state.current() is first

    state.clear_if_current(first)
    assert state.current() is None

    await first
    await second
