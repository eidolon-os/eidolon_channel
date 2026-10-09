import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eidolon.livekit.agent.session.job_lifecycle import start_job_pipeline


@pytest.mark.asyncio
async def test_entrypoint_returns_before_session_close_and_cleanup_joins_pipeline():
    closed = asyncio.Event()
    drained = asyncio.Event()
    callbacks = []
    ctx = SimpleNamespace(shutdown=Mock(), add_shutdown_callback=callbacks.append)
    async def pipeline(ready):
        ready.set()
        await closed.wait()
        drained.set()
    # SDK waits for the entrypoint before closing the session. This must return
    # with the session still open, rather than depending on server eviction.
    await asyncio.wait_for(start_job_pipeline(ctx, pipeline), .5)
    assert not drained.is_set()
    closed.set()  # SDK can now close immediately on dispatch withdrawal.
    await callbacks[0]("dispatch withdrawn")
    assert drained.is_set()
    ctx.shutdown.assert_called_once_with(reason="pipeline completed")


@pytest.mark.asyncio
async def test_startup_failure_propagates_without_leaking_a_task():
    ctx = SimpleNamespace(shutdown=Mock(), add_shutdown_callback=Mock())
    async def pipeline(ready):
        raise ValueError("failed before ready")
    with pytest.raises(ValueError, match="failed before ready"):
        await start_job_pipeline(ctx, pipeline)
    ctx.add_shutdown_callback.assert_not_called()


@pytest.mark.asyncio
async def test_entrypoint_cancellation_cancels_startup():
    cancelled = asyncio.Event()
    async def pipeline(ready):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    task = asyncio.create_task(start_job_pipeline(SimpleNamespace(), pipeline))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
