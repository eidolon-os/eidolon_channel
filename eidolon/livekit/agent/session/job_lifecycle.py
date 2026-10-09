"""Transfer a started pipeline to the LiveKit job without blocking its entrypoint.

LiveKit waits for the entrypoint before closing its primary AgentSession. A
pipeline waiting for that close must therefore be owned by a shutdown callback,
not awaited for the entire conversation by the entrypoint itself.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any


logger = logging.getLogger(__name__)


async def start_job_pipeline(ctx: Any, run: Callable[[asyncio.Event], Awaitable[None]]) -> None:
    ready = asyncio.Event()
    task = asyncio.create_task(run(ready), name="eidolon-job-pipeline")
    ready_wait = asyncio.create_task(ready.wait())
    try:
        await asyncio.wait({task, ready_wait}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task  # Startup failures remain entrypoint failures.
            ctx.shutdown(reason="pipeline completed")
            return
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        ready_wait.cancel()
        await asyncio.gather(ready_wait, return_exceptions=True)

    def finished(completed: asyncio.Task[None]) -> None:
        if completed.cancelled():
            reason = "pipeline cancelled"
        elif (error := completed.exception()) is not None:
            logger.error("job pipeline failed", exc_info=(type(error), error, error.__traceback__))
            reason = "pipeline error"
        else:
            reason = "pipeline completed"
        ctx.shutdown(reason=reason)

    async def cleanup(_reason: str) -> None:
        # SDK has already closed its primary session at this point. Await all
        # application resources too; cancellation bounds failures in cleanup.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    ctx.add_shutdown_callback(cleanup)
    task.add_done_callback(finished)
