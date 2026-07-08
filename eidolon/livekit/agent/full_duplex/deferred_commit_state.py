"""Runtime state adapter for deferred low-EOT commit tasks."""

from __future__ import annotations

import asyncio
from typing import Any


class FullDuplexDeferredCommitState:
    """Own the deferred low-EOT commit task reference on the pipeline."""

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def current(self) -> asyncio.Task | None:
        return getattr(self._pipeline, "_deferred_low_eot_commit_task", None)

    def replace(self, task: asyncio.Task) -> None:
        self._pipeline._deferred_low_eot_commit_task = task

    def cancel(self) -> bool:
        task = self.current()
        if task is None or task.done():
            self.clear()
            return False
        task.cancel()
        self.clear()
        return True

    def clear(self) -> None:
        self._pipeline._deferred_low_eot_commit_task = None

    def clear_if_current(self, task: asyncio.Task | None = None) -> None:
        current = task or asyncio.current_task()
        if self.current() is current:
            self.clear()
