"""Interruption side-effect helpers for a LiveKit streaming session."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger("agent.session.interruption")


class SoftInterruptController:
    """Fallback soft-interrupt timer.

    This does not decide whether a transcript is an interrupt. It only owns the
    small side-effect state machine for the legacy fallback path:

    - enter a pending soft interrupt,
    - cancel it when the signal is resolved as false,
    - upgrade to a hard interrupt when the timer expires.
    """

    def __init__(
        self,
        *,
        timeout_sec: float,
        on_timeout: Callable[[], None],
    ) -> None:
        self.timeout_sec = timeout_sec
        self._on_timeout = on_timeout
        self.active = False
        self.task: asyncio.Task | None = None

    def enter(self) -> None:
        self.active = True
        self.task = asyncio.create_task(self.run_timeout_task())
        logger.info(
            "[SoftInterruptController] entered (timeout=%.1fs)",
            self.timeout_sec,
        )

    def cancel(self) -> None:
        self.active = False
        current = asyncio.current_task()
        if self.task and self.task is not current:
            self.task.cancel()
        self.task = None
        logger.info("[SoftInterruptController] cancelled (false interruption)")

    async def run_timeout_task(self) -> None:
        try:
            await asyncio.sleep(self.timeout_sec)
            if self.active:
                logger.info(
                    "[SoftInterruptController] timeout; upgrading to hard interrupt"
                )
                self.cancel()
                self._on_timeout()
        except asyncio.CancelledError:
            pass
