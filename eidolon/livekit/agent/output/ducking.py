"""Output ducking side-effect controller.

This module owns the low-level ``OutputController`` state transitions. Higher
layers still decide *when* to cancel, rollback, or hold; this class only knows
how to install the output middleware and apply the requested output action.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .controller import OutputController

if TYPE_CHECKING:
    from livekit.agents.voice import AgentSession

logger = logging.getLogger("agent.output.ducking")


@dataclass(frozen=True)
class DuckingStats:
    """Snapshot of the current ducking window."""

    suspend_ms: float = 0.0
    buffered_frames: int = 0
    buffered_sec: float = 0.0


class OutputDuckingController:
    """Own ``OutputController`` installation and state transitions."""

    def __init__(self) -> None:
        self.mixer: OutputController | None = None
        self.timeout_task: asyncio.Task | None = None
        self.last_unduck_time: float = 0.0
        self.suspend_start: float = 0.0

    @property
    def installed(self) -> bool:
        return self.mixer is not None

    @property
    def is_suspended(self) -> bool:
        return self.mixer is not None and self.mixer.state == "SUSPENDED"

    @property
    def is_cancelled(self) -> bool:
        return self.mixer is not None and self.mixer.state == "CANCELLED"

    def install(self, session: "AgentSession", cfg: object) -> OutputController | None:
        """Install ``OutputController`` in the LiveKit audio output chain."""

        if not getattr(cfg, "duck_enabled", True):
            logger.info("[OutputDuckingController] duck_enabled=False; skipping")
            return None
        inner = session.output.audio
        if inner is None:
            logger.warning(
                "[OutputDuckingController] session.output.audio is None; "
                "ducking not installed"
            )
            return None

        mixer = OutputController(
            inner,
            fade_ms=cfg.duck_fade_ms,
            fade_in_ms=cfg.duck_fade_in_ms,
            suspend_volume=cfg.duck_suspend_volume,
            buffer_max_sec=cfg.duck_buffer_max_sec,
        )
        session.output.audio = mixer
        self.mixer = mixer
        return mixer

    def cancel_timeout(self) -> None:
        """Cancel the suspend-window fallback task if active."""

        if self.timeout_task is not None and not self.timeout_task.done():
            self.timeout_task.cancel()
        self.timeout_task = None

    def stats(self) -> DuckingStats:
        if self.mixer is None:
            return DuckingStats()
        return DuckingStats(
            suspend_ms=(time.monotonic() - self.suspend_start) * 1000,
            buffered_frames=self.mixer.buffered_frames,
            buffered_sec=self.mixer.buffered_sec,
        )

    def duck(self, *, now: float | None = None) -> None:
        if self.mixer is None:
            return
        self.suspend_start = time.monotonic() if now is None else now
        self.mixer.duck()

    def cancel_output(self) -> None:
        self.cancel_timeout()
        if self.mixer is not None:
            self.mixer.cancel()

    def unduck_if_suspended(self, *, drop_buffered: bool = False) -> bool:
        if self.mixer is None:
            return False
        self.cancel_timeout()
        if self.mixer.state != "SUSPENDED":
            return False
        self.mixer.unduck(drop_buffered=drop_buffered)
        self.last_unduck_time = time.monotonic()
        return True

    def on_agent_started_speaking(self) -> None:
        if self.mixer is None:
            return
        try:
            self.mixer.on_agent_started_speaking()
        except AttributeError:
            pass

    def reset_if_cancelled(self) -> bool:
        if self.mixer is None:
            return False
        try:
            if self.mixer.state == "CANCELLED":
                self.mixer.reset()
                return True
        except AttributeError:
            return False
        return False

    def get_metrics(self) -> dict[str, object] | None:
        if self.mixer is None:
            return None
        return self.mixer.get_metrics()
