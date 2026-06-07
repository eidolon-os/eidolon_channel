"""Idle-disconnect watchdog for a LiveKit streaming session."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.session.idle")


class IdleWatchdog:
    """Close an idle session after a configured period of no real activity."""

    def __init__(
        self,
        *,
        timeout_sec: float,
        get_session: Callable[[], Any | None],
        get_room: Callable[[], Any | None],
        get_timeline: Callable[[], TurnTimeline | None],
        session_closed_event: asyncio.Event,
        on_idle_disconnect: Callable[[], Awaitable[None]] | None = None,
        disconnect_grace_sec: float = 0.3,
    ) -> None:
        self.timeout_sec = timeout_sec
        self._get_session = get_session
        self._get_room = get_room
        self._get_timeline = get_timeline
        self._session_closed_event = session_closed_event
        self._on_idle_disconnect = on_idle_disconnect
        self.disconnect_grace_sec = disconnect_grace_sec
        self.task: asyncio.Task | None = None
        self.last_activity_monotonic: float = 0.0

    def mark_activity(self) -> None:
        """Record that the session is doing real work right now."""
        self.last_activity_monotonic = time.monotonic()

    def start(self) -> None:
        if self.timeout_sec <= 0:
            logger.info(
                "[IdleWatchdog] disabled (disconnect_after_idle_ms<=0)"
            )
            return
        self.mark_activity()
        self.task = asyncio.create_task(self.run())
        logger.info(
            "[IdleWatchdog] armed (timeout=%.0fs)",
            self.timeout_sec,
        )

    def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.task = None

    async def run(self) -> None:
        """Close the session once it has been idle past the configured timeout."""
        timeout = self.timeout_sec
        try:
            while not self._session_closed_event.is_set():
                elapsed = time.monotonic() - self.last_activity_monotonic
                remaining = timeout - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    continue
                session = self._get_session()
                if session is not None and (
                    session.agent_state in ("thinking", "speaking")
                    or session.user_state == "speaking"
                ):
                    self.mark_activity()
                    continue
                logger.info(
                    "[IdleWatchdog] idle for %.0fs >= %.0fs; disconnecting",
                    elapsed, timeout,
                )
                timeline = self._get_timeline()
                if timeline is not None:
                    timeline.mark("idle_timeout_triggered_at")
                await self.disconnect_idle()
                return
        except asyncio.CancelledError:
            pass

    async def disconnect_idle(self) -> None:
        """Notify the client, then disconnect the idle session."""
        await self.notify_client_idle_timeout()
        if self.disconnect_grace_sec > 0:
            await asyncio.sleep(self.disconnect_grace_sec)
        if self._on_idle_disconnect is not None:
            try:
                await self._on_idle_disconnect()
            except Exception:
                logger.exception("[IdleWatchdog] room-delete callback failed")
        else:
            session = self._get_session()
            if session is not None:
                try:
                    await session.aclose()
                except Exception:
                    logger.exception("[IdleWatchdog] error closing idle session")
        self._session_closed_event.set()

    async def notify_client_idle_timeout(self) -> None:
        """Best-effort: tell the client it is being dropped for inactivity."""
        room = self._get_room()
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            payload = json.dumps(
                {"type": "idle_timeout", "reason": "idle_timeout"}
            ).encode("utf-8")
            await local.publish_data(
                payload, reliable=True, topic="session_control"
            )
            logger.info("[IdleWatchdog] notified client of idle timeout")
        except Exception:
            logger.debug(
                "[IdleWatchdog] failed to notify client of idle timeout",
                exc_info=True,
            )
