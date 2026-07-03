"""Idle-disconnect watchdog for a LiveKit streaming session."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from eidolon_sdk.biz.contracts import (
    SESSION_CONTROL_TOPIC,
    SESSION_END_IDLE_NORMAL,
    SESSION_END_TYPE,
    WIRE_SCHEMA_VERSION,
)

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
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        disconnect_grace_sec: float = 0.3,
        idle_end_reason: str = SESSION_END_IDLE_NORMAL,
    ) -> None:
        self.timeout_sec = timeout_sec
        self._get_session = get_session
        self._get_room = get_room
        self._get_timeline = get_timeline
        self._session_closed_event = session_closed_event
        self._on_idle_disconnect = on_idle_disconnect
        # Centralised session_end{reason} publisher (server.py). When provided, the
        # watchdog routes its "the room is going away" notice through it with
        # ``idle_end_reason``, so every teardown path shares one idempotent
        # session_end emitter and one reason taxonomy. Falls back to a direct
        # publish when absent (keeps the unit-level watchdog usable standalone).
        self._on_session_end = on_session_end
        self.disconnect_grace_sec = disconnect_grace_sec
        # The session_end reason emitted on idle teardown — idle_normal_end for a
        # user_initiated session, proactive_done for a proactive wake-up (§3.2).
        self._idle_end_reason = idle_end_reason
        self.task: asyncio.Task | None = None
        self.last_activity_monotonic: float = 0.0

    def mark_activity(self) -> None:
        """Record that the session is doing real work right now."""
        self.last_activity_monotonic = time.monotonic()

    def _room_name(self) -> str | None:
        room = self._get_room()
        return getattr(room, "name", None) if room else None

    def start(self) -> None:
        if self.timeout_sec <= 0:
            logger.info(
                "[lifecycle][IdleWatchdog] disabled (disconnect_after_idle_ms<=0) room=%s",
                self._room_name(),
            )
            return
        self.mark_activity()
        self.task = asyncio.create_task(self.run())
        logger.info(
            "[lifecycle][IdleWatchdog] armed room=%s timeout=%.0fs",
            self._room_name(),
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
                    "[lifecycle][IdleWatchdog] idle %.0fs >= %.0fs; disconnecting room=%s "
                    "(reason=%s)",
                    elapsed, timeout, self._room_name(), self._idle_end_reason,
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
        """Tell the client the session is ending (idle) before the room is deleted.

        An idle disconnect is a *normal* end of conversation, not a join failure —
        the client must be able to tell the difference (plan §3.2). We therefore
        emit session_end{reason=<idle_end_reason>} (idle_normal_end for a user
        session, proactive_done for a proactive wake-up) rather than a bare
        "kicked" signal. Routed through the shared publisher when one is injected
        so the reason taxonomy and idempotency live in one place (server.py).
        """
        reason = self._idle_end_reason
        if self._on_session_end is not None:
            try:
                await self._on_session_end(reason)
            except Exception:
                logger.debug(
                    "[IdleWatchdog] on_session_end(%s) failed", reason, exc_info=True
                )
            return
        room = self._get_room()
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            payload = json.dumps(
                {
                    "schema_v": WIRE_SCHEMA_VERSION,
                    "type": SESSION_END_TYPE,
                    "reason": reason,
                }
            ).encode("utf-8")
            await local.publish_data(
                payload, reliable=True, topic=SESSION_CONTROL_TOPIC
            )
            logger.info(
                "[lifecycle][IdleWatchdog] sent session_end reason=%s "
                "room=%s (grace=%.2fs before delete)",
                reason,
                self._room_name(),
                self.disconnect_grace_sec,
            )
        except Exception:
            logger.debug(
                "[IdleWatchdog] failed to send session_end(%s)", reason, exc_info=True
            )
