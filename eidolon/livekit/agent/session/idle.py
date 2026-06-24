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
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        disconnect_grace_sec: float = 0.3,
        is_half_duplex: Callable[[], bool] | None = None,
        idle_end_reason: str = "idle_normal_end",
        keep_alive_half_duplex: bool = True,
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
        # Half-duplex (push-to-talk) clients are persistent appliances: the mic
        # is closed between holds, so silence is the normal resting state, not an
        # abandoned session. Idle-disconnecting them would force a reconnect (and
        # replay the welcome) on the next hold.
        self._is_half_duplex = is_half_duplex
        # The session_end reason emitted on idle teardown — idle_normal_end for a
        # user_initiated session, proactive_done for a proactive wake-up (§3.2).
        self._idle_end_reason = idle_end_reason
        # Whether the half_duplex keep-alive exemption applies (plan §3.3): a
        # user_initiated half_duplex appliance is kept alive; a proactive_initiated
        # session is NOT — it must be reclaimed on its short window even on a PTT
        # device. So the exemption is "half_duplex AND user_initiated".
        self._keep_alive_half_duplex = keep_alive_half_duplex
        self.task: asyncio.Task | None = None
        self.last_activity_monotonic: float = 0.0

    def mark_activity(self) -> None:
        """Record that the session is doing real work right now."""
        self.last_activity_monotonic = time.monotonic()

    def _room_name(self) -> str | None:
        room = self._get_room()
        return getattr(room, "name", None) if room else None

    def _half_duplex(self) -> bool:
        return self._is_half_duplex is not None and self._is_half_duplex()

    def start(self) -> None:
        if self.timeout_sec <= 0:
            logger.info(
                "[lifecycle][IdleWatchdog] disabled (disconnect_after_idle_ms<=0) room=%s",
                self._room_name(),
            )
            return
        self.mark_activity()
        self.task = asyncio.create_task(self.run())
        # half_duplex==True means this device is exempt from idle-disconnect (it is
        # a persistent push-to-talk appliance) — so for those clients the welcome
        # is NEVER followed by an idle room-delete. This line lets Phase 0 confirm
        # which regime a given session is in. See run()'s keep-alive branch.
        logger.info(
            "[lifecycle][IdleWatchdog] armed room=%s timeout=%.0fs half_duplex=%s",
            self._room_name(),
            self.timeout_sec,
            self._half_duplex(),
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
                if (
                    self._keep_alive_half_duplex
                    and self._is_half_duplex is not None
                    and self._is_half_duplex()
                ):
                    # Persistent push-to-talk appliance on a user_initiated session:
                    # stay connected so the next hold is instant and the welcome
                    # isn't replayed. A proactive_initiated session disables this
                    # (keep_alive_half_duplex=False) so an unanswered wake-up is
                    # still reclaimed on its short window (plan §3.3).
                    logger.info(
                        "[lifecycle][IdleWatchdog] idle %.0fs >= %.0fs but half_duplex "
                        "+ user_initiated; keeping room=%s alive (NOT deleting)",
                        elapsed, timeout, self._room_name(),
                    )
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
                {"type": "session_end", "reason": reason}
            ).encode("utf-8")
            await local.publish_data(
                payload, reliable=True, topic="eidolon.session_control"
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
