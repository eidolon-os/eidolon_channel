"""IdleWatchdog tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session import IdleWatchdog


def _watchdog(
    *,
    timeout_sec: float = 0.05,
    session=None,
    room=None,
    timeline=None,
    on_idle_disconnect=None,
    on_session_end=None,
    is_half_duplex=None,
    idle_end_reason="idle_normal_end",
    keep_alive_half_duplex=True,
) -> tuple[IdleWatchdog, asyncio.Event]:
    closed = asyncio.Event()
    watchdog = IdleWatchdog(
        timeout_sec=timeout_sec,
        get_session=lambda: session,
        get_room=lambda: room,
        get_timeline=lambda: timeline,
        session_closed_event=closed,
        on_idle_disconnect=on_idle_disconnect,
        on_session_end=on_session_end,
        disconnect_grace_sec=0.0,
        is_half_duplex=is_half_duplex,
        idle_end_reason=idle_end_reason,
        keep_alive_half_duplex=keep_alive_half_duplex,
    )
    return watchdog, closed


@pytest.mark.asyncio
async def test_idle_watchdog_notifies_client_and_calls_disconnect_callback() -> None:
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    on_idle = AsyncMock()
    timeline = TurnTimeline("turn-idle")
    watchdog, closed = _watchdog(
        room=room,
        timeline=timeline,
        on_idle_disconnect=on_idle,
    )

    watchdog.start()
    await asyncio.wait_for(watchdog.task, timeout=2.0)

    on_idle.assert_awaited_once()
    room.local_participant.publish_data.assert_awaited_once()
    payload = room.local_participant.publish_data.await_args.args[0]
    # Idle disconnect is a normal end of conversation, not a join failure: the
    # client is told so via session_end{reason=idle_normal_end} (plan §3.2).
    assert json.loads(payload) == {
        "type": "session_end",
        "reason": "idle_normal_end",
    }
    assert timeline.snapshot()["timestamps"]["idle_timeout_triggered_at"]
    assert closed.is_set()


@pytest.mark.asyncio
async def test_idle_watchdog_routes_through_injected_session_end() -> None:
    """When a shared session_end publisher is injected, the watchdog routes its
    notice through it (reason=idle_normal_end) instead of publishing directly, so
    the reason taxonomy and idempotency live in one place (server.py)."""
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    on_session_end = AsyncMock()
    on_idle = AsyncMock()
    watchdog, closed = _watchdog(
        room=room,
        on_idle_disconnect=on_idle,
        on_session_end=on_session_end,
    )

    watchdog.start()
    await asyncio.wait_for(watchdog.task, timeout=2.0)

    on_session_end.assert_awaited_once_with("idle_normal_end")
    # Routed through the shared publisher, not the watchdog's direct fallback.
    room.local_participant.publish_data.assert_not_awaited()
    on_idle.assert_awaited_once()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_idle_watchdog_falls_back_to_session_close() -> None:
    session = SimpleNamespace(
        agent_state="idle",
        user_state="listening",
        aclose=AsyncMock(),
    )
    watchdog, closed = _watchdog(session=session)

    watchdog.start()
    await asyncio.wait_for(watchdog.task, timeout=2.0)

    session.aclose.assert_awaited_once()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_idle_watchdog_rearms_while_agent_is_active() -> None:
    session = SimpleNamespace(
        agent_state="speaking",
        user_state="listening",
        aclose=AsyncMock(),
    )
    watchdog, _closed = _watchdog(session=session)

    watchdog.start()
    await asyncio.sleep(0.2)

    assert watchdog.task is not None
    assert not watchdog.task.done()
    session.aclose.assert_not_awaited()

    session.agent_state = "idle"
    await asyncio.wait_for(watchdog.task, timeout=2.0)
    session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_proactive_half_duplex_session_is_not_kept_alive() -> None:
    """A proactive wake-up (keep_alive_half_duplex=False) must NOT get the
    half_duplex keep-alive exemption: an unanswered report is reclaimed on its
    short window with reason=proactive_done (plan §3.2/§3.3)."""
    session = SimpleNamespace(
        agent_state="idle", user_state="listening", aclose=AsyncMock()
    )
    on_session_end = AsyncMock()
    on_idle = AsyncMock()
    watchdog, closed = _watchdog(
        session=session,
        on_idle_disconnect=on_idle,
        on_session_end=on_session_end,
        is_half_duplex=lambda: True,          # PTT device …
        keep_alive_half_duplex=False,         # … but proactive: no exemption
        idle_end_reason="proactive_done",
    )

    watchdog.start()
    await asyncio.wait_for(watchdog.task, timeout=2.0)

    # It disconnected (not kept alive) and reported the proactive reason.
    on_idle.assert_awaited_once()
    on_session_end.assert_awaited_once_with("proactive_done")
    assert closed.is_set()


@pytest.mark.asyncio
async def test_idle_watchdog_does_not_disconnect_half_duplex_session() -> None:
    """Half-duplex (push-to-talk) appliance: silent between holds is normal, so
    the watchdog must keep the session alive instead of idle-disconnecting (which
    would force a reconnect + welcome replay on the next hold)."""
    session = SimpleNamespace(
        agent_state="idle",
        user_state="listening",
        aclose=AsyncMock(),
    )
    on_idle = AsyncMock()
    watchdog, closed = _watchdog(
        session=session,
        on_idle_disconnect=on_idle,
        is_half_duplex=lambda: True,
    )

    watchdog.start()
    await asyncio.sleep(0.2)

    assert watchdog.task is not None
    assert not watchdog.task.done()
    on_idle.assert_not_awaited()
    session.aclose.assert_not_awaited()
    assert not closed.is_set()
    watchdog.stop()
