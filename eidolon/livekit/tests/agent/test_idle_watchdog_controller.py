"""IdleWatchdog tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from eidolon_sdk.biz.contracts import WIRE_SCHEMA_VERSION

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.idle import IdleWatchdog


def _watchdog(
    *,
    timeout_sec: float = 0.05,
    session=None,
    room=None,
    timeline=None,
    on_idle_disconnect=None,
    on_session_end=None,
    idle_end_reason="idle_normal_end",
    is_busy=None,
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
        idle_end_reason=idle_end_reason,
        is_busy=is_busy,
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
        "schema_v": WIRE_SCHEMA_VERSION,
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
async def test_idle_watchdog_uses_injected_busy_guard() -> None:
    session = SimpleNamespace(
        agent_state="idle",
        user_state="listening",
        aclose=AsyncMock(),
    )
    busy = True
    watchdog, _closed = _watchdog(session=session, is_busy=lambda: busy)

    watchdog.start()
    await asyncio.sleep(0.2)

    assert watchdog.task is not None
    assert not watchdog.task.done()
    session.aclose.assert_not_awaited()

    busy = False
    await asyncio.wait_for(watchdog.task, timeout=2.0)
    session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_idle_watchdog_uses_injected_proactive_reason() -> None:
    session = SimpleNamespace(
        agent_state="idle", user_state="listening", aclose=AsyncMock()
    )
    on_session_end = AsyncMock()
    on_idle = AsyncMock()
    watchdog, closed = _watchdog(
        session=session,
        on_idle_disconnect=on_idle,
        on_session_end=on_session_end,
        idle_end_reason="proactive_done",
    )

    watchdog.start()
    await asyncio.wait_for(watchdog.task, timeout=2.0)

    on_idle.assert_awaited_once()
    on_session_end.assert_awaited_once_with("proactive_done")
    assert closed.is_set()
