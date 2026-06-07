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
) -> tuple[IdleWatchdog, asyncio.Event]:
    closed = asyncio.Event()
    watchdog = IdleWatchdog(
        timeout_sec=timeout_sec,
        get_session=lambda: session,
        get_room=lambda: room,
        get_timeline=lambda: timeline,
        session_closed_event=closed,
        on_idle_disconnect=on_idle_disconnect,
        disconnect_grace_sec=0.0,
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
    assert json.loads(payload) == {
        "type": "idle_timeout",
        "reason": "idle_timeout",
    }
    assert timeline.snapshot()["timestamps"]["idle_timeout_triggered_at"]
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
