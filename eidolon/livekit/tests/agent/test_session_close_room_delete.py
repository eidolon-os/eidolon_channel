"""Prompt room teardown on session close (plan §10 follow-up).

Real-device bug (confirmed via 3-way logs): JOIN → X (leave) → quick re-JOIN →
"in room + agent_speaking state but NO audio". Cause: on the device's
disconnect the old agent + its audio track lingered in the fixed-name room for
the whole STT/TTS shutdown drain; the auto_subscribe=false client re-joining the
still-alive room subscribed to the STALE track. Fix: delete the room PROMPTLY on
session close, before the drain — so a re-JOIN gets a fresh room with only its
new agent. These tests pin the prompt-delete hook.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.contracts import SESSION_END_ERROR, SESSION_END_USER_LEFT

from eidolon.livekit.agent.full_duplex import StreamingPipeline


@pytest.mark.asyncio
async def test_session_close_deletes_room_promptly():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._on_session_closed = AsyncMock()
    await p._delete_room_on_close()
    p._on_session_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_close_is_noop_without_callback():
    """Direct-construction / tests with no callback wired must not blow up."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    await p._delete_room_on_close()  # no _on_session_closed attr → no-op


@pytest.mark.asyncio
async def test_session_close_delete_swallows_errors():
    """A failing room-delete must not break the run()/shutdown path."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._on_session_closed = AsyncMock(side_effect=RuntimeError("boom"))
    await p._delete_room_on_close()  # must not raise
    p._on_session_closed.assert_awaited_once()


# B2 (plan §3.2): every room-deletion path carries a session_end reason.


@pytest.mark.asyncio
async def test_session_close_publishes_user_left_before_delete():
    """Clean close → session_end{user_left} published BEFORE the prompt delete."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("delete", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = None
    await p._delete_room_on_close()
    assert calls == [("end", SESSION_END_USER_LEFT), ("delete", None)]


@pytest.mark.asyncio
async def test_session_close_publishes_error_on_error_close():
    """An error-close maps to session_end{error}, not user_left."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("delete", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = RuntimeError("boom")
    await p._delete_room_on_close()
    assert calls[0] == ("end", SESSION_END_ERROR)
    assert ("delete", None) in calls
