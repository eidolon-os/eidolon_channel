"""Prompt teardown on session close (plan §10 follow-up).

Real-device bug (confirmed via 3-way logs): JOIN → X (leave) → quick re-JOIN →
"in room + agent_speaking state but NO audio". Cause: on the device's
disconnect the old agent + its audio track lingered in the fixed-name room for
the whole STT/TTS shutdown drain; the auto_subscribe=false client re-entering
subscribed to the STALE track. Fix: give up serving PROMPTLY on session close,
before the drain — LiveKit then removes the agent from the room while the job is
still shutting down, so the next session finds only its own agent. These tests
pin the prompt-teardown hook; what the callback does with it (withdraw the
dispatch) belongs to server.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.contracts import SESSION_END_ERROR, SESSION_END_USER_LEFT

from eidolon.livekit.agent.full_duplex.lifecycle import FullDuplexSessionLifecycle
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from eidolon.livekit.agent.half_duplex import HalfDuplexPttPipeline


@pytest.mark.asyncio
async def test_session_close_gives_up_serving_promptly():
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._on_session_closed = AsyncMock()
    await FullDuplexSessionLifecycle(p)._end_serving_on_close()
    p._on_session_closed.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_close_is_noop_without_callback():
    """Direct-construction / tests with no callback wired must not blow up."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    await FullDuplexSessionLifecycle(p)._end_serving_on_close()


@pytest.mark.asyncio
async def test_session_close_teardown_swallows_errors():
    """A failing teardown must not break the run()/shutdown path."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._on_session_closed = AsyncMock(side_effect=RuntimeError("boom"))
    await FullDuplexSessionLifecycle(p)._end_serving_on_close()
    p._on_session_closed.assert_awaited_once()


# B2 (plan §3.2): every teardown path carries a session_end reason.


@pytest.mark.asyncio
async def test_session_close_publishes_user_left_before_teardown():
    """Clean close → session_end{user_left} published BEFORE the prompt teardown."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("teardown", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = None
    await FullDuplexSessionLifecycle(p)._end_serving_on_close()
    assert calls == [("end", SESSION_END_USER_LEFT), ("teardown", None)]


@pytest.mark.asyncio
async def test_session_close_publishes_error_on_error_close():
    """An error-close maps to session_end{error}, not user_left."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("teardown", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = RuntimeError("boom")
    await FullDuplexSessionLifecycle(p)._end_serving_on_close()
    assert calls[0] == ("end", SESSION_END_ERROR)
    assert ("teardown", None) in calls


@pytest.mark.asyncio
async def test_half_duplex_session_close_publishes_user_left_before_teardown():
    """Half-duplex close follows the same session_end-before-teardown contract."""
    p = HalfDuplexPttPipeline.__new__(HalfDuplexPttPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("teardown", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = None

    await p._end_serving_on_close()

    assert calls == [("end", SESSION_END_USER_LEFT), ("teardown", None)]


@pytest.mark.asyncio
async def test_half_duplex_session_close_publishes_error_on_error_close():
    """Half-duplex error close maps to session_end{error}, then still tears down."""
    p = HalfDuplexPttPipeline.__new__(HalfDuplexPttPipeline)
    calls: list[tuple[str, str | None]] = []

    async def _end(reason: str) -> None:
        calls.append(("end", reason))

    async def _closed() -> None:
        calls.append(("teardown", None))

    p._on_session_end = _end
    p._on_session_closed = _closed
    p._close_error = RuntimeError("boom")

    await p._end_serving_on_close()

    assert calls[0] == ("end", SESSION_END_ERROR)
    assert ("teardown", None) in calls
