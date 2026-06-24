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

from eidolon.livekit.agent.streaming import StreamingPipeline


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
