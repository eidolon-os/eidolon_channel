"""Unit tests for the idle-disconnect watchdog.

A client that connects and is never closed keeps STT streaming (and billing)
for the whole connection even while silent. The watchdog closes the session
after ``turn_policy.idle.disconnect_after_idle_ms`` of no recognized speech
and no agent activity. Bare VAD/noise (empty ASR) must NOT keep it alive.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_pipeline(
    *, timeout_sec: float, on_idle_disconnect=None
) -> "StreamingPipeline":
    """Minimally-initialised pipeline exercising only the watchdog slice."""
    from eidolon.livekit.agent.streaming import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._idle_timeout_sec = timeout_sec
    pipeline._idle_watchdog_task = None
    pipeline._last_activity_monotonic = 0.0
    pipeline._timeline = None
    pipeline._session_closed_event = asyncio.Event()
    pipeline._on_idle_disconnect = on_idle_disconnect
    pipeline._idle_disconnect_grace_sec = 0.0  # no grace delay in tests

    # Room with a local participant that records published data packets.
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    pipeline._room = room

    session = MagicMock()
    session.aclose = AsyncMock()
    # Inactive states so the watchdog's "don't cut a live turn" guard lets it fire.
    session.agent_state = "idle"
    session.user_state = "listening"
    pipeline._session = session
    return pipeline


@pytest.mark.asyncio
async def test_idle_watchdog_deletes_room_and_notifies_client():
    on_idle = AsyncMock()
    pipeline = _make_pipeline(timeout_sec=0.05, on_idle_disconnect=on_idle)
    pipeline._start_idle_watchdog()

    await asyncio.wait_for(pipeline._idle_watchdog_task, timeout=2.0)

    # Room deleted via the wired callback (not a bare session.aclose).
    on_idle.assert_awaited_once()
    pipeline._session.aclose.assert_not_awaited()
    # Client notified on the eidolon.session_control topic before the room went away.
    pipeline._room.local_participant.publish_data.assert_awaited_once()
    kwargs = pipeline._room.local_participant.publish_data.await_args.kwargs
    assert kwargs["topic"] == "eidolon.session_control"
    import json
    assert json.loads(pipeline._room.local_participant.publish_data.await_args.args[0]) == {
        "type": "idle_timeout",
        "reason": "idle_timeout",
    }
    # run() is released to shut down.
    assert pipeline._session_closed_event.is_set()


@pytest.mark.asyncio
async def test_idle_watchdog_fallback_closes_session_without_callback():
    """Without a room-delete callback, the watchdog at least stops STT/TTS."""
    pipeline = _make_pipeline(timeout_sec=0.05, on_idle_disconnect=None)
    pipeline._start_idle_watchdog()

    await asyncio.wait_for(pipeline._idle_watchdog_task, timeout=2.0)
    pipeline._session.aclose.assert_awaited_once()
    assert pipeline._session_closed_event.is_set()


@pytest.mark.asyncio
async def test_activity_postpones_disconnect():
    pipeline = _make_pipeline(timeout_sec=0.2)
    pipeline._start_idle_watchdog()

    # Keep marking activity faster than the timeout — must stay connected.
    for _ in range(5):
        await asyncio.sleep(0.1)
        pipeline._mark_activity()
    assert not pipeline._idle_watchdog_task.done()
    pipeline._session.aclose.assert_not_awaited()

    # Stop refreshing — now it should disconnect.
    await asyncio.wait_for(pipeline._idle_watchdog_task, timeout=2.0)
    pipeline._session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_timeout_disables_watchdog():
    pipeline = _make_pipeline(timeout_sec=0.0)
    pipeline._start_idle_watchdog()

    assert pipeline._idle_watchdog_task is None
    await asyncio.sleep(0.1)
    pipeline._session.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_agent_turn_is_not_cut():
    """A continuously-speaking agent (no state transition) must not be cut."""
    pipeline = _make_pipeline(timeout_sec=0.05)
    pipeline._session.agent_state = "speaking"  # mid-reply, no fresh transition
    pipeline._start_idle_watchdog()

    # Watchdog wakes past the deadline but sees the agent still speaking → re-arms.
    await asyncio.sleep(0.3)
    assert not pipeline._idle_watchdog_task.done()
    pipeline._session.aclose.assert_not_awaited()

    # Reply finishes → next wake cuts it.
    pipeline._session.agent_state = "idle"
    await asyncio.wait_for(pipeline._idle_watchdog_task, timeout=2.0)
    pipeline._session.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_recognized_speech_marks_activity_empty_does_not():
    """A non-empty transcript refreshes the activity clock; empty does not."""
    pipeline = _make_pipeline(timeout_sec=10.0)
    pipeline._ensure_runtime_defaults()
    pipeline._callbacks = MagicMock()
    pipeline._allow_interruptions = False
    pipeline._state = None
    pipeline._get_eot_model = MagicMock(return_value=MagicMock())
    pipeline._last_activity_monotonic = 0.0

    # Empty transcript → no activity bump.
    pipeline._on_user_transcribed(SimpleNamespace(transcript="", is_final=False))
    assert pipeline._last_activity_monotonic == 0.0

    # Real recognized text → activity bumped.
    pipeline._on_user_transcribed(
        SimpleNamespace(transcript="你好", is_final=True)
    )
    assert pipeline._last_activity_monotonic > 0.0
