"""Bailian TTS stream stall watchdog.

Root cause of the "long reply cut off mid-playback" bug: the stream had a fixed
~45s total-lifetime cap. A long reply whose text the LLM streams over many
seconds stays alive past 45s even though it is perfectly healthy (audio keeps
flowing), so the blanket cap fired mid-stream → APIError after partial audio →
framework can't retry (audio already played) → truncated reply.

The fix replaces the total cap with an INACTIVITY watchdog: the deadline resets
on every provider message and every text token, so only a genuine stall aborts.
These tests pin that behaviour on ``_await_stream_completion`` directly.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from livekit.agents import APIError

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianSynthesizeStream


def _stream(stall_timeout: float) -> BailianSynthesizeStream:
    # Build via __new__ to isolate _await_stream_completion — a real
    # ``tts.stream()`` would launch the framework's _main_task/_run in the
    # background, which resets _exit_event and would race this unit test.
    stream = BailianSynthesizeStream.__new__(BailianSynthesizeStream)
    stream._config = BailianTTSConfig(api_key="test")
    stream._config.stream_stall_timeout_sec = stall_timeout
    stream._exit_event = asyncio.Event()
    stream._last_stream_activity_at = time.monotonic()
    return stream


@pytest.mark.asyncio
async def test_aborts_after_inactivity() -> None:
    """No activity for the stall window → APIError(retryable)."""
    stream = _stream(0.3)
    with pytest.raises(APIError) as exc:
        await asyncio.wait_for(stream._await_stream_completion(), timeout=2.0)
    assert exc.value.retryable is True
    assert "stalled" in str(exc.value)


@pytest.mark.asyncio
async def test_continuous_activity_never_aborts() -> None:
    """A long stream that keeps making progress past the stall window must NOT
    be aborted — this is the long-reply case the old fixed cap broke."""
    stream = _stream(0.3)

    async def keep_alive() -> None:
        # ~5 stall windows of steady activity, then finish cleanly.
        for _ in range(15):
            await asyncio.sleep(0.1)
            stream._mark_stream_activity()
        stream._exit_event.set()

    ka = asyncio.create_task(keep_alive())
    # Must return (not raise) despite running well past stall_timeout.
    await asyncio.wait_for(stream._await_stream_completion(), timeout=3.0)
    await ka


@pytest.mark.asyncio
async def test_returns_on_finish() -> None:
    """exit_event set (task-finished) → returns promptly, no stall raise."""
    stream = _stream(5.0)
    stream._exit_event.set()
    await asyncio.wait_for(stream._await_stream_completion(), timeout=1.0)


@pytest.mark.asyncio
async def test_disabled_waits_for_finish_only() -> None:
    """stall_timeout<=0 disables the watchdog — plain wait for completion."""
    stream = _stream(0.0)
    stream._last_stream_activity_at = time.monotonic() - 100  # would stall if active

    async def finish_later() -> None:
        await asyncio.sleep(0.2)
        stream._exit_event.set()

    fl = asyncio.create_task(finish_later())
    await asyncio.wait_for(stream._await_stream_completion(), timeout=1.0)
    await fl
