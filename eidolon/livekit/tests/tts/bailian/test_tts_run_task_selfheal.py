"""Bailian TTS run-task self-heal at acquire time.

Pre-warmed pool connections only do the WebSocket connect; run-task is sent at
acquire time. DashScope intermittently rejects run-task on a connection that has
sat idle ("Invalid action('run-task')! Please follow the protocol!"). Instead of
bubbling that to the framework, ``_acquire_started_conn`` discards the bad
connection and retries on a fresh one — transparently and bounded.

Two layers tested:
  - client: ``start_task`` fast-fails on a handshake task-failed (no 15s wait).
  - pool:   ``_acquire_started_conn`` self-heals past a rejecting connection.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.bailian.tts import BailianTTS
from eidolon.livekit.plugins.tts.bailian.tts_client import (
    BailianTTSClient,
    BailianTTSError,
)


def _tts() -> BailianTTS:
    return BailianTTS(
        BailianTTSConfig(api_key="test", pool_size=1, pool_size_bootstrap=1)
    )


class _FakeConn:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = 0

    async def start_task(self) -> None:
        self.started += 1
        if self.fail:
            raise BailianTTSError(
                "[InvalidParameter] Invalid action('run-task')! Please follow the protocol!",
                recoverable=True,
            )


# ── client-level: start_task fast-fail ───────────────────────────────────


@pytest.mark.asyncio
async def test_start_task_fast_fails_on_task_failed() -> None:
    client = BailianTTSClient(
        uri="ws://x", api_key="k", model="m", voice="v",
        audio_format="pcm", sample_rate=16000,
    )
    client._connected = True
    client._send_json = AsyncMock()  # don't touch a real socket

    async def reject_soon() -> None:
        await asyncio.sleep(0.05)
        # Mimic the receive loop seeing a task-failed during the handshake.
        client._task_failed_error = BailianTTSError(
            "[InvalidParameter] Invalid action('run-task')!", recoverable=True
        )
        client._task_started_event.set()

    asyncio.create_task(reject_soon())
    with pytest.raises(BailianTTSError) as exc:
        await asyncio.wait_for(client.start_task(), timeout=2.0)
    assert "Invalid action" in str(exc.value)
    assert exc.value.recoverable is True


# ── pool-level: _acquire_started_conn self-heal ──────────────────────────


@pytest.mark.asyncio
async def test_acquire_started_conn_healthy_first_try() -> None:
    tts = _tts()
    good = _FakeConn()
    tts._acquire_conn = AsyncMock(return_value=good)

    result = await tts._acquire_started_conn()
    assert result is good
    assert good.started == 1


@pytest.mark.asyncio
async def test_acquire_started_conn_self_heals_past_reject() -> None:
    tts = _tts()
    bad, good = _FakeConn(fail=True), _FakeConn()
    seq = [bad, good]
    tts._acquire_conn = AsyncMock(side_effect=lambda: seq.pop(0))
    disposed: list = []
    tts._pool.mark_dirty = AsyncMock(side_effect=lambda c: disposed.append(c))

    result = await tts._acquire_started_conn(attempts=3)
    assert result is good            # got the healthy connection
    assert disposed == [bad]         # the rejecting one was discarded
    assert good.started == 1


@pytest.mark.asyncio
async def test_acquire_started_conn_raises_recoverable_after_attempts() -> None:
    tts = _tts()
    conns = [_FakeConn(fail=True) for _ in range(5)]
    seq = list(conns)
    tts._acquire_conn = AsyncMock(side_effect=lambda: seq.pop(0))
    tts._pool.mark_dirty = AsyncMock()

    with pytest.raises(BailianTTSError) as exc:
        await tts._acquire_started_conn(attempts=3)
    assert exc.value.recoverable is True
    assert tts._acquire_conn.await_count == 3  # bounded to attempts


@pytest.mark.asyncio
async def test_acquire_started_conn_propagates_non_recoverable() -> None:
    """A non-recoverable error (e.g. shutting down) is not retried."""
    tts = _tts()

    async def boom() -> None:
        raise BailianTTSError("[BailianTTS] shutting down", recoverable=False)

    tts._acquire_conn = boom  # type: ignore[method-assign]
    with pytest.raises(BailianTTSError) as exc:
        await tts._acquire_started_conn()
    assert exc.value.recoverable is False
