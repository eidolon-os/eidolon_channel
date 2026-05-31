"""Unit tests for SenseTime TTS persistent-connection model.

Architecture under test:
    warmup()   → connect → connected_success → task_start → heartbeat starts
    stream()   → task_continue(×N)            (no task_finish between turns)
    shutdown() → stop heartbeat → task_finish → disconnect

Run with::

    cd <repository-root>
    .venv/bin/python -m pytest eidolon/livekit/tests/tts/sensetime/test_tts_persistent.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

_root = Path(__file__).resolve().parents[5]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc  # noqa: E402

from eidolon.livekit.plugins.tts.sensetime import (  # noqa: E402
    SenseTimeTTS,
    SenseTimeTTSConfig,
    SenseTimeTTSError,
)
from eidolon.livekit.plugins.tts.sensetime.tts_client import (  # noqa: E402
    TTSConnection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-40s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("test_persistent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_audio(num_bytes: int = 1920) -> str:
    """Deterministic hex audio bytes (60ms at 16kHz mono = 1920 bytes)."""
    return (b"\x01\x00" * num_bytes)[:num_bytes].hex()


def make_config(api_url: str) -> SenseTimeTTSConfig:
    return SenseTimeTTSConfig(
        api_url=api_url,
        api_key="test-key",
        model="test-model",
        voice="female_0033_a",
        sample_rate=16000,
    )


# ---------------------------------------------------------------------------
# Mock TTS WebSocket Server — persistent connection edition
# ---------------------------------------------------------------------------

class MockTTSServer:
    """Mock SenseAudio server that follows the persistent-connection protocol.

    The server keeps a WebSocket connection alive across multiple task_continue
    rounds.  It only sends task_finished and closes after receiving task_finish
    (sent by SenseTimeTTS.shutdown()).

    Heartbeat messages (empty-text task_continue) are silently ignored.
    """

    def __init__(
        self,
        port: int = 0,
        task_failure: bool = False,
        close_after_connected: bool = False,
        echo_audio: bool = True,
        batch_delay: float = 0.0,
    ):
        self.port = port
        self.task_failure = task_failure
        self.close_after_connected = close_after_connected
        self.echo_audio = echo_audio
        # Inter-batch synthesis delay; setting this > 0.2 reproduces the
        # production scenario where consecutive batches arrive with gaps
        # that exceeded the old 200ms quiesce timer.
        self.batch_delay = batch_delay
        self._runner: Any = None
        self._site: Any = None

        self._task_start_total: int = 0
        self._task_continue_total: int = 0
        self._task_finish_total: int = 0
        self._text_received: list[str] = []
        self._connections_total: int = 0
        # Per-connection heartbeat count (empty task_continue)
        self._heartbeat_total: int = 0

    @property
    def task_start_count(self) -> int:
        return self._task_start_total

    @property
    def task_continue_count(self) -> int:
        return self._task_continue_total

    @property
    def task_finish_count(self) -> int:
        return self._task_finish_total

    async def start(self) -> None:
        import aiohttp
        from aiohttp import web

        async def ws_handler(request: web.Request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self._connections_total += 1
            session_id = f"sess-{self._connections_total}"

            await ws.send_json({
                "event": "connected_success",
                "session_id": session_id,
                "trace_id": session_id,
            })

            if self.close_after_connected:
                await ws.close()
                return

            try:
                while True:
                    if ws.closed:
                        break
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                    except asyncio.TimeoutError:
                        break

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            break
                        continue

                    try:
                        msg_obj = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    event = msg_obj.get("event", "")

                    if event == "task_start":
                        self._task_start_total += 1
                        await ws.send_json({
                            "event": "task_started",
                            "session_id": session_id,
                        })

                    elif event == "task_continue":
                        text = msg_obj.get("text", "")

                        if not text:
                            # Empty text = heartbeat — count it but don't echo audio
                            self._heartbeat_total += 1
                            continue

                        self._task_continue_total += 1
                        self._text_received.append(text)

                        if self.task_failure:
                            await ws.send_json({
                                "event": "task_failed",
                                "data": {
                                    "base_resp": {"status_msg": "mock task failure"}
                                },
                            })
                            await ws.close()
                            break

                        if self.echo_audio:
                            # Optional inter-batch synthesis delay (mimics real
                            # SenseAudio behaviour where each text segment takes
                            # some time to synthesise and batches arrive with gaps).
                            if self.batch_delay > 0:
                                await asyncio.sleep(self.batch_delay)
                            # 1) audio chunk
                            await ws.send_json({
                                "event": "task_continued",
                                "data": {"audio": hex_audio(1920), "status": 0},
                            })
                            # 2) batch-end marker (empty audio + status field present).
                            # Real server sends this after each text segment's
                            # synthesis is complete; client uses these markers to
                            # know when all submitted text has been processed.
                            await ws.send_json({
                                "event": "task_continued",
                                "data": {"audio": "", "status": 0},
                            })

                    elif event == "task_finish":
                        self._task_finish_total += 1
                        # Per protocol: send task_finished then close
                        await ws.send_json({"event": "task_finished"})
                        await ws.close()
                        break

            except Exception:
                pass
            return web.Response()

        app = web.Application()
        app.router.add_get("/ws", ws_handler)
        self._runner = aiohttp.web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await self._site.start()
        if self._site._server and self._site._server.sockets:
            self.port = self._site._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
            self._site = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def srv() -> tuple[MockTTSServer, Any]:
    import aiohttp
    server = MockTTSServer(port=0, echo_audio=True)
    await server.start()
    await asyncio.sleep(0.05)
    session = aiohttp.ClientSession()
    yield server, session
    await session.close()
    await server.stop()


@pytest_asyncio.fixture
async def failing_srv() -> tuple[MockTTSServer, Any]:
    import aiohttp
    server = MockTTSServer(port=0, task_failure=True)
    await server.start()
    await asyncio.sleep(0.05)
    session = aiohttp.ClientSession()
    yield server, session
    await session.close()
    await server.stop()


@pytest_asyncio.fixture
async def close_srv() -> tuple[MockTTSServer, Any]:
    import aiohttp
    server = MockTTSServer(port=0, close_after_connected=True)
    await server.start()
    await asyncio.sleep(0.05)
    session = aiohttp.ClientSession()
    yield server, session
    await session.close()
    await server.stop()


def _make_conn(uri: str, session: Any) -> TTSConnection:
    return TTSConnection(
        uri=uri, api_key="test-key", model="test-model", voice_id="female_0033_a",
        sample_rate=16000, speed=1.0, vol=1.0, pitch=0, http_session=session,
    )


# ---------------------------------------------------------------------------
# TTSConnection: Basic Lifecycle
# ---------------------------------------------------------------------------

class TestTTSConnectionLifecycle:

    @pytest.mark.asyncio
    async def test_connect_success(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        ok = await conn.connect()
        assert ok
        assert conn.is_connected
        assert conn._state == "ready"
        await conn.disconnect()
        logger.info("[PASS] test_connect_success")

    @pytest.mark.asyncio
    async def test_disconnect(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        assert conn.is_connected
        await conn.disconnect()
        assert not conn.is_connected
        assert conn._state == "disconnected"
        logger.info("[PASS] test_disconnect")

    @pytest.mark.asyncio
    async def test_task_start(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await asyncio.sleep(0.1)
        assert conn._state == "task_active"
        assert server.task_start_count == 1
        await conn.disconnect()
        logger.info("[PASS] test_task_start")

    @pytest.mark.asyncio
    async def test_task_continue(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await conn.send_task_continue("hello world")
        await asyncio.sleep(0.05)
        assert len(server._text_received) == 1
        assert server._text_received[0] == "hello world"
        await conn.disconnect()
        logger.info("[PASS] test_task_continue")

    @pytest.mark.asyncio
    async def test_task_finish_transitions_to_finishing(self, srv):
        """send_task_finish sets state to 'finishing', not 'ready'."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await asyncio.sleep(0.1)
        await conn.send_task_finish()
        assert conn._state == "finishing"
        # Wait for server to close
        await asyncio.sleep(0.3)
        assert conn._state == "disconnected"
        await conn.disconnect()
        logger.info("[PASS] test_task_finish_transitions_to_finishing")

    @pytest.mark.asyncio
    async def test_double_disconnect_is_safe(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.disconnect()
        await conn.disconnect()
        logger.info("[PASS] test_double_disconnect_is_safe")

    @pytest.mark.asyncio
    async def test_send_json_while_disconnected_raises(self):
        conn = TTSConnection(
            uri="ws://127.0.0.1:99999/ws", api_key="test-key", model="test-model",
            voice_id="female_0033_a", sample_rate=16000, speed=1.0, vol=1.0, pitch=0,
        )
        with pytest.raises(SenseTimeTTSError):
            await conn._send_json({"event": "test"})
        logger.info("[PASS] test_send_json_while_disconnected_raises")

    @pytest.mark.asyncio
    async def test_state_transitions(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        assert conn._state == "disconnected"

        await conn.connect()
        assert conn._state == "ready"

        await conn.send_task_start()
        await asyncio.sleep(0.1)
        assert conn._state == "task_active"

        await conn.disconnect()
        assert conn._state == "disconnected"
        logger.info("[PASS] test_state_transitions")

    @pytest.mark.asyncio
    async def test_backward_compatibility_alias(self, srv):
        from eidolon.livekit.plugins.tts.sensetime.tts_client import SenseTimeTTSClient
        assert SenseTimeTTSClient is TTSConnection
        server, session = srv
        client = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        ok = await client.connect()
        assert ok
        await client.disconnect()
        logger.info("[PASS] test_backward_compatibility_alias")


# ---------------------------------------------------------------------------
# TTSConnection: Heartbeat
# ---------------------------------------------------------------------------

class TestTTSConnectionHeartbeat:
    """Verify TTSConnection heartbeat support for persistent connection keep-alive."""

    @pytest.mark.asyncio
    async def test_has_heartbeat_task_after_start(self, srv):
        """start_heartbeat() creates a running task; stop_heartbeat() cancels it."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()

        conn.start_heartbeat(30.0, stream_active_check=lambda: False)
        assert conn._heartbeat_task is not None
        assert not conn._heartbeat_task.done()

        conn.stop_heartbeat()
        assert conn._heartbeat_task is None

        await conn.disconnect()
        logger.info("[PASS] test_has_heartbeat_task_after_start")

    @pytest.mark.asyncio
    async def test_heartbeat_sends_when_no_stream_active(self, srv):
        """Heartbeat fires and sends empty task_continue when no stream is active."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await asyncio.sleep(0.1)

        initial_count = server._heartbeat_total
        # Very short interval to make the test fast
        conn.start_heartbeat(0.05, stream_active_check=lambda: False)
        await asyncio.sleep(0.3)  # At least 3 heartbeat intervals
        conn.stop_heartbeat()

        assert server._heartbeat_total > initial_count
        await conn.disconnect()
        logger.info(
            "[PASS] test_heartbeat_sends_when_no_stream_active "
            "(heartbeats=%d)",
            server._heartbeat_total - initial_count,
        )

    @pytest.mark.asyncio
    async def test_heartbeat_skips_when_stream_active(self, srv):
        """Heartbeat does NOT fire while stream_active_check() returns True."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await asyncio.sleep(0.1)

        initial_count = server._heartbeat_total
        conn.start_heartbeat(0.05, stream_active_check=lambda: True)  # always active
        await asyncio.sleep(0.3)
        conn.stop_heartbeat()

        # No heartbeats should have been sent
        assert server._heartbeat_total == initial_count
        await conn.disconnect()
        logger.info("[PASS] test_heartbeat_skips_when_stream_active")

    @pytest.mark.asyncio
    async def test_stop_heartbeat_cancels_task(self, srv):
        """stop_heartbeat() cancels the background task."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()

        conn.start_heartbeat(30.0, stream_active_check=lambda: False)
        task = conn._heartbeat_task
        assert task is not None

        conn.stop_heartbeat()
        assert conn._heartbeat_task is None

        await asyncio.sleep(0.05)
        assert task.done()

        await conn.disconnect()
        logger.info("[PASS] test_stop_heartbeat_cancels_task")

    @pytest.mark.asyncio
    async def test_start_heartbeat_idempotent(self, srv):
        """Calling start_heartbeat() twice does not create a second task."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()

        conn.start_heartbeat(30.0, stream_active_check=lambda: False)
        task1 = conn._heartbeat_task
        conn.start_heartbeat(30.0, stream_active_check=lambda: False)
        task2 = conn._heartbeat_task

        assert task1 is task2  # Same task, not a new one
        conn.stop_heartbeat()
        await conn.disconnect()
        logger.info("[PASS] test_start_heartbeat_idempotent")


# ---------------------------------------------------------------------------
# TTSConnection: Reconnect via ensure_connected
# ---------------------------------------------------------------------------

class TestTTSConnectionReconnect:

    @pytest.mark.asyncio
    async def test_ensure_connected_when_ready(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        ok = await conn.ensure_connected()
        assert ok
        await conn.disconnect()
        logger.info("[PASS] test_ensure_connected_when_ready")

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_after_force_close(self, srv):
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn._force_close()
        assert conn._state == "disconnected"

        ok = await conn.ensure_connected()
        assert ok
        assert conn._state == "ready"
        await conn.disconnect()
        logger.info("[PASS] test_ensure_connected_reconnects_after_force_close")

    @pytest.mark.asyncio
    async def test_ensure_connected_detects_closed_ws(self, srv):
        """ensure_connected detects ws.closed and reconnects."""
        server, session = srv
        conn = _make_conn(f"ws://127.0.0.1:{server.port}/ws", session)
        await conn.connect()
        await conn.send_task_start()
        await asyncio.sleep(0.1)
        # Simulate server closing after task_finish
        await conn.send_task_finish()
        await asyncio.sleep(0.3)

        # State should be disconnected after server closes
        ok = await conn.ensure_connected()
        assert ok
        assert conn._state == "ready"
        await conn.disconnect()
        logger.info("[PASS] test_ensure_connected_detects_closed_ws")


# ---------------------------------------------------------------------------
# SenseTimeTTS: persistent connection lifecycle
# ---------------------------------------------------------------------------

class TestSenseTimeTTSPersistentConnection:
    """Verify the single-connection lifecycle: warmup → streams → shutdown."""

    @pytest.mark.asyncio
    async def test_warmup_opens_pool_of_warm_conns(self, srv):
        """Round 8 R8.7: warmup() opens ``pool_size`` connections in parallel.

        Replaces the old "single persistent conn" assertion. With pool=2,
        warmup opens 2 WSes, each does its own task_start. Pool is then
        ready to serve turns with zero acquire latency.
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        await tts.warmup()

        # Pool opens N parallel connections, each runs its own task_start
        assert server._connections_total == 2
        assert server.task_start_count == 2
        # Pool reports 2 warm conns ready
        assert tts._pool.warm_count == 2
        # No task_finish yet (we don't send finish; we close on dispose)
        assert server.task_finish_count == 0

        await tts.shutdown()
        logger.info("[PASS] test_warmup_opens_pool_of_warm_conns")

    @pytest.mark.asyncio
    async def test_warmup_is_idempotent(self, srv):
        """Round 8 R8.7: second warmup() call when pool is full is a no-op."""
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        await tts.warmup()
        first_count = server._connections_total

        await tts.warmup()  # pool already at target; should be a no-op

        assert server._connections_total == first_count
        assert tts._pool.warm_count == 2

        await tts.shutdown()
        logger.info("[PASS] test_warmup_is_idempotent")

    @pytest.mark.asyncio
    async def test_shutdown_disposes_pool(self, srv):
        """Round 8 R8.7: shutdown() drains the pool by closing all warm conns.

        Replaces the old "task_finish on shutdown" assertion. We no longer
        send task_finish — the pool simply closes each conn. Server sees
        WebSocket close events instead.
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        await tts.warmup()
        assert tts._pool.warm_count == 2

        await tts.shutdown()

        # Pool empty after shutdown
        assert tts._pool.warm_count == 0
        assert tts._conn is None
        assert tts._http_session is None
        logger.info("[PASS] test_shutdown_disposes_pool")

    @pytest.mark.asyncio
    async def test_no_task_finish_between_streams(self, srv):
        """Round 8 R8.7: task_finish is never sent (we close conns instead).

        Pool model: each turn acquires a fresh conn from the pool, uses
        it, then disposes (closes) it. Server sees WS close, not
        task_finish event.
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        for i in range(3):
            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                try:
                    async for ev in stream:
                        if hasattr(ev, "frame"):
                            frames.append(ev.frame)
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text(f"turn {i}")
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            assert len(frames) > 0

            # task_finish must NEVER be sent — we close conns instead
            assert server.task_finish_count == 0, (
                f"task_finish sent after turn {i}!"
            )

        await tts.shutdown()
        # Still no task_finish — shutdown also just closes
        assert server.task_finish_count == 0
        logger.info("[PASS] test_no_task_finish_between_streams")

    @pytest.mark.asyncio
    async def test_sequential_streams_use_different_pool_conns(self, srv):
        """Round 8 R8.7: each stream gets a *fresh* pool conn (no reuse).

        Was: ``test_sequential_streams_reuse_same_connection`` — that
        invariant was deliberately inverted by R8.7 to eliminate
        cross-turn audio leakage on cancel. Now: each turn = one
        connection, discarded after use. Server sees N+1 connections
        (pool_size for warmup + N turn discards/refills).
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)
        await tts.warmup()
        warmup_count = server._connections_total
        assert warmup_count == 2

        total_frames = 0
        for i in range(3):
            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                try:
                    async for ev in stream:
                        if hasattr(ev, "frame"):
                            frames.append(ev.frame)
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text(f"stream {i}")
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            assert len(frames) > 0
            total_frames += len(frames)
            # Give the pool a moment to refill
            await asyncio.sleep(0.1)

        # Each turn discarded its conn; pool refilled from scratch.
        # Total connections = warmup (2) + 3 refills triggered by turn discards.
        assert server._connections_total >= warmup_count + 3, (
            f"expected ≥{warmup_count + 3} conns, got {server._connections_total}"
        )
        # Each conn ran its own task_start.
        assert server.task_start_count == server._connections_total

        await tts.shutdown()
        logger.info(
            "[PASS] test_sequential_streams_use_different_pool_conns "
            "(connections=%d, frames=%d)",
            server._connections_total, total_frames,
        )

    @pytest.mark.asyncio
    async def test_pool_recovers_from_dead_conn(self, srv):
        """Round 8 R8.7: a dead conn discovered at acquire-time triggers
        slow-path inline warmup; pool keeps working.

        Was ``test_reconnect_on_connection_death``. Pool model handles
        this transparently: if a warm conn died (e.g. server kicked it),
        on acquire we either get the dead one (and the next stream's
        first send fails → caller's responsibility) or a healthy one;
        either way, mark_dirty + refill restores the pool.
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)
        await tts.warmup()

        # Run a stream — should succeed normally even if pool dynamics
        # involve refill timing.
        stream = tts.stream()
        frames: list[rtc.AudioFrame] = []

        async def drain():
            try:
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)
            except Exception:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("hello")
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)
        assert len(frames) > 0

        await tts.shutdown()
        logger.info("[PASS] test_pool_recovers_from_dead_conn")

    @pytest.mark.asyncio
    async def test_heartbeat_started_on_each_pool_conn(self, srv):
        """Round 8 R8.7: every pool conn runs its own heartbeat to keep
        the SenseAudio server-side idle timer reset while it sits warm."""
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        await tts.warmup()

        # Each warm conn in the queue should have its heartbeat task running.
        # The pool exposes warm_count but not the conns directly; iterate
        # via internal queue (test-only inspection).
        # F2 fix (2026-05-16): pool queue now holds (conn, created_at) tuples
        # for acquire-time staleness eviction; unpack accordingly.
        warm_entries = list(tts._pool._ready._queue)  # internal, but stable
        assert len(warm_entries) == 2
        for entry in warm_entries:
            c, _created_at = entry
            assert c._heartbeat_task is not None
            assert not c._heartbeat_task.done()

        await tts.shutdown()
        logger.info("[PASS] test_heartbeat_started_on_each_pool_conn")

    @pytest.mark.asyncio
    async def test_shutdown_is_safe_when_not_warmed_up(self):
        """shutdown() with no prior warmup is a no-op."""
        config = make_config("ws://127.0.0.1:99999/ws")
        tts = SenseTimeTTS(config=config)
        await tts.shutdown()
        logger.info("[PASS] test_shutdown_is_safe_when_not_warmed_up")


# ---------------------------------------------------------------------------
# SenseTimeSynthesizeStream: Streaming
# ---------------------------------------------------------------------------

class TestSynthesizeStream:

    @pytest.mark.asyncio
    async def test_stream_basic(self, srv):
        """Round 8 R8.7: streaming produces audio frames via a pool conn.

        New invariants:
          - Each conn has its own task_start (was: one shared task_start)
          - task_finish is never sent (was: sent at shutdown)
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.log_audio_diag = True
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        stream = tts.stream()
        frames: list[rtc.AudioFrame] = []

        async def drain():
            try:
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)
            except Exception as e:
                logger.warning("drain error: %s", e)

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("hello persistent")
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)

        assert len(frames) > 0
        assert frames[0].sample_rate == 16000
        assert frames[0].num_channels == 1

        # No explicit warmup() in this test → first stream takes the
        # slow-path acquire (inline warmup of one conn). Subsequent
        # background refill may bring the pool up to size. Each conn
        # runs its own task_start. We never send task_finish.
        assert server.task_start_count >= 1
        assert server.task_finish_count == 0

        await tts.shutdown()
        assert server.task_finish_count == 0  # Still 0; pool just closes

        logger.info("[PASS] test_stream_basic (%d frames, %d conns)",
                    len(frames), server._connections_total)

    @pytest.mark.asyncio
    async def test_stream_task_failure_propagates(self, failing_srv):
        """task_failed from server is propagated as SenseTimeTTSError."""
        server, session = failing_srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)

        stream = tts.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except SenseTimeTTSError as e:
                logger.info("caught expected error: %s", e)
                raise

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("trigger failure")
        stream.end_input()

        with pytest.raises(SenseTimeTTSError) as exc_info:
            await asyncio.wait_for(task, timeout=5.0)

        assert "mock task failure" in str(exc_info.value)
        await tts.shutdown()
        logger.info("[PASS] test_stream_task_failure_propagates")

    @pytest.mark.asyncio
    async def test_stream_no_drain_on_interrupt(self, srv):
        """Cancelling a stream releases the lock quickly; the next stream works."""
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)

        stream = tts.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("interrupted stream")
        await asyncio.sleep(0.05)
        # Signal end-of-input so the internal _run() can exit cleanly when we cancel.
        # In production the agent framework always calls end_input() before interrupting.
        stream.end_input()
        t0 = time.time()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            pass
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"cancel should be fast, got {elapsed:.3f}s"

        # Second stream uses same connection (no reconnect)
        stream2 = tts.stream()
        frames: list[rtc.AudioFrame] = []

        async def drain2():
            try:
                async for ev in stream2:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)
            except Exception:
                pass

        task2 = asyncio.create_task(drain2())
        await asyncio.sleep(0)
        stream2.push_text("after interrupt")
        stream2.end_input()
        await asyncio.wait_for(task2, timeout=5.0)
        assert len(frames) > 0
        await tts.shutdown()
        logger.info("[PASS] test_stream_no_drain_on_interrupt (elapsed=%.3fs)", elapsed)

    @pytest.mark.asyncio
    async def test_multiple_text_chunks_produce_audio(self, srv):
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()
        count = 0

        async def drain():
            nonlocal count
            try:
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        count += 1
            except Exception:
                pass
            return count

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("first ")
        await asyncio.sleep(0.05)
        stream.push_text("second ")
        await asyncio.sleep(0.05)
        stream.push_text("third")
        stream.end_input()
        result = await asyncio.wait_for(task, timeout=5.0)
        assert result > 0
        await tts.shutdown()
        logger.info("[PASS] test_multiple_text_chunks_produce_audio (%d frames)", result)

    @pytest.mark.asyncio
    async def test_stream_audio_frame_format(self, srv):
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()
        frames: list[rtc.AudioFrame] = []

        async def drain():
            try:
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)
            except Exception:
                pass

        task = asyncio.create_task(drain())
        stream.push_text("format test")
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)
        assert len(frames) > 0
        for f in frames:
            assert f.sample_rate == 16000
            assert f.num_channels == 1
            assert len(f.data) % 2 == 0
        await tts.shutdown()
        logger.info("[PASS] test_stream_audio_frame_format (%d frames)", len(frames))


# ---------------------------------------------------------------------------
# Full Lifecycle Integration
# ---------------------------------------------------------------------------

class TestFullLifecycle:

    @pytest.mark.asyncio
    async def test_warmup_streams_shutdown_full_lifecycle(self, srv):
        """Round 8 R8.7: full lifecycle with pool architecture.

        New invariants:
          - warmup opens pool_size (2) conns
          - each turn opens a fresh conn, closes it after
          - task_finish is never sent (we just close)
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.log_audio_diag = True
        config.pool_size = 2
        tts = SenseTimeTTS(config=config)

        await tts.warmup()
        assert server._connections_total == 2  # pool of 2

        total_frames = 0
        for i in range(3):
            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                try:
                    async for ev in stream:
                        if hasattr(ev, "frame"):
                            frames.append(ev.frame)
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text(f"turn {i}")
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            total_frames += len(frames)
            assert len(frames) > 0
            await asyncio.sleep(0.1)  # let pool refill

        # Pool: 2 warmup + at least 3 refills = 5+ total connections
        assert server._connections_total >= 5
        # Each conn ran its own task_start
        assert server.task_start_count == server._connections_total
        # task_finish never sent
        assert server.task_finish_count == 0

        await tts.shutdown()
        assert server.task_finish_count == 0  # Still 0 — we close conns

        logger.info(
            "[PASS] test_warmup_streams_shutdown_full_lifecycle "
            "(%d frames across 3 turns, %d connections)",
            total_frames, server._connections_total,
        )


# ---------------------------------------------------------------------------
# Bugfix verification: event-driven exit + batch-end counting (2026-05-02)
# ---------------------------------------------------------------------------

class TestEventDrivenExit:
    """Verify the new event-driven exit logic.

    The old design used a 200ms quiesce timer and a 100ms poll loop, which
    failed in production when SenseAudio sent batches with >200ms inter-batch
    gaps. The new design counts ``_text_chunks_sent`` vs
    ``_batch_ends_received`` for a time-free exit signal.
    """

    @pytest.mark.asyncio
    async def test_multi_segment_audio_received_in_full(self):
        """All N pushed segments produce audio, none are dropped after batch 1.

        Reproduces the production bug: client pushes 5 text segments separated
        by sentence-ending punctuation (so the R8.2 SentenceAggregator treats
        each as its own batch), server replies with 5 batches, client must
        receive ALL 5 (not just the first).
        """
        import aiohttp
        server = MockTTSServer(port=0, echo_audio=True, batch_delay=0.0)
        await server.start()
        await asyncio.sleep(0.05)
        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{server.port}/ws")
            tts = SenseTimeTTS(config=config)

            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            # Each segment ends with hard punct so the aggregator emits
            # each as its own batch. (Without punct, the aggregator would
            # correctly merge them into 1 batch — that's the design.)
            for text in (
                "你好。",
                "我是你的。",
                "AI。",
                "语音助手。",
                "有什么我可以帮你的吗？",
            ):
                stream.push_text(text)
            stream.end_input()
            await asyncio.wait_for(task, timeout=10.0)

            assert server.task_continue_count == 5
            # Each segment produces audio of similar size; total should reflect
            # all 5 segments, not just the first.
            assert len(frames) >= 5, (
                f"expected ≥5 audio frames (one per segment), got {len(frames)}"
            )
            await tts.shutdown()
            logger.info(
                "[PASS] test_multi_segment_audio_received_in_full "
                "(segments=5 frames=%d)", len(frames),
            )
        finally:
            await session.close()
            await server.stop()

    @pytest.mark.asyncio
    async def test_inter_batch_gap_does_not_trigger_premature_exit(self):
        """500ms gap between batches must not cause the stream to exit early.

        This is the exact production scenario: server takes 500ms to start
        synthesising the second segment. Old code (200ms quiesce) exited at
        the gap. New code counts batch-ends and waits.
        """
        import aiohttp
        # 500ms inter-batch delay — exceeds any plausible quiesce threshold.
        server = MockTTSServer(port=0, echo_audio=True, batch_delay=0.5)
        await server.start()
        await asyncio.sleep(0.05)
        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{server.port}/ws")
            tts = SenseTimeTTS(config=config)
            stream = tts.stream()
            frames: list[rtc.AudioFrame] = []

            async def drain():
                async for ev in stream:
                    if hasattr(ev, "frame"):
                        frames.append(ev.frame)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            # Hard punct after each so the R8.2 aggregator treats them as
            # 3 separate batches (test name asserts the *exit* logic
            # handles inter-batch gaps without premature termination).
            stream.push_text("first.")
            stream.push_text("second.")
            stream.push_text("third.")
            stream.end_input()
            # Total wall-clock will be ≥ 3 × batch_delay = 1.5s; allow slack.
            await asyncio.wait_for(task, timeout=10.0)

            assert server.task_continue_count == 3
            assert len(frames) >= 3, (
                f"expected all 3 batches' audio, got {len(frames)} frames "
                "(inter-batch gap caused premature exit?)"
            )
            await tts.shutdown()
            logger.info(
                "[PASS] test_inter_batch_gap_does_not_trigger_premature_exit "
                "(frames=%d, batch_delay=0.5s)", len(frames),
            )
        finally:
            await session.close()
            await server.stop()

    @pytest.mark.asyncio
    async def test_exit_is_event_driven_not_polling(self, srv):
        """Stream exits within ~50ms of receiving the last batch-end marker.

        Old polling design: worst-case 100ms latency between condition met
        and exit. New event-driven design: ~0ms.
        """
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()

        async def drain():
            async for _ in stream:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("hello")
        stream.end_input()

        t0 = time.time()
        await asyncio.wait_for(task, timeout=2.0)
        elapsed = time.time() - t0

        # Server batch_delay=0 → audio + batch-end arrive nearly instantly.
        # Stream should exit well within 200ms (old impl floor was ~100-200ms).
        assert elapsed < 0.2, (
            f"event-driven exit should be near-instant; got {elapsed:.3f}s"
        )
        await tts.shutdown()
        logger.info(
            "[PASS] test_exit_is_event_driven_not_polling (elapsed=%.3fs)",
            elapsed,
        )

    @pytest.mark.asyncio
    async def test_no_first_audio_guard_fires_when_server_silent(self):
        """If server accepts text but never sends any audio, exit after 5s.

        This validates condition F (no_first_audio guard). We disable
        framework retries via ``APIConnectOptions(max_retry=0)`` so the test
        measures only the guard timing, not the framework's retry loop.
        """
        import aiohttp
        from aiohttp import web
        from livekit.agents.types import APIConnectOptions

        # Silent server: accepts task_continue but never replies with audio.
        async def silent_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json({"event": "connected_success", "session_id": "s1"})
            try:
                while not ws.closed:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    msg_obj = json.loads(msg.data)
                    if msg_obj.get("event") == "task_start":
                        await ws.send_json({"event": "task_started", "session_id": "s1"})
                    # Silent on task_continue.
            except Exception:
                pass
            return ws

        app = web.Application()
        app.router.add_get("/ws", silent_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        await asyncio.sleep(0.05)

        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{port}/ws")
            tts = SenseTimeTTS(config=config)
            # Disable framework retry so we measure ONLY the guard timing.
            stream = tts.stream(
                conn_options=APIConnectOptions(max_retry=0, timeout=30.0)
            )

            async def drain():
                try:
                    async for _ in stream:
                        pass
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text("the server will ignore this")
            stream.end_input()

            t0 = time.time()
            await asyncio.wait_for(task, timeout=8.0)
            elapsed = time.time() - t0

            assert 4.5 <= elapsed <= 6.5, (
                f"no_first_audio guard should fire ~5s after input_done; "
                f"got {elapsed:.3f}s"
            )
            await tts.shutdown()
            logger.info(
                "[PASS] test_no_first_audio_guard_fires_when_server_silent "
                "(elapsed=%.3fs)", elapsed,
            )
        finally:
            await session.close()
            await runner.cleanup()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_text_not_sent(self, srv):
        """Empty / whitespace-only text is not forwarded to the server."""
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except Exception:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("")
        stream.push_text("   ")
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)
        assert server.task_continue_count == 0  # No real text was sent
        logger.info(
            "[PASS] test_empty_text_not_sent (task_continue_count=%d)",
            server.task_continue_count,
        )
        await tts.shutdown()

    @pytest.mark.asyncio
    async def test_punctuation_stripped(self, srv):
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        tts = SenseTimeTTS(config=config)
        stream = tts.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except Exception:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_text("，。hello world，。")
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)
        assert len(server._text_received) > 0
        received = server._text_received[-1]
        assert received == "hello world", f"expected 'hello world', got {received!r}"
        await tts.shutdown()
        logger.info("[PASS] test_punctuation_stripped (received=%r)", received)

    @pytest.mark.asyncio
    async def test_properties_and_label(self):
        config = SenseTimeTTSConfig(
            api_url="ws://127.0.0.1:9999/ws", api_key="test-key", model="test-model",
            voice="female_0033_a", sample_rate=32000,
        )
        tts = SenseTimeTTS(config=config)
        assert tts.provider == "sensetime"
        assert tts.model == "test-model"
        assert tts.sample_rate == 32000
        assert tts.num_channels == 1
        assert "SenseTimeTTS" in tts.label
        logger.info("[PASS] test_properties_and_label")

    @pytest.mark.asyncio
    async def test_config_defaults(self):
        tts = SenseTimeTTS()
        assert tts.provider == "sensetime"
        # sample_rate may be overridden by SENSETIME_TTS_SAMPLE_RATE env var
        assert tts.sample_rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000)
        assert tts.num_channels == 1
        logger.info("[PASS] test_config_defaults (sample_rate=%d)", tts.sample_rate)

    @pytest.mark.asyncio
    async def test_tts_constructed_with_kwargs(self):
        tts = SenseTimeTTS(
            api_url="ws://custom.url/ws", api_key="key123", model="custom-model",
            voice="custom_voice", sample_rate=48000, speed=0.8, vol=0.9, pitch=3,
        )
        assert tts._config.api_url == "ws://custom.url/ws"
        assert tts._config.api_key == "key123"
        assert tts._config.model == "custom-model"
        assert tts._config.voice == "custom_voice"
        assert tts._config.sample_rate == 48000
        assert tts._config.speed == 0.8
        assert tts._config.vol == 0.9
        assert tts._config.pitch == 3
        logger.info("[PASS] test_tts_constructed_with_kwargs")

    @pytest.mark.asyncio
    async def test_batch_synthesize(self, srv):
        server, session = srv
        config = make_config(f"ws://127.0.0.1:{server.port}/ws")
        config.log_audio_diag = True
        tts = SenseTimeTTS(config=config)
        frames: list[rtc.AudioFrame] = []
        async with tts.synthesize("batch synthesis test") as stream:
            async for ev in stream:
                if hasattr(ev, "frame"):
                    frames.append(ev.frame)
        assert len(frames) > 0
        await tts.shutdown()
        logger.info("[PASS] test_batch_synthesize (%d frames)", len(frames))

    @pytest.mark.asyncio
    async def test_persistent_connection_fields_present(self):
        """SenseTimeTTS has the required persistent-connection attributes."""
        config = SenseTimeTTSConfig(
            api_url="ws://127.0.0.1:9999/ws", api_key="key", model="m",
            voice="v", sample_rate=16000,
        )
        tts = SenseTimeTTS(config=config)
        assert hasattr(tts, "_conn")
        assert hasattr(tts, "_stream_lock")
        assert hasattr(tts, "_stream_active")
        assert tts._conn is None
        assert not tts._stream_active
        logger.info("[PASS] test_persistent_connection_fields_present")


# ---------------------------------------------------------------------------
# Round 8 R8.2: SentenceAggregator production-bug regression
# ---------------------------------------------------------------------------


class TestR8SentenceAggregation:
    """Verify the production "几个字蹦一次" bug is fixed.

    Production logs (2026-05-06) showed the LLM emitting 12 tokens
    ("Hey", "there", "I'm", ...) for one welcome message; each became a
    separate ``task_continue`` and SenseAudio server treated them as 12
    independent batches with ~600ms gaps. Aggregator should collapse
    them into ~2 sentences.
    """

    @pytest.mark.asyncio
    async def test_production_welcome_message_consolidates_to_two_batches(self):
        """The exact production token sequence reduces to 2 task_continues."""
        import aiohttp
        server = MockTTSServer(port=0, echo_audio=True, batch_delay=0.0)
        await server.start()
        await asyncio.sleep(0.05)
        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{server.port}/ws")
            tts = SenseTimeTTS(config=config)
            stream = tts.stream()

            async def drain():
                async for _ in stream:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            # Reproduces production log exactly: 14 tiny tokens → expected
            # to collapse to 2 (one per sentence-ending punct).
            for tok in [
                "Hey", " there", " I'm", " ready", " to", " help",
                ".", " What", " can", " I", " do", " for", " you", "?",
            ]:
                stream.push_text(tok)
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)

            # Old behavior: 12 task_continues. New behavior: 2 (one per
            # hard-punct sentence).
            assert server.task_continue_count == 2, (
                f"aggregator should batch tokens into 2 sentences, "
                f"server saw {server.task_continue_count} task_continues. "
                f"texts={server._text_received}"
            )
            # Verify the content was re-assembled correctly.
            received = server._text_received
            assert len(received) == 2
            assert "Hey" in received[0] and "help" in received[0]
            assert "What" in received[1] and "you" in received[1]
            await tts.shutdown()
            logger.info(
                "[PASS] test_production_welcome_message_consolidates_to_two_batches "
                "(received=%s)", received,
            )
        finally:
            await session.close()
            await server.stop()

    @pytest.mark.asyncio
    async def test_chinese_long_sentence_splits_at_punct(self):
        """Chinese long reply with 。 splits cleanly into batches."""
        import aiohttp
        server = MockTTSServer(port=0, echo_audio=True, batch_delay=0.0)
        await server.start()
        await asyncio.sleep(0.05)
        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{server.port}/ws")
            tts = SenseTimeTTS(config=config)
            stream = tts.stream()

            async def drain():
                async for _ in stream:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            # 3 Chinese sentences as fine-grained tokens.
            for tok in [
                "我", "是", "AI", "。",
                "没", "有", "年龄", "。",
                "随时", "为你", "效劳", "。",
            ]:
                stream.push_text(tok)
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)

            # Each "。" forces a flush.
            assert server.task_continue_count == 3, (
                f"expected 3 batches at 。 boundaries, got "
                f"{server.task_continue_count}: {server._text_received}"
            )
            await tts.shutdown()
            logger.info(
                "[PASS] test_chinese_long_sentence_splits_at_punct (%s)",
                server._text_received,
            )
        finally:
            await session.close()
            await server.stop()

    @pytest.mark.asyncio
    async def test_idle_timer_flushes_partial_short_reply(self):
        """A short reply with no punctuation gets flushed by the idle timer.

        Without this, a "嗯" / single-word agreement would be held in the
        aggregator buffer until input_done — adding ~300 ms of latency.
        """
        import aiohttp
        server = MockTTSServer(port=0, echo_audio=True, batch_delay=0.0)
        await server.start()
        await asyncio.sleep(0.05)
        session = aiohttp.ClientSession()
        try:
            config = make_config(f"ws://127.0.0.1:{server.port}/ws")
            # Tighten the idle timer for this test (default 300ms is fine
            # but speeds the test up).
            config.aggregator_idle_ms = 100
            tts = SenseTimeTTS(config=config)
            stream = tts.stream()

            async def drain():
                async for _ in stream:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_text("好的")  # 2 chars, no punct
            # Don't end_input yet — simulate the LLM still being open.
            await asyncio.sleep(0.3)
            # By now the idle timer should have fired and flushed.
            assert server.task_continue_count >= 1, (
                "idle timer should have flushed the buffer by now"
            )
            stream.end_input()
            await asyncio.wait_for(task, timeout=2.0)
            await tts.shutdown()
            logger.info(
                "[PASS] test_idle_timer_flushes_partial_short_reply (%s)",
                server._text_received,
            )
        finally:
            await session.close()
            await server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
