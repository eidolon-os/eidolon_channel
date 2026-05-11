"""Unit tests for SenseTime STT persistent-connection model.

Architecture under test (mirrors SenseTime TTS post-refactor):

    warmup()   → connect → connected_success            (task_start NOT sent here)
    stream()   → task_start → audio (binary) → task_finish → task_finished
    shutdown() → disconnect

    WS-level ping/pong (configured inside ``websockets.connect``) is the only
    keep-alive. No application-level heartbeat.

Run with::

    cd <repository-root>
    .venv/bin/python -m pytest \
        eidolon/livekit/tests/stt/sensetime/test_stt_persistent.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
import websockets

_root = Path(__file__).resolve().parents[5]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

from livekit import rtc

from eidolon.livekit.plugins.stt.sensetime import (
    SenseTimeSTT,
    SenseTimeSTTConfig,
    SenseTimeSTTError,
)
from eidolon.livekit.plugins.stt.sensetime.connection import STTConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(name)-40s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("test_stt_persistent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pcm(num_samples: int = 1600) -> bytes:
    """Deterministic PCM bytes (100 ms @ 16 kHz mono = 1600 samples)."""
    return np.int16([256] * num_samples).tobytes()


def make_audio_frame(num_samples: int = 1600) -> rtc.AudioFrame:
    arr = np.int16([256] * num_samples)
    return rtc.AudioFrame(
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=num_samples,
        data=arr.tobytes(),
    )


def make_config(api_url: str) -> SenseTimeSTTConfig:
    return SenseTimeSTTConfig(
        api_url=api_url,
        api_key="test-key",
        model="senseaudio-asr-deepthink-1.5-260319",
        sample_rate=16000,
        language="zh",
    )


def _make_conn(uri: str) -> STTConnection:
    return STTConnection(
        uri=uri, api_key="test-key", model="m", sample_rate=16000, language="zh",
    )


# ---------------------------------------------------------------------------
# Mock SenseAudio STT WebSocket Server
# ---------------------------------------------------------------------------

class MockSTTServer:
    """Mock server that follows the SenseAudio STT protocol.

    Per protocol:
      - Server sends: connected_success / task_started / result_final / task_finished / task_failed
      - Client sends: task_start (JSON) / audio (binary) / task_finish (JSON)
      - WS stays open across multiple tasks for the same connection

    Configuration:
      - ``echo_transcript``: send a result_final after each task_finish
      - ``segment_delay``: sleep before sending result_final (simulates ASR latency)
      - ``task_failure``: send task_failed instead of task_started
      - ``silent``: never send result_final after task_finish
      - ``close_after_connected``: close immediately after connected_success
    """

    def __init__(
        self,
        port: int = 0,
        echo_transcript: bool = True,
        segment_delay: float = 0.0,
        task_failure: bool = False,
        silent: bool = False,
        close_after_connected: bool = False,
    ):
        self.port = port
        self.echo_transcript = echo_transcript
        self.segment_delay = segment_delay
        self.task_failure = task_failure
        self.silent = silent
        self.close_after_connected = close_after_connected
        self._server: Any = None
        self._stop_evt = asyncio.Event()

        # Counters
        self.task_start_count: int = 0
        self.task_finish_count: int = 0
        self.connections_total: int = 0
        self.binary_frames_received: int = 0
        self.binary_bytes_received: int = 0
        self.json_messages_received: int = 0
        # Recorded JSON events (for debugging)
        self.received_events: list[str] = []

    async def start(self) -> None:
        async def handler(ws: Any) -> None:
            self.connections_total += 1
            session_id = f"sess-{self.connections_total}"

            await ws.send(json.dumps({
                "event": "connected_success",
                "session_id": session_id,
                "trace_id": session_id,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }))

            if self.close_after_connected:
                await ws.close()
                return

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        # Audio binary frame
                        self.binary_frames_received += 1
                        self.binary_bytes_received += len(raw)
                        continue

                    # JSON control frame
                    self.json_messages_received += 1
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event = msg.get("event", "")
                    self.received_events.append(event)

                    if event == "task_start":
                        self.task_start_count += 1
                        if self.task_failure:
                            await ws.send(json.dumps({
                                "event": "task_failed",
                                "session_id": session_id,
                                "base_resp": {
                                    "status_code": 2013,
                                    "status_msg": "mock task failure",
                                },
                            }))
                            await ws.close()
                            return
                        await ws.send(json.dumps({
                            "event": "task_started",
                            "session_id": session_id,
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))

                    elif event == "task_finish":
                        self.task_finish_count += 1
                        if self.silent:
                            # Truly silent: no result_final, no task_finished.
                            # Used to verify the 5 s no_first_transcript guard.
                            continue
                        if self.echo_transcript:
                            if self.segment_delay > 0:
                                await asyncio.sleep(self.segment_delay)
                            await ws.send(json.dumps({
                                "event": "result_final",
                                "session_id": session_id,
                                "data": {
                                    "text": f"hello world {self.task_finish_count}",
                                    "is_final": True,
                                    "segment_id": self.task_finish_count,
                                    "timestamp_end": int(time.time() * 1000),
                                },
                                "base_resp": {"status_code": 0, "status_msg": "success"},
                            }))
                        await ws.send(json.dumps({
                            "event": "task_finished",
                            "session_id": session_id,
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        # WS stays open per protocol
            except websockets.exceptions.ConnectionClosed:
                pass

        self._server = await websockets.serve(
            handler, "127.0.0.1", self.port,
            ping_interval=None, ping_timeout=None,
        )
        # If we asked for a random port, capture the assigned one
        sock = self._server.sockets[0]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def srv():
    server = MockSTTServer(port=0, echo_transcript=True)
    await server.start()
    await asyncio.sleep(0.05)
    yield server
    await server.stop()


@pytest_asyncio.fixture
async def failing_srv():
    server = MockSTTServer(port=0, task_failure=True)
    await server.start()
    await asyncio.sleep(0.05)
    yield server
    await server.stop()


@pytest_asyncio.fixture
async def close_srv():
    server = MockSTTServer(port=0, close_after_connected=True)
    await server.start()
    await asyncio.sleep(0.05)
    yield server
    await server.stop()


# ---------------------------------------------------------------------------
# STTConnection: lifecycle smoke tests
# ---------------------------------------------------------------------------

class TestSTTConnectionLifecycle:

    @pytest.mark.asyncio
    async def test_connect_success(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        ok = await conn.connect()
        assert ok
        assert conn.is_connected
        assert conn._state == "ready"
        await conn.disconnect()
        logger.info("[PASS] test_connect_success")

    @pytest.mark.asyncio
    async def test_disconnect(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        assert conn.is_connected
        await conn.disconnect()
        assert not conn.is_connected
        assert conn._state == "disconnected"
        logger.info("[PASS] test_disconnect")

    @pytest.mark.asyncio
    async def test_task_start_awaits_ack(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        await conn.send_task_start()
        # task_started event was set by recv loop synchronously before
        # send_task_start returned (it awaits the ack within 5 s).
        assert conn._task_started_event.is_set()
        assert conn._state == "task_active"
        assert srv.task_start_count == 1
        await conn.disconnect()
        logger.info("[PASS] test_task_start_awaits_ack")

    @pytest.mark.asyncio
    async def test_task_finish_transitions_to_finishing(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        await conn.send_task_start()
        await conn.send_task_finish()
        assert conn._state == "finishing"
        # Wait for server's task_finished
        try:
            await asyncio.wait_for(conn._task_finished_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("task_finished_event never set")
        await conn.disconnect()
        logger.info("[PASS] test_task_finish_transitions_to_finishing")

    @pytest.mark.asyncio
    async def test_double_disconnect_is_safe(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        await conn.disconnect()
        await conn.disconnect()  # idempotent
        logger.info("[PASS] test_double_disconnect_is_safe")

    @pytest.mark.asyncio
    async def test_send_audio_while_disconnected_raises(self):
        conn = _make_conn("ws://localhost:1/ws")
        with pytest.raises(SenseTimeSTTError):
            await conn.send_audio_binary(make_pcm())
        logger.info("[PASS] test_send_audio_while_disconnected_raises")


# ---------------------------------------------------------------------------
# Reconnect / ensure_connected
# ---------------------------------------------------------------------------

class TestSTTConnectionReconnect:

    @pytest.mark.asyncio
    async def test_ensure_connected_when_ready(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        ok = await conn.ensure_connected()
        assert ok
        assert conn.is_connected
        await conn.disconnect()
        logger.info("[PASS] test_ensure_connected_when_ready")

    @pytest.mark.asyncio
    async def test_ensure_connected_reconnects_after_force_close(self, srv):
        conn = _make_conn(f"ws://localhost:{srv.port}/ws")
        await conn.connect()
        await conn._force_close()
        assert not conn.is_connected
        ok = await conn.ensure_connected()
        assert ok
        assert conn.is_connected
        # Server saw two separate connections
        assert srv.connections_total == 2
        await conn.disconnect()
        logger.info("[PASS] test_ensure_connected_reconnects_after_force_close")


# ---------------------------------------------------------------------------
# SenseTimeSTT: warmup / shutdown / persistent connection
# ---------------------------------------------------------------------------

class TestSenseTimeSTTPersistentConnection:

    @pytest.mark.asyncio
    async def test_warmup_does_not_send_task_start(self, srv):
        """warmup opens the WS but task_start is per-utterance, not session-wide."""
        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        assert stt._conn is not None
        assert stt._conn.is_connected
        # Allow the recv loop to settle
        await asyncio.sleep(0.05)
        # CRITICAL: warmup must NOT send task_start
        assert srv.task_start_count == 0, (
            f"warmup should not send task_start, got {srv.task_start_count}"
        )
        await stt.shutdown()
        logger.info("[PASS] test_warmup_does_not_send_task_start")

    @pytest.mark.asyncio
    async def test_warmup_is_idempotent(self, srv):
        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        first_conn = stt._conn
        await stt.warmup()
        assert stt._conn is first_conn  # same instance
        assert srv.connections_total == 1
        await stt.shutdown()
        logger.info("[PASS] test_warmup_is_idempotent")

    @pytest.mark.asyncio
    async def test_shutdown_closes_connection(self, srv):
        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        await stt.shutdown()
        assert stt._conn is None
        logger.info("[PASS] test_shutdown_closes_connection")

    @pytest.mark.asyncio
    async def test_shutdown_safe_when_not_warmed_up(self):
        config = make_config("ws://localhost:9999/ws")
        stt = SenseTimeSTT(config=config)
        await stt.shutdown()  # should not raise
        logger.info("[PASS] test_shutdown_safe_when_not_warmed_up")

    @pytest.mark.asyncio
    async def test_persistent_connection_fields_present(self):
        """Sanity check: stream_lock and _ensure_conn are wired up.

        Round 8 R8.11: ``_stream_active`` field was removed (dead — only
        set, never read). TTS still has its own ``_stream_active`` for
        heartbeat suppression; STT doesn't need it because its heartbeat
        is WS-protocol level (``ping_interval`` on aiohttp).
        """
        config = make_config("ws://localhost:9999/ws")
        stt = SenseTimeSTT(config=config)
        assert isinstance(stt._stream_lock, asyncio.Lock)
        assert hasattr(stt, "_ensure_conn")
        assert hasattr(stt, "warmup")
        assert hasattr(stt, "shutdown")
        logger.info("[PASS] test_persistent_connection_fields_present")


# ---------------------------------------------------------------------------
# Streaming (the protocol-correct, event-driven, no-polling refactor)
# ---------------------------------------------------------------------------

class TestSpeechStream:

    @pytest.mark.asyncio
    async def test_stream_basic_emits_final_transcript(self, srv):
        """Push audio + flush → server returns result_final + task_finished."""
        from livekit.agents.stt import SpeechEventType

        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        stream = stt.stream()

        finals: list[str] = []
        end_of_speech_count = 0

        async def drain():
            nonlocal end_of_speech_count
            try:
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)
                    elif ev.type == SpeechEventType.END_OF_SPEECH:
                        end_of_speech_count += 1
            except Exception:
                logger.exception("drain error")

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        # Push 3 frames worth of audio (300 ms total) then flush
        for _ in range(3):
            stream.push_frame(make_audio_frame())
        stream.flush()
        stream.end_input()

        await asyncio.wait_for(task, timeout=5.0)

        assert srv.task_start_count == 1
        assert srv.task_finish_count == 1
        # Server received audio as binary
        assert srv.binary_frames_received >= 1
        assert srv.binary_bytes_received >= 3 * 3200  # 3 × 100 ms × 32 bytes/ms
        # Got final + end_of_speech
        assert len(finals) == 1, f"expected 1 final, got {finals}"
        assert end_of_speech_count == 1
        await stt.shutdown()
        logger.info("[PASS] test_stream_basic_emits_final_transcript (final=%r)", finals)

    @pytest.mark.asyncio
    async def test_audio_sent_as_binary_not_json(self, srv):
        """Protocol verification: audio MUST be binary frames, not JSON+hex."""
        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        stream = stt.stream()

        async def drain():
            try:
                async for _ in stream:
                    pass
            except Exception:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        for _ in range(2):
            stream.push_frame(make_audio_frame())
        stream.flush()
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)

        # The protocol MUST be binary frames for audio.
        assert srv.binary_frames_received >= 1, (
            "audio should arrive as binary frames"
        )
        # JSON frames should be: task_start + task_finish only (no task_continue)
        assert "task_continue" not in srv.received_events, (
            f"STT must not send task_continue events, got {srv.received_events}"
        )
        assert set(srv.received_events) <= {"task_start", "task_finish"}, (
            f"unexpected JSON events: {srv.received_events}"
        )
        await stt.shutdown()
        logger.info(
            "[PASS] test_audio_sent_as_binary_not_json (binary_frames=%d events=%s)",
            srv.binary_frames_received, srv.received_events,
        )

    @pytest.mark.asyncio
    async def test_stream_task_failure_propagates(self, failing_srv):
        from livekit.agents.types import APIConnectOptions

        config = make_config(f"ws://localhost:{failing_srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        # Disable framework retries so the error surfaces directly
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=0, timeout=10.0),
        )

        captured: list[Exception] = []

        async def drain():
            try:
                async for _ in stream:
                    pass
            except Exception as e:
                captured.append(e)

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_frame(make_audio_frame())
        stream.flush()
        stream.end_input()
        await asyncio.wait_for(task, timeout=5.0)

        # Task failed before audio came back — should propagate as error
        assert len(captured) > 0, "expected an exception"
        await stt.shutdown()
        logger.info(
            "[PASS] test_stream_task_failure_propagates (err=%s)", captured[0],
        )

    @pytest.mark.asyncio
    async def test_sequential_streams_reuse_same_connection(self, srv):
        """Two sequential utterances → one WS, two task_start/task_finish pairs."""
        from livekit.agents.stt import SpeechEventType

        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()

        for i in range(2):
            stream = stt.stream()
            finals: list[str] = []

            async def drain(s=stream, f=finals):
                async for ev in s:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        f.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_frame(make_audio_frame())
            stream.flush()
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            assert len(finals) == 1, f"utterance {i}: expected 1 final, got {finals}"

        # Single connection, two task lifecycles
        assert srv.connections_total == 1
        assert srv.task_start_count == 2
        assert srv.task_finish_count == 2
        await stt.shutdown()
        logger.info(
            "[PASS] test_sequential_streams_reuse_same_connection "
            "(connections=%d task_starts=%d)",
            srv.connections_total, srv.task_start_count,
        )


# ---------------------------------------------------------------------------
# Event-driven exit verification (mirrors TTS bugfix tests)
# ---------------------------------------------------------------------------

class TestEventDrivenExit:
    """The exit logic must be event-driven, not time-window based."""

    @pytest.mark.asyncio
    async def test_task_finished_triggers_exit(self, srv):
        """Condition B (canonical): server task_finished → immediate exit."""
        config = make_config(f"ws://localhost:{srv.port}/ws")
        stt = SenseTimeSTT(config=config)
        await stt.warmup()
        stream = stt.stream()

        async def drain():
            async for _ in stream:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        stream.push_frame(make_audio_frame())
        stream.flush()
        stream.end_input()

        t0 = time.time()
        await asyncio.wait_for(task, timeout=2.0)
        elapsed = time.time() - t0

        # Should exit < 200 ms after task_finish round-trip (server delay = 0).
        assert elapsed < 0.5, (
            f"event-driven exit should be near-instant; got {elapsed:.3f}s"
        )
        await stt.shutdown()
        logger.info(
            "[PASS] test_task_finished_triggers_exit (elapsed=%.3fs)", elapsed,
        )

    @pytest.mark.asyncio
    async def test_inter_segment_delay_does_not_break_exit(self):
        """Delays between flush and result_final must not break exit logic.

        Unlike TTS which had to count batch_ends, STT relies on task_finished
        as the canonical signal — so even a 1 s ASR processing delay is fine.
        """
        from livekit.agents.stt import SpeechEventType

        server = MockSTTServer(port=0, echo_transcript=True, segment_delay=1.0)
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []

            async def drain():
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_frame(make_audio_frame())
            stream.flush()
            stream.end_input()
            await asyncio.wait_for(task, timeout=5.0)
            assert len(finals) == 1, (
                f"expected 1 final after 1 s server delay, got {finals}"
            )
            await stt.shutdown()
            logger.info(
                "[PASS] test_inter_segment_delay_does_not_break_exit"
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_silent_server_cleanup_ack_timeout(self):
        """Round 8: when the server never returns task_finished after we
        send task_finish in cleanup, the 5 s cleanup-ack timeout still
        bounds shutdown time.

        Mock server is silent (never echoes anything). Stream sends a
        frame, framework calls end_input → input_ch closes → _send_loop
        exits → _input_done sets _exit_event → finally clause sends
        task_finish, waits up to 5s for task_finished ack (silent
        server never replies) → forces close. Total elapsed time should
        be ~5 s (the cleanup-ack timeout), not 30 s+ (no longer relevant
        with the safety net removed).
        """
        from livekit.agents.types import APIConnectOptions

        server = MockSTTServer(port=0, silent=True)
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            stream = stt.stream(
                conn_options=APIConnectOptions(max_retry=0, timeout=30.0),
            )

            captured: list[Exception] = []

            async def drain():
                try:
                    async for _ in stream:
                        pass
                except Exception as e:
                    captured.append(e)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            stream.push_frame(make_audio_frame())
            stream.flush()
            stream.end_input()

            t0 = time.time()
            await asyncio.wait_for(task, timeout=10.0)
            elapsed = time.time() - t0
            # input_done immediate exit + 2s send-cancel + 5s ack-timeout
            assert 4.5 <= elapsed <= 8.0, (
                f"cleanup-ack timeout should bound shutdown to ~5-7s; "
                f"got {elapsed:.3f}s"
            )
            await stt.shutdown()
            logger.info(
                "[PASS] test_silent_server_cleanup_ack_timeout "
                "(elapsed=%.3fs)", elapsed,
            )
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# State-sync bridge (Round 7 G11)
# ---------------------------------------------------------------------------


class TestStateSyncBridge:
    """Round 7 G11 — verify ``signal_user_away`` aborts in-flight stream
    immediately, instead of waiting for the 30 s safety net.

    This is the key architectural fix: framework's high-level user_state
    drives plugin lifecycle, not just low-level VAD events.
    """

    @pytest.mark.asyncio
    async def test_signal_user_away_sets_event(self):
        """Direct API: signal_user_away sets the underlying asyncio.Event."""
        config = make_config("ws://localhost:9999/ws")
        stt = SenseTimeSTT(config=config)
        assert not stt._user_away_event.is_set()
        stt.signal_user_away()
        assert stt._user_away_event.is_set()
        # Idempotent
        stt.signal_user_away()
        assert stt._user_away_event.is_set()

    @pytest.mark.asyncio
    async def test_signal_user_present_clears_event(self):
        """Reverse signal: signal_user_present clears the event."""
        config = make_config("ws://localhost:9999/ws")
        stt = SenseTimeSTT(config=config)
        stt.signal_user_away()
        assert stt._user_away_event.is_set()
        stt.signal_user_present()
        assert not stt._user_away_event.is_set()

    @pytest.mark.asyncio
    async def test_user_away_does_NOT_abort_stream(self):
        """Round 8 R8.8: ``user_state="away"`` is a 15 s wall-clock signal
        (mutual silence), NOT "user actually left". Exiting on it kills
        the entire session's STT — framework's ``_STTPipeline`` doesn't
        recreate streams.

        This test inverts the old G11 invariant: now the stream MUST
        stay alive after user_away signal, so the user can come back
        later and still be transcribed.
        """
        from livekit.agents.types import APIConnectOptions

        server = MockSTTServer(port=0, silent=True)
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            stream = stt.stream(
                conn_options=APIConnectOptions(max_retry=0, timeout=30.0),
            )

            async def drain():
                try:
                    async for _ in stream:
                        pass
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)
            for _ in range(3):
                stream.push_frame(make_audio_frame())
            await asyncio.sleep(0.2)

            # Signal user_away — stream MUST stay alive
            stt.signal_user_away()
            await asyncio.sleep(0.3)  # plenty of time for any (wrong) exit
            assert not task.done(), (
                "stream must NOT exit on user_away (R8.8 architectural fix)"
            )

            # Signal user_present and verify stream is still healthy
            stt.signal_user_present()
            await asyncio.sleep(0.1)
            assert not task.done(), "stream still alive after present signal"

            # Cleanup: close the stream properly so the test can finish
            await stream.aclose()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass

            await stt.shutdown()
            logger.info("[PASS] test_user_away_does_NOT_abort_stream")
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_user_away_event_cleared_at_stream_start(self):
        """Each new stream starts with a fresh state — a stale user_away
        signal from a prior stream must not abort a new stream."""
        from livekit.agents.types import APIConnectOptions
        from livekit.agents.stt import SpeechEventType

        server = MockSTTServer(port=0, echo_transcript=True)
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)

            # Simulate stale away signal from prior stream / state.
            stt.signal_user_away()
            assert stt._user_away_event.is_set()

            stream = stt.stream()
            finals = []

            async def drain():
                try:
                    async for ev in stream:
                        if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                            finals.append(ev.alternatives[0].text)
                except Exception:
                    pass

            task = asyncio.create_task(drain())
            await asyncio.sleep(0.05)  # let _run() start (clears the event)
            stream.push_frame(make_audio_frame())
            stream.flush()
            stream.end_input()

            # Should complete normally — get a final transcript
            await asyncio.wait_for(task, timeout=5.0)
            assert len(finals) == 1, f"expected 1 final, got {finals}"
            await stt.shutdown()
            logger.info(
                "[PASS] test_user_away_event_cleared_at_stream_start"
            )
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_properties(self):
        config = make_config("ws://localhost:9999/ws")
        stt = SenseTimeSTT(config=config)
        assert stt.provider == "sensetime"
        assert "SenseTime STT" in stt.label
        assert stt.api_url == config.api_url
        assert stt.api_key == "test-key"
        assert stt.language == "zh"
        assert stt.sample_rate == 16000
        logger.info("[PASS] test_properties")

    @pytest.mark.asyncio
    async def test_constructed_with_kwargs(self):
        stt = SenseTimeSTT(
            api_url="ws://x/ws",
            api_key="k",
            model="m",
            sample_rate=16000,
            language="en",
        )
        assert stt.language == "en"
        logger.info("[PASS] test_constructed_with_kwargs")


# ---------------------------------------------------------------------------
# Round 8: Session-long multi-utterance behaviour
# ---------------------------------------------------------------------------


class MockSessionLongSTTServer:
    """Mock server modeling SenseAudio's session-long task semantics.

    Each task (one task_start..task_finish pair) hosts MANY result_finals,
    each one representing a server-VAD-segmented utterance. The client
    sends audio continuously; the server fires result_finals on a list of
    pre-scripted offsets after task_started.

    Configuration:
        utterance_offsets — list of (delay_after_start_s, text) pairs
        emit_task_finished_on_finish — whether to ack task_finish (default True)
    """

    def __init__(
        self,
        port: int = 0,
        *,
        utterance_offsets: list[tuple[float, str]] | None = None,
        emit_task_finished_on_finish: bool = True,
    ) -> None:
        self.port = port
        self.utterance_offsets = utterance_offsets or [
            (0.3, "第一句话"),
            (0.6, "第二句话"),
            (0.9, "第三句话"),
        ]
        self.emit_task_finished_on_finish = emit_task_finished_on_finish
        self._server: Any = None

        # Counters — observable from tests.
        self.task_start_count: int = 0
        self.task_finish_count: int = 0
        self.connections_total: int = 0
        self.binary_frames_received: int = 0
        self.binary_bytes_received: int = 0
        self.result_finals_emitted: int = 0

    async def start(self) -> None:
        async def handler(ws: Any) -> None:
            self.connections_total += 1
            session_id = f"sess-long-{self.connections_total}"

            await ws.send(json.dumps({
                "event": "connected_success",
                "session_id": session_id,
                "trace_id": session_id,
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }))

            utterance_task: asyncio.Task[None] | None = None

            async def emit_utterances_after_start() -> None:
                """Fire result_finals at the scripted offsets after task_started."""
                t0 = asyncio.get_event_loop().time()
                for delay, text in self.utterance_offsets:
                    target = t0 + delay
                    while True:
                        wait = target - asyncio.get_event_loop().time()
                        if wait <= 0:
                            break
                        await asyncio.sleep(wait)
                    try:
                        await ws.send(json.dumps({
                            "event": "result_final",
                            "session_id": session_id,
                            "data": {
                                "text": text,
                                "is_final": True,
                                "segment_id": self.result_finals_emitted,
                                "timestamp_end": int(time.time() * 1000),
                            },
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        self.result_finals_emitted += 1
                    except websockets.exceptions.ConnectionClosed:
                        return

            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        self.binary_frames_received += 1
                        self.binary_bytes_received += len(raw)
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    event = msg.get("event")
                    if event == "task_start":
                        self.task_start_count += 1
                        await ws.send(json.dumps({
                            "event": "task_started",
                            "session_id": session_id,
                            "base_resp": {"status_code": 0, "status_msg": "success"},
                        }))
                        utterance_task = asyncio.create_task(
                            emit_utterances_after_start()
                        )
                    elif event == "task_finish":
                        self.task_finish_count += 1
                        if utterance_task and not utterance_task.done():
                            utterance_task.cancel()
                        if self.emit_task_finished_on_finish:
                            await ws.send(json.dumps({
                                "event": "task_finished",
                                "session_id": session_id,
                                "base_resp": {"status_code": 0, "status_msg": "success"},
                            }))
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if utterance_task and not utterance_task.done():
                    utterance_task.cancel()

        self._server = await websockets.serve(
            handler, "127.0.0.1", self.port,
            ping_interval=None, ping_timeout=None,
        )
        sock = self._server.sockets[0]
        self.port = sock.getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


class TestSessionLongMultiUtterance:
    """Round 8 R8.1 — verify the session-long task model.

    These tests focus on the architectural fix: ONE task_start covers the
    entire session, hosting many result_finals (server-VAD-segmented
    utterances). The 30s safety net is gone; only ``input_done``,
    ``user_away``, ``conn_closed``, and ``task_failed`` exit the stream.
    """

    @pytest.mark.asyncio
    async def test_three_utterances_one_task(self):
        """Server emits 3 result_finals over the lifetime of one task.
        All three must reach the framework as FINAL_TRANSCRIPT events,
        and exactly one task_start / one task_finish should be observed.
        """
        from livekit.agents.stt import SpeechEventType

        server = MockSessionLongSTTServer(
            port=0,
            utterance_offsets=[
                (0.2, "第一段"),
                (0.5, "第二段"),
                (0.8, "第三段"),
            ],
        )
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []
            end_of_speech_count = 0

            async def drain():
                nonlocal end_of_speech_count
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)
                    elif ev.type == SpeechEventType.END_OF_SPEECH:
                        end_of_speech_count += 1

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)

            # Push frames continuously for ~1.2s (cover all 3 utterances).
            for _ in range(12):
                stream.push_frame(make_audio_frame())
                await asyncio.sleep(0.1)

            # End the session — framework would do this on AgentSession close.
            stream.end_input()
            await asyncio.wait_for(task, timeout=10.0)

            assert finals == ["第一段", "第二段", "第三段"], (
                f"expected 3 finals in one task, got {finals}"
            )
            # Per-utterance END_OF_SPEECH (server VAD signal) — one per final.
            assert end_of_speech_count == 3, (
                f"expected 3 END_OF_SPEECH (one per utterance), got "
                f"{end_of_speech_count}"
            )
            assert server.task_start_count == 1, (
                f"expected ONE task_start (session-long), got "
                f"{server.task_start_count}"
            )
            assert server.task_finish_count == 1, (
                f"expected ONE task_finish (cleanup only), got "
                f"{server.task_finish_count}"
            )
            await stt.shutdown()
            logger.info(
                "[PASS] test_three_utterances_one_task "
                "(finals=%d eos=%d task_starts=%d task_finishes=%d)",
                len(finals), end_of_speech_count,
                server.task_start_count, server.task_finish_count,
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_long_idle_does_not_kill_stream(self):
        """Stream stays alive through a long idle period between utterances.

        Under the old per-utterance model with 30s safety net, this would
        terminate after 30s and lose subsequent utterances. Under the
        session-long model, it must keep going.
        """
        from livekit.agents.stt import SpeechEventType

        # Two utterances 8s apart — short enough for a fast test, long
        # enough to prove the safety net is gone.
        server = MockSessionLongSTTServer(
            port=0,
            utterance_offsets=[(0.3, "前一句"), (8.0, "后一句")],
        )
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []

            async def drain():
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)

            # Send a frame periodically for 9s.
            for _ in range(90):
                stream.push_frame(make_audio_frame())
                await asyncio.sleep(0.1)

            stream.end_input()
            await asyncio.wait_for(task, timeout=15.0)

            assert finals == ["前一句", "后一句"], (
                f"both utterances should arrive across the 8s gap, "
                f"got {finals}"
            )
            assert server.task_start_count == 1
            await stt.shutdown()
            logger.info(
                "[PASS] test_long_idle_does_not_kill_stream (finals=%s)",
                finals,
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_user_away_during_session_long_stream_does_NOT_kill_it(self):
        """Round 8 R8.8: in session-long mode, user_away must NOT terminate
        the stream — the user could come back and need transcription.

        Was: ``test_user_away_during_session_long_stream_aborts_fast`` —
        the old G11 invariant that R8.8 inverted because killing the
        stream killed the entire session's STT (no way to revive).
        """
        from livekit.agents.stt import SpeechEventType

        server = MockSessionLongSTTServer(
            port=0,
            utterance_offsets=[(0.2, "第一段"), (3.0, "第二段")],
        )
        await server.start()
        try:
            config = make_config(f"ws://localhost:{server.port}/ws")
            stt = SenseTimeSTT(config=config)
            await stt.warmup()
            stream = stt.stream()

            finals: list[str] = []

            async def drain():
                async for ev in stream:
                    if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                        finals.append(ev.alternatives[0].text)

            task = asyncio.create_task(drain())
            await asyncio.sleep(0)

            # Feed frames, get first utterance.
            for _ in range(5):
                stream.push_frame(make_audio_frame())
                await asyncio.sleep(0.1)
            t_wait = time.time()
            while not finals and time.time() - t_wait < 2.0:
                await asyncio.sleep(0.05)
            assert finals == ["第一段"]

            # Signal user_away — stream must STAY ALIVE
            stt.signal_user_away()
            await asyncio.sleep(0.5)
            assert not task.done(), (
                "stream must not exit on user_away in session-long mode"
            )

            # User comes back: feed more frames, expect second utterance
            for _ in range(8):
                stream.push_frame(make_audio_frame())
                await asyncio.sleep(0.1)
            t_wait = time.time()
            while len(finals) < 2 and time.time() - t_wait < 5.0:
                await asyncio.sleep(0.1)
            assert finals == ["第一段", "第二段"], (
                f"expected both utterances after away→present, got {finals}"
            )

            stt.signal_user_present()
            await stream.aclose()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            await stt.shutdown()
            logger.info(
                "[PASS] test_user_away_during_session_long_stream_does_NOT_kill_it "
                "(finals=%s)", finals,
            )
        finally:
            await server.stop()
