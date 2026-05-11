"""Standalone tests for the Bailian FunASR STT plugin.

These tests spin up a local WebSocket mock server that simulates the
Bailian FunASR protocol, allowing the plugin to be tested without
network access or a real API key.

Run with::

    python -m pytest eidolon/livekit/plugins/bailian/test_stt.py -v

Requirements::

    pytest pytest-asyncio websockets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import uuid
from pathlib import Path
from typing import Any

import av
import pytest
import websockets
from websockets import ServerConnection

# Ensure the package is importable from the repo root
_root = Path(__file__).resolve().parents[6]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

from livekit import rtc
from livekit.agents import stt as lk_stt

from eidolon.livekit.plugins.stt.bailian import (
    BailianConnectionManager,
    BailianFunASRSTT,
)
from eidolon.livekit.plugins.stt.bailian.connection_manager import (
    BailianConnectionError,
)
from livekit.agents.utils.aio.channel import ChanClosed

_AUDIO_FILE = Path(__file__).resolve().parents[5] / "pipeline" / "data" / "vad.m4a"


def load_audio_chunks(
    path: Path,
    chunk_samples: int = 1600,
    target_sr: int = 16000,
) -> list[rtc.AudioFrame]:
    """Decode an m4a file and return audio as LiveKit AudioFrame chunks.

    Converts the audio to 16 kHz mono 16-bit PCM and splits it into
    ``chunk_samples``-size pieces (100 ms each at 16 kHz = 1600 samples).

    Args:
        path: Path to the audio file (m4a / aac).
        chunk_samples: Number of samples per output frame (default 1600 = 100 ms).
        target_sr: Target sample rate (default 16000).

    Returns:
        List of :class:`rtc.AudioFrame` objects ready to push into a stream.
    """
    ctx = av.open(str(path))
    stream = ctx.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )
    frames: list[rtc.AudioFrame] = []

    for packet in ctx.demux(stream):
        for frame in packet.decode():
            resampled = resampler.resample(frame)
            if resampled is not None:
                for rf in resampled:
                    samples = rf.to_ndarray()
                    # Split into fixed-size chunks
                    total = samples.shape[-1]
                    for offset in range(0, total, chunk_samples):
                        chunk = samples[..., offset : offset + chunk_samples]
                        if chunk.size == 0:
                            continue
                        frames.append(
                            rtc.AudioFrame(
                                data=bytearray(chunk.tobytes()),
                                sample_rate=target_sr,
                                num_channels=1,
                                samples_per_channel=chunk.shape[-1],
                            )
                        )

    # Flush the resampler
    flushed = resampler.resample(None)
    if flushed is not None:
        for rf in flushed:
            samples = rf.to_ndarray()
            for offset in range(0, samples.shape[-1], chunk_samples):
                chunk = samples[..., offset : offset + chunk_samples]
                if chunk.size == 0:
                    continue
                frames.append(
                    rtc.AudioFrame(
                        data=bytearray(chunk.tobytes()),
                        sample_rate=target_sr,
                        num_channels=1,
                        samples_per_channel=chunk.shape[-1],
                    )
                )

    ctx.close()
    return frames

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_bailian")


# ---------------------------------------------------------------------------
# Mock FunASR Server (websockets 16.x API)
# ---------------------------------------------------------------------------

_MOCK_PORT = 0  # 0 = let OS pick a free port


class MockFunASRServer:
    """A local WebSocket server that simulates the Bailian FunASR protocol.

    - On ``run-task`` (first JSON message) it replies with ``task-started``.
    - On binary audio it collects the chunks.
    - On ``finish-task`` action it sends ``task-finished`` and closes.

    Results are sent immediately (no delay) to make tests fast and deterministic.
    """

    def __init__(self, port: int = _MOCK_PORT):
        self.port = port
        self._server: websockets.WebSocketServer | None = None
        self._task_id: str = ""
        self._audio_received: list[bytes] = []

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "localhost", self.port)
        # Record the actual port the OS assigned
        if self._server.sockets:
            self.port = self._server.sockets[0].getsockname()[1]
        logger.info("Mock FunASR server started on port %d", self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("Mock FunASR server stopped")

    async def _handler(self, ws: ServerConnection) -> None:
        """WebSocket handler — websockets 16.x signature."""
        self._audio_received.clear()
        self._task_id = ""

        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    self._audio_received.append(raw)
                elif isinstance(raw, str):
                    msg = json.loads(raw)
                    await self._handle_message(ws, msg)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _handle_message(
        self, ws: ServerConnection, msg: dict[str, Any]
    ) -> None:
        header = msg.get("header", {})
        action = header.get("action", "")

        if action == "finish-task":
            await ws.send(json.dumps({
                "header": {
                    "event": "task-finished",
                    "task_id": self._task_id,
                    "request_id": "mock-req-001",
                },
                "payload": {},
            }))
            await ws.close()
            return

        if action == "run-task":
            self._task_id = header.get("task_id", "") or str(uuid.uuid4())[:32]
            # Send task-started
            await ws.send(json.dumps({
                "header": {
                    "event": "task-started",
                    "task_id": self._task_id,
                    "task_status": "RUNNING",
                },
                "payload": {},
            }))
            # Send all results immediately
            await self._send_results(ws)

    async def _send_results(self, ws: ServerConnection) -> None:
        """Send all simulated result-generated events immediately."""
        results = [
            {
                "header": {
                    "event": "result-generated",
                    "task_id": self._task_id,
                    "request_id": "mock-req-001",
                },
                "payload": {
                    "output": {
                        "sentence": {
                            "text": "你",
                            "begin_time": 0,
                            "end_time": 200,
                            "sentence_end": False,
                            "words": [
                                {"text": "你", "begin_time": 0, "end_time": 200}
                            ],
                        }
                    }
                }
            },
            {
                "header": {
                    "event": "result-generated",
                    "task_id": self._task_id,
                    "request_id": "mock-req-001",
                },
                "payload": {
                    "output": {
                        "sentence": {
                            "text": "你好",
                            "begin_time": 0,
                            "end_time": 400,
                            "sentence_end": False,
                            "words": [
                                {"text": "你", "begin_time": 0, "end_time": 200},
                                {"text": "好", "begin_time": 200, "end_time": 400},
                            ],
                        }
                    }
                }
            },
            {
                "header": {
                    "event": "result-generated",
                    "task_id": self._task_id,
                    "request_id": "mock-req-001",
                },
                "payload": {
                    "output": {
                        "sentence": {
                            "text": "你好北京",
                            "begin_time": 0,
                            "end_time": 800,
                            "sentence_end": True,
                            "text_with_punct": "你好北京。",
                            "words": [
                                {"text": "你", "begin_time": 0, "end_time": 200},
                                {"text": "好", "begin_time": 200, "end_time": 400},
                                {"text": "北", "begin_time": 400, "end_time": 600},
                                {"text": "京", "begin_time": 600, "end_time": 800},
                            ],
                        }
                    }
                }
            },
        ]
        for result in results:
            await ws.send(json.dumps(result))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def make_audio_frame(
    sample_rate: int = 16000,
    duration_ms: int = 100,
    num_channels: int = 1,
    amplitude: int = 0,
) -> rtc.AudioFrame:
    """Build a LiveKit AudioFrame (silent by default)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    if amplitude == 0:
        data = bytearray(b"\x00\x00" * num_samples * num_channels)
    else:
        import math
        data = bytearray()
        for i in range(num_samples):
            value = int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
            data.extend(struct.pack("<h", value) * num_channels)

    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=num_samples,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def mock_server():
    srv = MockFunASRServer(port=_MOCK_PORT)
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture(scope="module")
def audio_chunks():
    """Load pipeline test m4a when present; else synthetic frames for mock-server tests."""
    if _AUDIO_FILE.is_file():
        return load_audio_chunks(_AUDIO_FILE, chunk_samples=1600, target_sr=16000)
    return [make_audio_frame(duration_ms=100) for _ in range(50)]


# ---------------------------------------------------------------------------
# Connection Manager Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connection_manager_basic(mock_server):
    """connect() completes the run-task handshake and returns successfully."""
    conn = BailianConnectionManager(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
        model="fun-asr-realtime-2026-02-28",
    )

    received: list[dict[str, Any]] = []

    # connect() reads task-started internally, then calls the callback with it
    await conn.connect(message_callback=lambda m: received.append(m))

    assert conn._connected, "Connection should be marked as connected"
    assert conn._task_id != "", "task_id should be set"
    assert len(received) == 1, f"Should have received exactly one task-started event, got {len(received)}"
    assert received[0]["header"]["event"] == "task-started"

    await conn.close()


@pytest.mark.asyncio
async def test_connection_manager_send_audio(mock_server):
    """send_audio() delivers raw bytes over the WebSocket."""
    conn = BailianConnectionManager(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    await conn.connect()

    audio = b"\x01\x02\x03\x04"
    await conn.send_audio(audio)
    await asyncio.sleep(0.05)

    assert b"\x01\x02\x03\x04" in mock_server._audio_received, \
        f"Server should have received audio bytes. Received: {mock_server._audio_received}"

    await conn.close()


@pytest.mark.asyncio
async def test_connection_manager_finish(mock_server):
    """finish() sends a finish-task message and the server sends task-finished."""
    conn = BailianConnectionManager(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    await conn.connect()

    finish_received = asyncio.Event()

    async def on_msg(msg: dict[str, Any]) -> None:
        if msg.get("header", {}).get("event") == "task-finished":
            finish_received.set()

    recv_task = asyncio.create_task(conn.receive_loop(message_cb=on_msg))
    await conn.send_audio(b"\x00\x00" * 1600)
    await conn.finish()

    try:
        await asyncio.wait_for(finish_received.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        pytest.fail("task-finished event not received within timeout")
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        await conn.close()


# ---------------------------------------------------------------------------
# Speech Stream Tests
# ---------------------------------------------------------------------------

async def _wait_for_final(events: list[lk_stt.SpeechEvent], timeout: float = 3.0) -> None:
    """Poll until FINAL_TRANSCRIPT appears."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT for e in events):
            return
        await asyncio.sleep(0.05)
    raise asyncio.TimeoutError(
        f"FINAL_TRANSCRIPT not seen in {timeout}s. Got: {[e.type for e in events]}"
    )


async def _drain_stream(stream, events: list[lk_stt.SpeechEvent], errors: list[Exception]) -> None:
    try:
        async for ev in stream:
            events.append(ev)
    except Exception as e:
        errors.append(e)


@pytest.mark.asyncio
async def test_streaming_basic(mock_server, audio_chunks):
    """Push real audio frames and verify we receive INTERIM then FINAL_TRANSCRIPT."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )

    stream = stt.stream()
    events: list[lk_stt.SpeechEvent] = []
    errors: list[Exception] = []

    task = asyncio.create_task(_drain_stream(stream, events, errors))

    # Push the first few real audio chunks
    for frame in audio_chunks[:5]:
        stream.push_frame(frame)
        await asyncio.sleep(0.02)

    stream.end_input()

    try:
        await _wait_for_final(events, timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail(
            f"Timeout waiting for FINAL_TRANSCRIPT. Got events: {events}. "
            f"Errors: {errors}"
        )
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    event_types = [e.type for e in events]
    assert lk_stt.SpeechEventType.INTERIM_TRANSCRIPT in event_types, \
        f"Should have received INTERIM_TRANSCRIPT. Got: {event_types}"
    assert lk_stt.SpeechEventType.FINAL_TRANSCRIPT in event_types, \
        f"Should have received FINAL_TRANSCRIPT. Got: {event_types}"
    assert not errors, f"Should not have any errors: {errors}"


@pytest.mark.asyncio
async def test_streaming_flush(mock_server, audio_chunks):
    """flush() sends any buffered audio immediately."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    stream = stt.stream()

    task = asyncio.create_task(_drain_stream(stream, [], []))

    stream.push_frame(audio_chunks[0])  # push real audio
    stream.flush()
    stream.end_input()
    await asyncio.sleep(0.15)

    assert len(mock_server._audio_received) > 0, \
        "flush() should have sent audio to the server"

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_streaming_end_of_speech(mock_server, audio_chunks):
    """After end_input() the stream emits END_OF_SPEECH."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    stream = stt.stream()
    events: list[lk_stt.SpeechEvent] = []

    task = asyncio.create_task(_drain_stream(stream, events, []))

    for frame in audio_chunks[:5]:
        stream.push_frame(frame)
        await asyncio.sleep(0.02)

    stream.end_input()

    deadline = asyncio.get_event_loop().time() + 3.0
    while asyncio.get_event_loop().time() < deadline:
        if any(e.type == lk_stt.SpeechEventType.END_OF_SPEECH for e in events):
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f"END_OF_SPEECH not received. Got: {[e.type for e in events]}")

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_word_timestamps(mock_server, audio_chunks):
    """FINAL_TRANSCRIPT should include word-level TimedString data."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    stream = stt.stream()
    events: list[lk_stt.SpeechEvent] = []

    task = asyncio.create_task(_drain_stream(stream, events, []))

    for frame in audio_chunks[:10]:
        stream.push_frame(frame)
        await asyncio.sleep(0.02)

    stream.end_input()

    try:
        await _wait_for_final(events, timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Final event not received. Events: {events}")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    final_ev = next(
        e for e in events if e.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT
    )
    words = final_ev.alternatives[0].words
    assert words, "FINAL_TRANSCRIPT should contain word-level data"
    # Mock server sends 4 words: 你 好 北 京
    assert len(words) >= 4, f"Expected >=4 words, got {len(words)}: {words}"
    assert words[0]["text"] == "你"
    assert words[-1]["text"] == "京"


# ---------------------------------------------------------------------------
# STT Class Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_properties():
    """BailianFunASRSTT exposes correct provider/model/label."""
    stt = BailianFunASRSTT(
        api_url="ws://localhost:9999",  # not actually connected in this test
        api_key="test-key",
        model="fun-asr-realtime-2026-02-28",
        language="zh",
    )
    assert stt.provider == "bailian"
    assert stt.model == "fun-asr-realtime-2026-02-28"
    assert "Bailian" in stt.label


@pytest.mark.asyncio
async def test_batch_recognize(mock_server, audio_chunks):
    """recognize() (batch mode) returns a FINAL_TRANSCRIPT with text."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )

    # Concatenate all chunks into one frame (trim to even sample count)
    total_samples = sum(c.samples_per_channel for c in audio_chunks)
    total_samples = total_samples - (total_samples % 2)
    total_bytes = total_samples * 2
    combined = bytearray(total_bytes)
    offset = 0
    for c in audio_chunks:
        chunk_bytes = min(c.samples_per_channel * 2, total_bytes - offset)
        if chunk_bytes <= 0:
            break
        combined[offset : offset + chunk_bytes] = c.data[:chunk_bytes]
        offset += chunk_bytes
    frame = rtc.AudioFrame(
        data=combined,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=total_samples,
    )

    # Use stt.recognize() — the method now spawns a receive loop internally
    event = await stt.recognize(frame)
    assert event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT, \
        f"Expected FINAL_TRANSCRIPT, got {event.type}"
    assert len(event.alternatives) > 0
    text = event.alternatives[0].text
    assert text != "", f"Transcript text should not be empty, got: {text!r}"


# ---------------------------------------------------------------------------
# Connection Failure Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connection_failure_bad_url():
    """Connecting to a non-existent host raises BailianConnectionError."""
    conn = BailianConnectionManager(
        api_url="ws://localhost:59999",  # nothing listening here
        api_key="test-key",
    )
    try:
        await conn.connect()
        pytest.fail("Expected BailianConnectionError or OSError")
    except (BailianConnectionError, OSError):
        pass


@pytest.mark.asyncio
async def test_stream_after_end_is_noop(mock_server, audio_chunks):
    """Once end_input() is called, further push_frame raises ChanClosed."""
    stt = BailianFunASRSTT(
        api_url=f"ws://localhost:{mock_server.port}",
        api_key="test-key",
    )
    stream = stt.stream()

    task = asyncio.create_task(_drain_stream(stream, [], []))

    stream.push_frame(audio_chunks[0])
    stream.end_input()

    # Pushing after end_input: livekit-agents 1.5.x raises RuntimeError (input ended).
    with pytest.raises((ChanClosed, RuntimeError)):
        stream.push_frame(audio_chunks[1])

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Module-level smoke test (run with: python test_stt.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
