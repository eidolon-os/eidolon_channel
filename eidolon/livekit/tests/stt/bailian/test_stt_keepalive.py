"""Bailian STT connection keepalive.

DashScope FunASR kills an idle recognition task after ~23s of no audio
("request timeout after 23 seconds"). Between turns — notably during a long
agent reply when a half-duplex device closes its mic — no user audio flows, so
the freshly opened task would hit that timeout and churn a reconnect.

The send loop must push periodic silence during idle gaps to keep the task
alive. This is independent of the VAD billing gate (which has its own keepalive).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import pytest
import websockets
from websockets import ServerConnection

_root = Path(__file__).resolve().parents[6]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

from livekit.agents.types import APIConnectOptions  # noqa: E402

from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT  # noqa: E402


class _IdleCountingServer:
    """Accepts run-task → task-started, then stays open and counts the binary
    (audio) frames it receives — without the client sending any real audio."""

    def __init__(self) -> None:
        self.port = 0
        self._server: websockets.WebSocketServer | None = None
        self.binary_frames = 0

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            task_id = msg["header"].get("task_id") or str(uuid.uuid4())[:32]
            await ws.send(json.dumps({
                "header": {
                    "event": "task-started",
                    "task_id": task_id,
                    "task_status": "RUNNING",
                },
                "payload": {},
            }))
            async for frame in ws:
                if isinstance(frame, (bytes, bytearray)):
                    self.binary_frames += 1
        except Exception:
            pass


@pytest.fixture
async def idle_server() -> "_IdleCountingServer":
    srv = _IdleCountingServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_send_loop_keepalives_during_idle(idle_server, monkeypatch) -> None:
    """With no audio flowing, the send loop pushes silence keepalive frames so
    the DashScope task stays alive past its idle timeout."""
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "0.2")

    stt = BailianFunASRSTT(
        api_url=f"ws://127.0.0.1:{idle_server.port}",
        api_key="mock-key",
        conn_options=APIConnectOptions(max_retry=0, timeout=5.0),
    )
    assert stt.keepalive_interval_sec == pytest.approx(0.2)

    stream = stt.stream()
    # Let the stream connect (run-task → task-started) and sit idle: no audio
    # pushed, input not ended. ~1s ≈ 5 keepalive intervals.
    await asyncio.sleep(1.0)
    await stream.aclose()

    # Several silence keepalives should have reached the server despite zero
    # real audio and no end-of-input.
    assert idle_server.binary_frames >= 2, (
        f"expected idle keepalives, got {idle_server.binary_frames}"
    )


@pytest.mark.asyncio
async def test_keepalive_disabled_sends_nothing_when_idle(
    idle_server, monkeypatch
) -> None:
    """keepalive_interval_sec<=0 disables it — no silence is sent during idle."""
    monkeypatch.setenv("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "0")

    stt = BailianFunASRSTT(
        api_url=f"ws://127.0.0.1:{idle_server.port}",
        api_key="mock-key",
        conn_options=APIConnectOptions(max_retry=0, timeout=5.0),
    )
    assert stt.keepalive_interval_sec == 0

    stream = stt.stream()
    await asyncio.sleep(0.6)
    await stream.aclose()

    assert idle_server.binary_frames == 0
