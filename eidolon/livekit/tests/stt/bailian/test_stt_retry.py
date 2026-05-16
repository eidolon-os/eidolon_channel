"""G7 regression: Bailian STT must raise APIError on unexpected WS close
so the framework's RecognizeStream._main_task retry loop kicks in.

Pre-G7 behaviour: ``_run`` swallowed ConnectionClosedError, called
``_emit_error`` (which only emits an event), then returned normally.
The framework saw success → never retried → STT dead for the rest of
the session. Observed in production after AEC warmup windows.

Post-G7: ``_run`` re-raises APIError(retryable=True) for any unclean
exit (recv_loop crashed, server closed without TASK_FINISHED).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from pathlib import Path
import os

import pytest
import websockets
from websockets import ServerConnection

_root = Path(__file__).resolve().parents[6]
if str(_root) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        str(_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )

from livekit.agents import APIError
from livekit.agents.types import APIConnectOptions

from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT


class AbruptCloseServer:
    """Mock server that accepts run-task, sends task-started, then ABRUPTLY
    closes the connection without sending task-finished — simulating
    DashScope server-side close during a long-lived session."""

    def __init__(self) -> None:
        self.port = 0
        self._server: websockets.WebSocketServer | None = None
        self.handler_invocations = 0

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "localhost", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, ws: ServerConnection) -> None:
        self.handler_invocations += 1
        try:
            # Read run-task
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            task_id = msg["header"].get("task_id") or str(uuid.uuid4())[:32]
            # Send task-started
            await ws.send(json.dumps({
                "header": {
                    "event": "task-started",
                    "task_id": task_id,
                    "task_status": "RUNNING",
                },
                "payload": {},
            }))
            # Receive a bit of audio to mimic a real session, then close abruptly.
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            # Close WITHOUT sending task-finished — the unclean-exit case.
            await ws.close(code=1011, reason="Simulated server crash")
        except Exception:
            pass


@pytest.fixture
async def abrupt_server() -> "AbruptCloseServer":
    srv = AbruptCloseServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.mark.asyncio
async def test_run_raises_api_error_on_abrupt_close(abrupt_server) -> None:
    """After server closes mid-session without task-finished, _run() must
    raise APIError(retryable=True) — which is what triggers framework's
    automatic retry path."""
    stt = BailianFunASRSTT(
        api_url=f"ws://127.0.0.1:{abrupt_server.port}",
        api_key="mock-key",
        # Disable framework retries so the test measures ONE _run() cycle's
        # behavior (not the retry loop). max_retry=0 means our APIError
        # bubbles out instead of being caught and retried by _main_task.
        conn_options=APIConnectOptions(max_retry=0, timeout=5.0),
    )
    stream = stt.stream()

    # Drive the stream by pushing some audio + ending input.
    from livekit import rtc

    silent = rtc.AudioFrame(
        data=b"\x00\x00" * 800,  # 50ms silence
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=800,
    )

    async def driver() -> None:
        stream.push_frame(silent)
        # Don't end_input — let server's abrupt close drive the failure.

    asyncio.create_task(driver())

    # Consume events until the stream errors out. Track whether an APIError
    # surfaced.
    raised: Exception | None = None
    try:
        async for _ev in stream:
            pass
    except APIError as e:
        raised = e
    except Exception as e:
        raised = e

    assert raised is not None, "stream should have raised on abrupt close"
    # Either our APIError directly or APIConnectionError (which framework
    # wraps it in after exhausting retries — but max_retry=0 here, so we
    # should see APIError directly).
    assert isinstance(raised, APIError), (
        f"expected APIError, got {type(raised).__name__}: {raised}"
    )


# Note: a follow-up test could verify ``handler_invocations >= 2`` after a
# multi-retry cycle (framework's _main_task should call _run() multiple
# times). That requires a more complete stub server that survives multiple
# connect/close cycles cleanly; deferred. The framework retry-on-APIError
# behaviour itself is framework code (stt.py:384-407) and not ours to test.
