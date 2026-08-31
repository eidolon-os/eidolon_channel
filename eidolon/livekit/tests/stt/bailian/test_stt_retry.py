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

from livekit.agents import APIError  # noqa: E402
from livekit.agents.types import APIConnectOptions  # noqa: E402

from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT  # noqa: E402


class AbruptCloseServer:
    """Mock server that accepts run-task, sends task-started, then ABRUPTLY
    closes the connection without sending task-finished — simulating
    DashScope server-side close during a long-lived session."""

    def __init__(self) -> None:
        self.port = 0
        self._server: websockets.WebSocketServer | None = None
        self.handler_invocations = 0

    async def start(self) -> None:
        self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
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


class RecoverOnSecondConnectionServer(AbruptCloseServer):
    """Crash one recognition task, then complete the retried task normally."""

    async def _handler(self, ws: ServerConnection) -> None:
        self.handler_invocations += 1
        attempt = self.handler_invocations
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            task_id = msg["header"].get("task_id") or str(uuid.uuid4())[:32]
            await ws.send(
                json.dumps(
                    {
                        "header": {
                            "event": "task-started",
                            "task_id": task_id,
                            "task_status": "RUNNING",
                        },
                        "payload": {},
                    }
                )
            )
            if attempt == 1:
                await asyncio.wait_for(ws.recv(), timeout=1.0)
                await ws.close(code=1011, reason="Injected first-attempt crash")
                return

            await ws.send(
                json.dumps(
                    {
                        "header": {
                            "event": "result-generated",
                            "task_id": task_id,
                            "request_id": "retry-request",
                        },
                        "payload": {
                            "output": {
                                "sentence": {
                                    "sentence_id": 1,
                                    "text": "重连成功",
                                    "text_with_punct": "重连成功。",
                                    "begin_time": 0,
                                    "end_time": 800,
                                    "sentence_end": True,
                                }
                            }
                        },
                    }
                )
            )
            async for item in ws:
                if not isinstance(item, str):
                    continue
                request = json.loads(item)
                if request.get("header", {}).get("action") != "finish-task":
                    continue
                await ws.send(
                    json.dumps(
                        {
                            "header": {
                                "event": "task-finished",
                                "task_id": task_id,
                                "request_id": "retry-request",
                            },
                            "payload": {},
                        }
                    )
                )
                await ws.close()
                return
        except websockets.exceptions.ConnectionClosed:
            return


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


@pytest.mark.asyncio
async def test_framework_retry_recovers_after_provider_disconnect() -> None:
    """The adapter must reconnect and resume emitting public SpeechEvents."""

    server = RecoverOnSecondConnectionServer()
    await server.start()
    try:
        stt = BailianFunASRSTT(
            api_url=f"ws://127.0.0.1:{server.port}",
            api_key="mock-key",
        )
        stream = stt.stream(
            conn_options=APIConnectOptions(
                max_retry=1,
                retry_interval=0.0,
                timeout=5.0,
            )
        )
        from livekit import rtc
        from livekit.agents import stt as lk_stt

        frame = rtc.AudioFrame(
            data=b"\x00\x00" * 1600,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=1600,
        )
        events: list[lk_stt.SpeechEvent] = []

        async def consume() -> None:
            async for event in stream:
                events.append(event)

        consumer = asyncio.create_task(consume())
        stream.push_frame(frame)
        await asyncio.wait_for(
            _wait_until(lambda: server.handler_invocations == 2),
            timeout=3.0,
        )
        stream.push_frame(frame)
        stream.end_input()
        await asyncio.wait_for(consumer, timeout=3.0)

        finals = [
            event
            for event in events
            if event.type is lk_stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert server.handler_invocations == 2
        assert [event.alternatives[0].text for event in finals] == ["重连成功"]
    finally:
        await server.stop()


async def _wait_until(predicate, *, interval_sec: float = 0.01) -> None:
    while not predicate():
        await asyncio.sleep(interval_sec)
