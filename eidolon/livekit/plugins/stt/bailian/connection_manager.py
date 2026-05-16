"""WebSocket connection manager for Bailian FunASR DashScope API.

Manages the full lifecycle of a single WebSocket connection:
- Establishes the connection with the 'run-task' handshake
- Sends binary audio frames as WAV-wrapped PCM
- Sends 'finish-task' to signal end of input
- Closes cleanly on finish or error
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Callable

import websockets
from websockets.exceptions import WebSocketException

from .models import (
    FunASREventType,
    FunASRTaskFailed,
    FunASRTaskFinished,
    FunASRTaskStarted,
    parse_funasr_message,
)

logger = logging.getLogger("bailian.funasr")


class BailianConnectionError(Exception):
    """Raised when the Bailian FunASR connection fails or reports an error."""

    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


class BailianTaskFailedError(BailianConnectionError):
    """Raised when the FunASR task reports a failure via the server."""

    def __init__(
        self,
        task_id: str,
        error_message: str,
        error_code: str = "",
    ):
        super().__init__(
            f"FunASR task {task_id!r} failed: {error_message} (code={error_code})",
            recoverable=True,
        )
        self.task_id = task_id
        self.error_message = error_message
        self.error_code = error_code


# ---------------------------------------------------------------------------
# Message builders (FunASR DashScope protocol)
# ---------------------------------------------------------------------------


def _build_run_task_payload(
    model: str,
    task_id: str,
    *,
    sample_rate: int = 16000,
    itn: bool = True,
    language_hints: str | None = None,
) -> dict[str, Any]:
    # F5 fix (2026-05-16): wire ``language_hints`` into the DashScope payload.
    # Previously the parameter was accepted by every caller but **never**
    # serialized into the run-task message, so DashScope FunASR fell back to
    # auto language detection — producing English misrecognitions like "You"
    # and "That" at the start of Chinese utterances. The DashScope FunASR
    # protocol expects ``language_hints`` as a list of language codes
    # (e.g. ``["zh"]`` to force Chinese; ``["zh","en"]`` for code-switching).
    parameters: dict[str, Any] = {
        "sample_rate": sample_rate,
        "itn": str(itn).lower(),
        # G7-A (2026-05-17): protocol-level keepalive flag. Per FunASR docs:
        # "若启用 heartbeat 参数，即使发送静音音频也能保持连接开放".
        # Even though we currently push continuous mic audio (so server sees
        # constant traffic), this flag is the canonical way to signal
        # "long-lived session" intent — important for edge cases like AEC
        # warmup windows (1-3s of skip_stt) and the future VAD-gate mode
        # (G16) where silence audio is sent at 1Hz instead of 20Hz.
        "heartbeat": True,
    }
    if language_hints:
        # Support both ``"zh"`` and ``"zh,en"`` env-style inputs by splitting
        # on commas. Single-language case yields a 1-element list.
        hints = [h.strip() for h in language_hints.split(",") if h.strip()]
        if hints:
            parameters["language_hints"] = hints
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "input": {},
            "parameters": parameters,
        },
    }


def _build_finish_task_payload(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "input": {},
        },
    }


# ---------------------------------------------------------------------------
# BailianConnectionManager
# ---------------------------------------------------------------------------


class BailianConnectionManager:
    """Manages a single WebSocket session to the Bailian FunASR API.

    Usage::

        conn = BailianConnectionManager()
        await conn.connect(task_id="my-task", model="fun-asr-realtime-2026-02-28",
                           api_key="sk-...")
        async for msg in conn.receive_loop():
            ...
        await conn.send_audio(pcm_bytes)
        await conn.finish()
        await conn.close()
    """

    CONNECT_TIMEOUT: float = 15.0
    READ_TIMEOUT: float = 30.0  # per-message read timeout
    CLOSE_CODE_NORMAL: int = 1000

    def __init__(
        self,
        api_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        api_key: str | None = None,
        model: str = "fun-asr-realtime-2026-02-28",
        sample_rate: int = 16000,
        itn: bool = True,
        language_hints: str | None = None,
    ):
        self._api_url = api_url
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._model = model
        self._sample_rate = sample_rate
        self._itn = itn
        self._language_hints = language_hints

        self._ws: websockets.ClientProtocol | None = None
        self._connected: bool = False
        self._closed_by_us: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._task_id: str | None = None
        self._receive_task: asyncio.Task | None = None
        self._user_message_cb: Callable[[dict[str, Any]], None] | None = None
        self._user_binary_cb: Callable[[bytes], None] | None = None
        self._user_closed_cb: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(
        self,
        message_callback: Callable[[dict[str, Any]], None] | None = None,
        binary_callback: Callable[[bytes], None] | None = None,
        closed_callback: Callable[[], None] | None = None,
    ) -> None:
        """Establish WebSocket connection and perform the 'run-task' handshake.

        Sends 'run-task', waits for 'task-started', then returns.
        Raises BailianConnectionError on failure.
        Raises BailianTaskFailedError if the server reports task failure.
        """
        self._user_message_cb = message_callback
        self._user_binary_cb = binary_callback
        self._user_closed_cb = closed_callback

        headers: list[tuple[str, str]] = []
        if self._api_key:
            headers.append(("Authorization", f"Bearer {self._api_key}"))

        async with self._lock:
            if self._connected:
                logger.warning(
                    "BailianConnectionManager.connect() called while already connected"
                )
                return

            try:
                logger.debug("Connecting to %s", self._api_url)
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self._api_url,
                        additional_headers=dict(headers),
                        max_size=None,
                        ping_interval=20,
                        ping_timeout=10,
                    ),
                    timeout=self.CONNECT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise BailianConnectionError(
                    f"Connection timeout after {self.CONNECT_TIMEOUT}s",
                    recoverable=False,
                )
            except WebSocketException as e:
                raise BailianConnectionError(
                    f"WebSocket connection failed: {e}",
                    recoverable=False,
                )

            # Generate task_id before sending (DashScope protocol requires it in header)
            self._task_id = uuid.uuid4().hex[:32]

            # Send run-task payload
            payload = _build_run_task_payload(
                self._model,
                self._task_id,
                sample_rate=self._sample_rate,
                itn=self._itn,
                language_hints=self._language_hints,
            )
            await self._ws.send(json.dumps(payload))
            logger.debug("Sent run-task: %s", payload)

            # Wait for task-started
            started = await self._recv_message()
            if started is None:
                raise BailianConnectionError(
                    "Connection closed during run-task handshake",
                    recoverable=False,
                )

            event_name, parsed = parse_funasr_message(started)

            if event_name == FunASREventType.TASK_FAILED.value:
                failed: FunASRTaskFailed = parsed
                raise BailianTaskFailedError(
                    task_id=failed.task_id,
                    error_message=failed.error_message,
                    error_code=failed.error_code,
                )

            if event_name != FunASREventType.TASK_STARTED.value:
                raise BailianConnectionError(
                    f"Unexpected handshake response: event={event_name}, data={started}",
                    recoverable=False,
                )

            started_evt: FunASRTaskStarted = parsed
            # Use the task_id returned by the server if available, otherwise keep ours
            self._task_id = started_evt.task_id or self._task_id
            self._connected = True
            logger.info("Connected. task_id=%s", self._task_id)

            # Deliver task-started to the user callback
            if self._user_message_cb is not None and started is not None:
                result = self._user_message_cb(started)
                if asyncio.iscoroutine(result):
                    await result

    async def send_audio(self, audio: bytes) -> None:
        """Send raw PCM audio bytes over the WebSocket.

        The caller is responsible for correct sample rate (16kHz) and
        chunking strategy (e.g. 100ms frames).
        """
        async with self._lock:
            if not self._ws or not self._connected:
                raise BailianConnectionError("Not connected", recoverable=True)
            await self._ws.send(audio)

    async def finish(self) -> None:
        """Signal end of audio input by sending 'finish-task'."""
        async with self._lock:
            if not self._connected or not self._task_id:
                return
            payload = _build_finish_task_payload(self._task_id)
            try:
                if self._ws:
                    await self._ws.send(json.dumps(payload))
                    logger.debug("Sent finish-task: task_id=%s", self._task_id)
            except WebSocketException:
                pass

    async def close(self, code: int = CLOSE_CODE_NORMAL) -> None:
        """Close the WebSocket connection."""
        async with self._lock:
            if not self._connected:
                return
            self._closed_by_us = True
            self._connected = False
            if self._ws:
                try:
                    await self._ws.close(code=code)
                except WebSocketException:
                    pass
                self._ws = None
            logger.debug("Connection closed")

    # ------------------------------------------------------------------
    # Receive helpers
    # ------------------------------------------------------------------

    async def receive_loop(
        self,
        message_cb: Callable[[dict[str, Any]], None],
        binary_cb: Callable[[bytes], None] | None = None,
    ) -> None:
        """Run the receive loop until the connection closes.

        Dispatches incoming messages to `message_cb` (for JSON) or `binary_cb`
        (for binary frames, if provided).
        """
        ws = self._ws
        if ws is None:
            return

        try:
            async for raw in ws:
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                        result = message_cb(data)
                        if asyncio.iscoroutine(result):
                            await result
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON text: %s", raw[:200])
                elif isinstance(raw, bytes):
                    if binary_cb:
                        result = binary_cb(raw)
                        if asyncio.iscoroutine(result):
                            await result
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._connected = False
            if self._user_closed_cb:
                result = self._user_closed_cb()
                if asyncio.iscoroutine(result):
                    await result

    async def _recv_message(self) -> dict[str, Any] | None:
        """Receive and parse a single JSON message. Returns None on close."""
        ws = self._ws
        if ws is None:
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=self.READ_TIMEOUT)
        except asyncio.TimeoutError:
            raise BailianConnectionError(
                "Read timeout waiting for server message",
                recoverable=True,
            )
        except websockets.exceptions.ConnectionClosed:
            return None

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Received non-JSON text in _recv_message: %s", raw[:200]
                )
                return None
        return None
