"""WebSocket client for Bailian CosyVoice TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

import aiohttp

from . import protocol

logger = logging.getLogger("bailian.tts.client")


class BailianTTSError(Exception):
    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


class BailianTTSClient:
    CONNECT_TIMEOUT = 15.0

    def __init__(
        self,
        *,
        uri: str,
        api_key: str,
        model: str,
        voice: str,
        audio_format: str,
        sample_rate: int,
        rate: float = 1.0,
        volume: int = 50,
        pitch: float = 1.0,
        task_started_timeout: float = 12.0,
        task_finished_timeout: float = 20.0,
        http_session: aiohttp.ClientSession | None = None,
        ws_heartbeat_sec: float = 15.0,
    ) -> None:
        self._uri = uri
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._audio_format = audio_format
        self._sample_rate = sample_rate
        self._rate = rate
        self._volume = volume
        self._pitch = pitch
        self._task_started_timeout = task_started_timeout
        self._task_finished_timeout = task_finished_timeout
        self._http_session = http_session
        # G10 (2026-05-17): aiohttp ws heartbeat parameter. 0 / negative disables.
        self._ws_heartbeat_sec = ws_heartbeat_sec if ws_heartbeat_sec > 0 else None

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._lock = asyncio.Lock()
        self._connected = False
        self._task_id: str | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._task_started_event = asyncio.Event()
        self._task_finished_event = asyncio.Event()

    _on_message_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None
    _on_binary_callback: Callable[[bytes], Awaitable[None] | None] | None = None
    _on_closed_callback: Callable[[], Awaitable[None] | None] | None = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            from livekit.agents import utils

            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def connect(self) -> bool:
        if not self._api_key:
            logger.warning("[BailianTTSClient] empty api key")
            return False
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            session = self._ensure_session()
            # G10 (2026-05-17): heartbeat = WS-level PING/PONG keepalive.
            # aiohttp source (client_ws.py:104-167) confirms: PING every
            # heartbeat seconds; PONG must arrive within heartbeat/2 or
            # _pong_not_received → ws.closed=True → next send raises.
            # This is the actual root-cause fix for round-2 silent-death.
            self._ws = await asyncio.wait_for(
                session.ws_connect(
                    self._uri,
                    headers=headers,
                    heartbeat=self._ws_heartbeat_sec,
                ),
                timeout=self.CONNECT_TIMEOUT,
            )
        except Exception as e:
            logger.error("[BailianTTSClient] connect failed: %s", e)
            return False

        self._connected = True
        self._recv_task = asyncio.create_task(self._receive_loop())
        return True

    async def start_task(self) -> None:
        if not self._connected:
            raise BailianTTSError("Not connected", recoverable=True)
        self._task_id = uuid.uuid4().hex
        self._task_started_event.clear()
        payload = protocol.build_run_task_payload(
            task_id=self._task_id,
            model=self._model,
            voice=self._voice,
            audio_format=self._audio_format,
            sample_rate=self._sample_rate,
            rate=self._rate,
            volume=self._volume,
            pitch=self._pitch,
        )
        await self._send_json(payload)
        try:
            await asyncio.wait_for(
                self._task_started_event.wait(), timeout=self._task_started_timeout
            )
        except asyncio.TimeoutError as e:
            raise BailianTTSError("task-started timeout", recoverable=True) from e

    async def send_continue(self, text: str) -> None:
        task_id = self._task_id
        if not task_id:
            raise BailianTTSError("task not started", recoverable=True)
        payload = protocol.build_continue_task_payload(task_id=task_id, text=text)
        await self._send_json(payload)

    async def send_finish(self) -> None:
        task_id = self._task_id
        if not task_id:
            return
        self._task_finished_event.clear()
        payload = protocol.build_finish_task_payload(task_id=task_id)
        await self._send_json(payload)

    async def wait_task_finished(self, timeout_s: float | None = None) -> None:
        timeout = self._task_finished_timeout if timeout_s is None else timeout_s
        try:
            await asyncio.wait_for(self._task_finished_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise BailianTTSError("task-finished timeout", recoverable=True) from e

    async def disconnect(self) -> None:
        await self._force_close()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            ws = self._ws
            if ws is None or ws.closed:
                raise BailianTTSError("WebSocket disconnected", recoverable=True)
            try:
                await ws.send_str(json.dumps(payload))
            except Exception as e:
                self._connected = False
                raise BailianTTSError(f"send failed: {e}", recoverable=True) from e

    async def _force_close(self) -> None:
        self._connected = False
        recv_task = self._recv_task
        if recv_task and not recv_task.done():
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass
        self._recv_task = None
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        self._ws = None
        self._task_id = None

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if self._on_binary_callback:
                        r = self._on_binary_callback(msg.data)
                        if asyncio.iscoroutine(r):
                            await r
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("[BailianTTSClient] invalid json")
                    continue

                event = protocol.parse_event(data)
                if event == protocol.EVENT_TASK_STARTED:
                    self._task_started_event.set()
                elif event == protocol.EVENT_TASK_FINISHED:
                    self._task_finished_event.set()
                elif event == protocol.EVENT_TASK_FAILED:
                    self._task_finished_event.set()

                if self._on_message_callback:
                    r = self._on_message_callback(data)
                    if asyncio.iscoroutine(r):
                        await r
        except asyncio.CancelledError:
            raise
        finally:
            self._connected = False
            if self._on_closed_callback:
                r = self._on_closed_callback()
                if asyncio.iscoroutine(r):
                    await r
