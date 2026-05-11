"""WebSocket client for SenseAudio TTS protocol using aiohttp."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

import aiohttp

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sensetime.tts.client")


class SenseTimeTTSError(Exception):
    """Raised when the SenseAudio TTS connection fails or reports an error."""

    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


class TTSConnection:
    """Manages a single-task WebSocket connection to the SenseAudio TTS API.

    Per SenseAudio protocol, each TTS task has a fixed lifecycle:
        connect → connected_success → task_start → task_started →
        task_continue(×N) → task_finish → task_finished → server closes WS

    Each TTSConnection instance handles exactly one task. After task_finish,
    the server closes the WebSocket — the connection cannot be reused.

    Usage::

        conn = TTSConnection(uri=..., api_key=..., ...)
        await conn.connect()
        await conn.send_task_start()
        await conn.send_task_continue("hello world")
        await conn.send_task_finish()
        # server sends task_finished, then closes WS
        await conn.disconnect()
    """

    CONNECT_TIMEOUT = 15.0

    def __init__(
        self,
        uri: str,
        api_key: str,
        model: str,
        voice_id: str,
        sample_rate: int,
        speed: float,
        vol: float,
        pitch: int,
        audio_format: str = "pcm",
        bitrate: int = 128000,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._uri = uri
        self._api_key = api_key
        self._model = model
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._speed = speed
        self._vol = vol
        self._pitch = pitch
        self._audio_format = audio_format
        self._bitrate = bitrate
        self._http_session = http_session

        # Connection state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._connected_success_received = False
        self._msg_seq = 0
        self._recv_task: asyncio.Task | None = None

        # State machine: "disconnected" | "connecting" | "connected" | "ready" | "task_active" | "finishing"
        self._state: str = "disconnected"
        self._session_id: str | None = None
        # Event signaled when server sends task_started (ack for task_start).
        self._task_started_event = asyncio.Event()
        # Event signaled when server sends task_finished.
        self._task_finished_event = asyncio.Event()
        # Tracks whether the current task was initiated by us (vs inherited from warmup).
        # Used to determine whether we should send task_finish when the task ends.
        self._stream_initiated_task = False

        # Heartbeat task for persistent connection keep-alive
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and self._ws is not None
            and not self._ws.closed
            and self._state in ("ready", "task_active")
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            from livekit.agents import utils

            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def connect(self) -> bool:
        """Open WebSocket, wait for connected_success, return True on success.

        After connecting, call send_task_start() to initialize the TTS session.
        """
        return await self._do_connect()

    async def _do_connect(self) -> bool:
        """Internal connect: opens WebSocket and waits for connected_success."""
        if not self._api_key:
            logger.warning("[TTSConnection] No api_key provided")
            return False

        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            session = self._ensure_session()
            logger.info("[TTSConnection] Connecting to %s", self._uri)
            self._ws = await asyncio.wait_for(
                session.ws_connect(self._uri, headers=headers),
                timeout=self.CONNECT_TIMEOUT,
            )
            logger.info("[TTSConnection] WebSocket opened, starting receive loop")

            self._recv_task = asyncio.create_task(self._receive_loop())

        except asyncio.TimeoutError:
            logger.error(
                "[TTSConnection] Connection timeout after %ss", self.CONNECT_TIMEOUT
            )
            return False
        except Exception as e:
            logger.error("[TTSConnection] Connection failed: %s", e)
            return False

        self._connected = True
        self._state = "connected"

        # Wait for connected_success event
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("[TTSConnection] No connected_success within 8s")
            await self._force_close()
            return False

        self._state = "ready"
        self._task_finished_event.clear()
        logger.info("[TTSConnection] Connected successfully")
        return True

    async def ensure_connected(self) -> bool:
        """Ensure the connection is in READY state.

        - If already READY or TASK_ACTIVE with a live socket: return immediately (True).
        - If CONNECTING: wait for it to complete.
        - If DISCONNECTED or socket is dead: reconnect and re-send task_start.

        Returns True if the connection is ready for use, False on failure.
        """
        # Fast path: state is active and socket is alive.
        if self._state in ("ready", "task_active"):
            if self._ws is None or self._ws.closed:
                self._state = "disconnected"
                self._connected = False
            else:
                return True

        if self._state == "connecting":
            await asyncio.wait_for(self._connected_event.wait(), timeout=8.0)
            return self._state == "ready"

        # FINISHING — wait for server to close, then reconnect
        if self._state == "finishing":
            try:
                await asyncio.wait_for(self._task_finished_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[TTSConnection] timeout waiting for task_finished during ensure_connected")
            await self._force_close()

        # DISCONNECTED — reconnect
        self._state = "connecting"
        ok = await self._do_connect()
        if not ok:
            self._state = "disconnected"
            return False

        logger.info("[TTSConnection] reconnected")
        return True

    async def send_task_start(self) -> None:
        """Send task_start to initialize a TTS session.

        Call this once after connect() to initialize the session. After a
        reconnect, call it again to re-initialize.
        """
        from .protocol import (
            EVENT_TASK_START,
            KEY_AUDIO_SETTING,
            KEY_BITRATE,
            KEY_CHANNEL,
            KEY_EVENT,
            KEY_FORMAT,
            KEY_MODEL,
            KEY_PITCH,
            KEY_SAMPLE_RATE,
            KEY_SPEED,
            KEY_VOICE_ID,
            KEY_VOL,
            KEY_VOICE_SETTING,
        )

        msg = {
            KEY_EVENT: EVENT_TASK_START,
            KEY_MODEL: self._model,
            KEY_VOICE_SETTING: {
                KEY_VOICE_ID: self._voice_id,
                KEY_SPEED: self._speed,
                KEY_VOL: self._vol,
                KEY_PITCH: self._pitch,
            },
            KEY_AUDIO_SETTING: {
                KEY_SAMPLE_RATE: self._sample_rate,
                KEY_BITRATE: self._bitrate,
                KEY_FORMAT: self._audio_format,
                KEY_CHANNEL: 1,
            },
        }
        self._task_started_event.clear()
        await self._send_json(msg)
        self._state = "task_active"
        self._stream_initiated_task = True
        logger.info(
            "[TTSConnection] Sent task_start model=%s voice=%s sample_rate=%s",
            self._model,
            self._voice_id,
            self._sample_rate,
        )
        # Wait for server to acknowledge with task_started
        try:
            await asyncio.wait_for(self._task_started_event.wait(), timeout=5.0)
            logger.info("[TTSConnection] task_started acknowledged by server")
        except asyncio.TimeoutError:
            logger.warning("[TTSConnection] timeout waiting for task_started ack")

    async def send_task_continue(self, text: str) -> None:
        """Send task_continue with text to be synthesized."""
        from .protocol import EVENT_TASK_CONTINUE, KEY_EVENT, KEY_TEXT

        msg = {KEY_EVENT: EVENT_TASK_CONTINUE, KEY_TEXT: text}
        await self._send_json(msg)
        logger.debug("[TTSConnection] Sent task_continue text_len=%s", len(text))

    async def send_task_finish(self) -> None:
        """Send task_finish to signal end of input.

        Per SenseAudio protocol, after task_finish the server sends task_finished
        then closes the WebSocket. We set _state="finishing" so receive_loop and
        ensure_connected know the connection is winding down.
        """
        from .protocol import EVENT_TASK_FINISH, KEY_EVENT

        self._state = "finishing"
        self._stream_initiated_task = False
        self._task_finished_event.clear()
        msg = {KEY_EVENT: EVENT_TASK_FINISH}
        await self._send_json(msg)
        logger.info("[TTSConnection] Sent task_finish, state -> finishing")

    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        await self._force_close()
        logger.debug("[TTSConnection] Disconnected")

    async def _force_close(self) -> None:
        self.stop_heartbeat()  # cancel heartbeat before closing
        async with self._lock:
            self._connected = False
            self._state = "disconnected"
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
                try:
                    await self._recv_task
                except (asyncio.CancelledError, asyncio.InvalidStateError):
                    pass
                self._recv_task = None
            if self._ws:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

    async def _close_without_wait(self) -> None:
        """Fire-and-forget close: called via asyncio.create_task()."""
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass
        self._ws = None

    async def _send_json(self, msg: dict[str, Any]) -> None:
        async with self._lock:
            if self._ws is None or self._ws.closed:
                raise SenseTimeTTSError("Not connected", recoverable=True)
            raw_bytes = json.dumps(msg)
            logger.info(
                "[TTSConnection] SEND raw_bytes_len=%d msg_preview=%s",
                len(raw_bytes),
                raw_bytes[:100],
            )
            try:
                await self._ws.send_str(raw_bytes)
            except (ConnectionResetError, aiohttp.ClientError) as e:
                self._connected = False
                self._state = "disconnected"
                raise SenseTimeTTSError(f"Send failed: {e}", recoverable=True) from e

    async def _receive_loop(self) -> None:
        """Receive JSON messages and dispatch to callback.

        Exits when the server sends a CLOSE frame (expected after task_finished).
        """
        from .protocol import EVENT_CONNECTED_SUCCESS, EVENT_TASK_FINISHED, EVENT_TASK_STARTED

        ws = self._ws
        if ws is None:
            return

        try:
            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.warning("[TTSConnection] WebSocket closed by server")
                    self._connected = False
                    self._state = "disconnected"
                    self._ws = None
                    self._connected_event.clear()
                    # Signal the active stream that connection is gone
                    if self._on_message_callback:
                        result = self._on_message_callback(None)
                        if asyncio.iscoroutine(result):
                            await result
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning(
                        "[TTSConnection] Unexpected message type: %s", msg.type
                    )
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(
                        "[TTSConnection] Failed to decode JSON: %s", msg.data
                    )
                    continue

                self._msg_seq += 1
                evt = data.get("event")
                if evt not in ("task_continue", "task_continued"):
                    logger.info(
                        "[TTSConnection] RAW msg[#%d]: event=%s ts=%.3f data=%s",
                        self._msg_seq,
                        evt,
                        time.time(),
                        data,
                    )
                else:
                    # All fields below are only logged at DEBUG; kept as
                    # commented reference for future diagnostics.
                    pass
                    # data_section = data.get("data") or {}
                    # audio_present = (
                    #     "audio" in data_section and bool(data_section.get("audio"))
                    # )
                    # logger.info(
                    #     "[TTSConnection] RAW msg[#%d]: event=%s ts=%.3f has_audio=%s status=%s",
                    #     self._msg_seq,
                    #     evt,
                    #     time.time(),
                    #     audio_present,
                    #     data_section.get("status"),
                    # )

                if data.get("event") == EVENT_CONNECTED_SUCCESS:
                    self._connected_success_received = True
                    self._session_id = data.get("session_id")
                    self._connected_event.set()
                    logger.info(
                        "[TTSConnection] Received connected_success session_id=%s",
                        self._session_id,
                    )
                    continue

                if evt == EVENT_TASK_STARTED:
                    self._task_started_event.set()
                    logger.info("[TTSConnection] Received task_started")

                if evt == EVENT_TASK_FINISHED:
                    self._task_finished_event.set()

                # Dispatch to the active stream's callback (set by SenseTimeSynthesizeStream)
                if self._on_message_callback:
                    result = self._on_message_callback(data)
                    if asyncio.iscoroutine(result):
                        await result

        except asyncio.CancelledError:
            logger.info("[TTSConnection] Receive loop cancelled")
        finally:
            self._connected = False

    # Callback set by SenseTimeSynthesizeStream to receive messages
    _on_message_callback: Any = None

    # ------------------------------------------------------------------
    # Heartbeat: keep the persistent connection alive during idle periods
    # ------------------------------------------------------------------

    def start_heartbeat(
        self,
        interval: float,
        stream_active_check: Callable[[], bool],
    ) -> None:
        """Start a heartbeat task that sends empty task_continue during idle.

        The heartbeat fires every *interval* seconds, but only when:
        - ``_state == "task_active"`` (task has been started)
        - ``stream_active_check()`` returns ``False`` (no stream is active)

        This prevents the server's 120s inactivity timeout from closing the
        connection between conversation turns.
        """
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return  # Already running
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(interval, stream_active_check)
        )

    def stop_heartbeat(self) -> None:
        """Cancel the heartbeat task (synchronous, fire-and-forget cancel)."""
        if self._heartbeat_task is not None:
            if not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(
        self,
        interval: float,
        stream_active_check: Callable[[], bool],
    ) -> None:
        """Internal heartbeat loop: send empty task_continue when idle."""
        try:
            while True:
                await asyncio.sleep(interval)
                if self._state != "task_active":
                    logger.debug(
                        "[TTSConnection] heartbeat skip: state=%s", self._state
                    )
                    continue
                if stream_active_check():
                    logger.debug("[TTSConnection] heartbeat skip: stream active")
                    continue
                try:
                    await self.send_task_continue("")
                    logger.debug("[TTSConnection] heartbeat sent")
                except Exception as e:
                    logger.warning("[TTSConnection] heartbeat failed: %s", e)
        except asyncio.CancelledError:
            logger.debug("[TTSConnection] heartbeat cancelled")


# Backward-compatibility alias
SenseTimeTTSClient = TTSConnection
