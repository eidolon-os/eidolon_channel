"""WebSocket client for SenseAudio STT protocol.

Per SenseAudio STT protocol (https://senseaudio.cn/docs/speech_recognition/websocket):

    connect → connected_success → task_start → task_started →
        [binary audio frames + result_final responses] →
    task_finish → task_finished → (WS stays open for next task)

KEY DIFFERENCES from SenseAudio TTS:
- Audio is sent as **raw binary WebSocket frames**, NOT JSON-wrapped hex.
- Server emits ``result_final`` (not ``task_continued``) for transcription.
- No application-level heartbeat — relies on WS-level ping/pong (configured
  via ``heartbeat`` in ``aiohttp.ws_connect``).
- task_start is **per-utterance** (sent at stream start), not per-session.

This connection is shared (one WS per ``SenseTimeSTT`` instance) across many
sequential utterances; the upper layer serialises stream usage via a
``_stream_lock`` so message dispatch is unambiguous.

============================================================================
ARCHITECTURE NOTE — uses ``aiohttp`` not ``websockets``
============================================================================

This module previously used the ``websockets`` library; migrated to
``aiohttp`` in Round 8 to match the TTS plugin and the framework's
own HTTP/WebSocket stack. Reasons:

  1. Network stack consistency — livekit-agents internally uses
     ``aiohttp.ClientSession`` for all I/O. STT, TTS, and framework
     now share one connection pool / proxy logic / SSL config.
  2. Proxy semantics — ``websockets`` reads ``ALL_PROXY`` env and
     requires ``python-socks`` for SOCKS support, which surprised
     ops in production. ``aiohttp`` ignores SOCKS env by default.
  3. Dependency reduction — removes ``websockets`` from runtime deps.

We don't use any ``websockets``-specific feature (extensions,
custom subprotocols, advanced handshakes), so the migration is
behaviour-preserving.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    pass

logger = logging.getLogger("sensetime.stt.client")


class SenseTimeSTTError(Exception):
    """Raised when the SenseAudio STT connection fails or reports an error."""

    def __init__(self, message: str, recoverable: bool = False):
        super().__init__(message)
        self.recoverable = recoverable


def _ws_is_open(ws: Any) -> bool:
    """Return True iff the aiohttp ClientWebSocketResponse is currently open."""
    if ws is None:
        return False
    return not ws.closed


class STTConnection:
    """Manages a single persistent WebSocket connection to the SenseAudio STT API.

    State machine:
        disconnected → connecting → connected → ready
                                                    ↓ (per-utterance)
                                                task_active → finishing → ready
                                                                            ↓ (next)
                                                                        task_active

    Notes:
    - State transitions are protected by ``_lock``. Reconnects are serialised
      via ``_reconnect_lock`` so two concurrent ``ensure_connected`` calls
      don't both try to open a new socket.
    - WS-level ping/pong is the only keep-alive (see ``ping_interval`` /
      ``ping_timeout`` in ``_do_connect``). No application-level heartbeat.
    - On unexpected close, the receive loop transitions to ``disconnected``
      and signals the active stream via a ``None`` sentinel through
      ``_on_message_callback``. The next ``ensure_connected`` will reconnect.
    """

    CONNECT_TIMEOUT = 15.0

    def __init__(
        self,
        uri: str,
        api_key: str,
        model: str,
        sample_rate: int = 16000,
        language: str = "zh",
        # Round 8 R8.12.a: server-side VAD segmentation tuning. SenseAudio
        # task_start accepts a ``vad_setting`` block; we now send it
        # explicitly so the operator can override server defaults
        # (which observed too-eager 500 ms silence segmentation that
        # split natural Chinese pauses).
        silence_duration_ms: int = 800,
        min_speech_duration_ms: int = 300,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._uri = uri
        self._api_key = api_key
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._silence_duration_ms = silence_duration_ms
        self._min_speech_duration_ms = min_speech_duration_ms
        # aiohttp ClientSession used for ws_connect. If None, we'll lazily
        # acquire from the framework's shared http_context (matches TTS
        # plugin's pattern). When we create our own (test/standalone),
        # ``_owns_http_session`` is True so disconnect() closes it.
        self._http_session = http_session
        self._owns_http_session: bool = False

        # Connection state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._task_started_event = asyncio.Event()
        self._task_finished_event = asyncio.Event()
        # Set when the server sends task_failed in response to task_start.
        # Allows send_task_start to fail fast instead of waiting 5 s for an
        # ack that will never arrive.
        self._task_failed_event = asyncio.Event()
        self._connected_success_received = False
        self._msg_seq = 0
        self._recv_task: asyncio.Task | None = None

        # State machine: "disconnected" | "connecting" | "connected" | "ready"
        #                "task_active" | "finishing"
        self._state: str = "disconnected"
        self._session_id: str | None = None

        # Callback set by SenseTimeSpeechStream to receive non-handshake messages.
        self._on_message_callback: Any = None

    @property
    def is_connected(self) -> bool:
        """True iff the WS is open and we are in a usable state."""
        if not self._connected:
            return False
        if not _ws_is_open(self._ws):
            return False
        return self._state in ("connected", "ready", "task_active")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Open the WebSocket and wait for ``connected_success``.

        Does NOT send ``task_start`` automatically — STT uses per-utterance
        tasks. Call :meth:`send_task_start` from the SpeechStream at the
        beginning of each utterance.
        """
        return await self._do_connect()

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazy-acquire aiohttp ClientSession.

        Two paths (matches ``SenseTimeSTT._ensure_http_session``):
        * Inside livekit-agents worker → borrow framework's shared
          ``utils.http_context.http_session()``.
        * Outside worker (tests / standalone) → create own ClientSession.
          Caller is responsible for closing via ``disconnect`` (we don't
          own the session lifecycle when one was passed in via ``__init__``).
        """
        if self._http_session is not None:
            return self._http_session
        try:
            from livekit.agents import utils
            self._http_session = utils.http_context.http_session()
        except Exception:
            # Outside worker context — create our own.
            self._http_session = aiohttp.ClientSession()
            self._owns_http_session = True
        return self._http_session

    async def _do_connect(self) -> bool:
        if not self._api_key:
            logger.warning("[STTConnection] No api_key provided")
            return False

        headers = {"Authorization": f"Bearer {self._api_key}"}

        # Reset events so a stale signal from a prior connection can't fool us.
        self._connected_event.clear()
        self._task_started_event.clear()
        self._task_finished_event.clear()
        self._task_failed_event.clear()

        try:
            session = self._ensure_session()
            logger.info("[STTConnection] Connecting to %s", self._uri)
            self._ws = await asyncio.wait_for(
                session.ws_connect(
                    self._uri,
                    headers=headers,
                    max_msg_size=0,  # 0 = unlimited (was max_size=None)
                    # aiohttp WS-level ping. autoping=True (default) sends
                    # pings every ``heartbeat`` seconds and tracks pongs;
                    # closing the WS if no pong arrives within the heartbeat
                    # interval. Matches the previous ping_interval=20s behavior.
                    heartbeat=20.0,
                ),
                timeout=self.CONNECT_TIMEOUT,
            )
            logger.info("[STTConnection] WebSocket opened, starting receive loop")
            self._recv_task = asyncio.create_task(self._receive_loop())
        except asyncio.TimeoutError:
            logger.error(
                "[STTConnection] Connection timeout after %ss", self.CONNECT_TIMEOUT
            )
            return False
        except (aiohttp.ClientError, OSError) as e:
            logger.error("[STTConnection] Connection failed: %s", e)
            return False
        except Exception as e:
            logger.error("[STTConnection] Connection failed (unexpected): %s", e)
            return False

        self._connected = True
        self._state = "connected"

        # Wait for server's connected_success
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("[STTConnection] No connected_success within 8s")
            await self._force_close()
            return False

        self._state = "ready"
        logger.info("[STTConnection] Connected successfully")
        return True

    async def ensure_connected(self) -> bool:
        """Ensure the WS is open and in a usable state, reconnecting if needed.

        Returns True on success, False on permanent failure. Multiple concurrent
        callers are serialised via ``_reconnect_lock`` to avoid duplicate
        reconnect attempts.
        """
        # Fast path: already usable.
        if self.is_connected:
            return True

        # Stale state: socket died but state wasn't updated yet.
        if self._ws is not None and not _ws_is_open(self._ws):
            self._state = "disconnected"
            self._connected = False

        if self._state == "connecting":
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=8.0)
            except asyncio.TimeoutError:
                pass
            return self.is_connected

        async with self._reconnect_lock:
            # Re-check after acquiring; another task may have reconnected.
            if self.is_connected:
                return True
            self._state = "connecting"
            ok = await self._do_connect()
            if not ok:
                self._state = "disconnected"
                return False
            logger.info("[STTConnection] reconnected")
            return True

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket connection."""
        await self._force_close()
        # If we created our own ClientSession (tests / standalone), close it.
        if self._owns_http_session and self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None
            self._owns_http_session = False
        logger.debug("[STTConnection] Disconnected")

    async def _force_close(self) -> None:
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
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
                self._ws = None

    # ------------------------------------------------------------------
    # Protocol messages
    # ------------------------------------------------------------------

    async def send_task_start(self) -> None:
        """Send ``task_start`` and wait for the server's ``task_started`` ack.

        This sets state to ``task_active``. Per STT protocol, ``task_start`` is
        sent at the **beginning of each utterance** (not once per session like
        TTS), so the SpeechStream calls this in ``_run()``, not in warmup.
        """
        from .protocol import (
            EVENT_TASK_START,
            KEY_AUDIO_SETTING,
            KEY_CHANNEL,
            KEY_EVENT,
            KEY_FORMAT,
            KEY_MODEL,
            KEY_SAMPLE_RATE,
            KEY_VAD_SETTING,
        )

        msg: dict[str, Any] = {
            KEY_EVENT: EVENT_TASK_START,
            KEY_AUDIO_SETTING: {
                KEY_SAMPLE_RATE: self._sample_rate,
                KEY_FORMAT: "pcm",
                KEY_CHANNEL: 1,
            },
            # Round 8 R8.12.a: explicit server-side VAD segmentation
            # config. Without this, server uses internal defaults (~500 ms
            # silence threshold) which split natural Chinese pauses.
            KEY_VAD_SETTING: {
                "silence_duration": self._silence_duration_ms,
                "min_speech_duration": self._min_speech_duration_ms,
            },
        }
        if self._model:
            msg[KEY_MODEL] = self._model

        self._task_started_event.clear()
        self._task_finished_event.clear()
        self._task_failed_event.clear()
        await self._send_json(msg)
        self._state = "task_active"
        logger.info(
            "[STTConnection] Sent task_start model=%s sample_rate=%s "
            "vad_silence=%dms vad_min_speech=%dms",
            self._model,
            self._sample_rate,
            self._silence_duration_ms,
            self._min_speech_duration_ms,
        )

        # Race: task_started ack OR task_failed rejection OR 5 s timeout.
        # Failing fast on task_failed avoids a pointless 5 s stall when the
        # server immediately rejects the start (e.g. invalid model).
        started_t = asyncio.create_task(self._task_started_event.wait())
        failed_t = asyncio.create_task(self._task_failed_event.wait())
        try:
            done, pending = await asyncio.wait(
                {started_t, failed_t},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5.0,
            )
            for p in pending:
                p.cancel()
        finally:
            for t in (started_t, failed_t):
                if not t.done():
                    t.cancel()

        if self._task_failed_event.is_set():
            self._state = "disconnected"
            raise SenseTimeSTTError(
                "task_start was rejected by server (task_failed)",
                recoverable=False,
            )
        if not self._task_started_event.is_set():
            logger.warning("[STTConnection] timeout waiting for task_started ack")
        else:
            logger.info("[STTConnection] task_started acknowledged by server")

    async def send_audio_binary(self, audio: bytes) -> None:
        """Send PCM audio as a raw WebSocket BINARY frame.

        Per SenseAudio STT protocol, audio is transmitted as binary WS frames
        (NOT as JSON-wrapped hex like TTS).
        """
        async with self._lock:
            if self._ws is None or self._ws.closed:
                raise SenseTimeSTTError("Not connected", recoverable=True)
            try:
                await self._ws.send_bytes(audio)
            except (aiohttp.ClientError, ConnectionResetError) as e:
                self._connected = False
                self._state = "disconnected"
                raise SenseTimeSTTError(
                    f"Audio send failed: {e}", recoverable=True
                ) from e
        # logger.debug("[STTConnection] Sent audio binary bytes=%d", len(audio))

    async def send_task_finish(self) -> None:
        """Send ``task_finish`` to signal end of utterance audio.

        Per protocol, the server then emits any remaining ``result_final``
        events, then ``task_finished``, and **keeps the WS open** for the
        next utterance.
        """
        from .protocol import EVENT_TASK_FINISH, KEY_EVENT

        self._state = "finishing"
        self._task_finished_event.clear()
        await self._send_json({KEY_EVENT: EVENT_TASK_FINISH})
        logger.info("[STTConnection] Sent task_finish, state -> finishing")

    async def _send_json(self, msg: dict[str, Any]) -> None:
        async with self._lock:
            if self._ws is None or self._ws.closed:
                raise SenseTimeSTTError("Not connected", recoverable=True)
            raw = json.dumps(msg)
            logger.info(
                "[STTConnection] SEND raw_bytes_len=%d msg_preview=%s",
                len(raw),
                raw[:100],
            )
            try:
                await self._ws.send_str(raw)
            except (aiohttp.ClientError, ConnectionResetError) as e:
                self._connected = False
                self._state = "disconnected"
                raise SenseTimeSTTError(
                    f"Send failed: {e}", recoverable=True
                ) from e

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Receive JSON messages from the server.

        Handshake events (``connected_success`` / ``task_started`` /
        ``task_finished``) are dispatched to internal events. All events are
        also forwarded to ``_on_message_callback`` (set by the active
        SpeechStream) so the stream can process ``result_final`` etc.

        On unexpected close, signals the callback with a ``None`` sentinel
        and transitions state to ``disconnected``.
        """
        from .protocol import (
            EVENT_CONNECTED_SUCCESS,
            EVENT_TASK_FAILED,
            EVENT_TASK_FINISHED,
            EVENT_TASK_STARTED,
        )

        ws = self._ws
        if ws is None:
            return

        try:
            async for msg in ws:
                # aiohttp delivers WSMessage objects with .type and .data.
                # SenseAudio STT only sends TEXT frames server→client; accept
                # BINARY defensively in case of a future protocol change.
                if msg.type == aiohttp.WSMsgType.TEXT:
                    raw_str = msg.data
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    try:
                        raw_str = msg.data.decode("utf-8")
                    except Exception:
                        logger.warning("[STTConnection] non-utf8 binary message")
                        continue
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.warning(
                        "[STTConnection] WebSocket closed by server "
                        "(WSMsgType=%s)",
                        msg.type,
                    )
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "[STTConnection] WebSocket error: %s",
                        ws.exception(),
                    )
                    break
                else:
                    logger.warning(
                        "[STTConnection] Unexpected WSMsgType: %s",
                        msg.type,
                    )
                    continue

                try:
                    data = json.loads(raw_str)
                except json.JSONDecodeError:
                    logger.warning("[STTConnection] Failed to decode JSON: %s", raw_str)
                    continue

                self._msg_seq += 1
                evt = data.get("event")
                logger.info(
                    "[STTConnection] RAW msg[#%d]: event=%s ts=%.3f data=%s",
                    self._msg_seq,
                    evt,
                    time.time(),
                    data,
                )

                # Handshake bookkeeping
                if evt == EVENT_CONNECTED_SUCCESS:
                    self._connected_success_received = True
                    self._session_id = data.get("session_id")
                    self._connected_event.set()
                    logger.info(
                        "[STTConnection] Received connected_success session_id=%s",
                        self._session_id,
                    )
                    # connected_success is purely a handshake; don't dispatch
                    # to stream callback.
                    continue

                if evt == EVENT_TASK_STARTED:
                    self._task_started_event.set()
                    logger.info("[STTConnection] Received task_started")
                    # fall through — no callback dispatch needed but harmless

                if evt == EVENT_TASK_FINISHED:
                    self._task_finished_event.set()
                    # State: finishing → ready. The WS stays open per STT
                    # protocol so the next utterance's task_start can reuse it.
                    if self._state == "finishing":
                        self._state = "ready"
                    # fall through so the active stream can react.

                if evt == EVENT_TASK_FAILED:
                    # Wake send_task_start if it's waiting for ack.
                    self._task_failed_event.set()
                    # Defensive: if the failure happened mid-task, mark the
                    # state machine as "ready" so a future stream's task_start
                    # is still possible (the WS may stay open).
                    if self._state in ("task_active", "finishing"):
                        self._state = "ready"
                    # fall through so the active stream can record the error.

                # Forward to active stream's callback (result_final / task_failed
                # / task_finished / etc).
                if self._on_message_callback:
                    try:
                        result = self._on_message_callback(data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("[STTConnection] callback raised")

        except asyncio.CancelledError:
            logger.info("[STTConnection] Receive loop cancelled")
            raise
        except Exception:
            logger.exception("[STTConnection] receive loop unexpected error")
        finally:
            # Mark dead and wake the active stream so it doesn't hang.
            self._connected = False
            self._state = "disconnected"
            self._connected_event.clear()
            if self._on_message_callback:
                try:
                    result = self._on_message_callback(None)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    pass
