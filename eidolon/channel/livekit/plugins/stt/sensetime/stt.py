"""SenseTime SenseAudio STT plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

import aiohttp
import livekit
from livekit.agents import stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType, STTCapabilities
from livekit.agents.types import APIConnectOptions

from .config import SenseTimeSTTConfig
from .connection import SenseTimeSTTError, STTConnection

if TYPE_CHECKING:
    from .speech_stream import SenseTimeSpeechStream

logger = logging.getLogger("sensetime.stt")

DEFAULT_API_KEY = os.environ.get("SENSEAUDIO_API_KEY", "")


class SenseTimeSTT(stt.STT):
    """SenseTime SenseAudio STT provider for LiveKit Agents.

    Holds a single persistent WebSocket connection that is reused across all
    streaming sessions for the entire LiveKit session lifetime. The connection
    is opened by :meth:`warmup` (typically during the pipeline warmup phase),
    which establishes the WebSocket but does **not** send ``task_start`` —
    per SenseAudio STT protocol each utterance is its own task, so the
    SpeechStream sends ``task_start`` at the beginning of each ``_run()``.

    Concurrent stream usage is serialised by ``_stream_lock`` (only one stream
    may use the connection at a time, so message dispatch via
    ``_on_message_callback`` is unambiguous).

    Can be instantiated with either a :class:`SenseTimeSTTConfig` object or
    individual keyword arguments (used when ``config`` is ``None``).
    """

    def __init__(
        self,
        config: SenseTimeSTTConfig | None = None,
        *,
        api_url: str = "wss://api.senseaudio.cn/ws/v1/audio/transcriptions",
        api_key: str = "",
        model: str = "",  # Empty to use server default
        sample_rate: int = 16000,
        language: str = "zh",
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = SenseTimeSTTConfig(
                api_url=api_url,
                api_key=api_key,
                model=model if model else "",
                sample_rate=sample_rate,
                language=language,
            )
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=True,
                offline_recognize=False,
                aligned_transcript=False,
                diarization=False,
            ),
        )
        self._conn_options = conn_options or livekit.agents.types.DEFAULT_API_CONNECT_OPTIONS

        # Persistent connection (single WS shared across utterances)
        self._conn: STTConnection | None = None
        self._conn_lock = asyncio.Lock()  # Protects _conn slot + warmup serialisation

        # Round 8 — aiohttp ClientSession used for ws_connect.
        # In production: lazily acquired from framework's http_context.
        # In tests / standalone: lazily created as own ClientSession.
        # (Mirrors SenseTimeTTS._ensure_http_session pattern.)
        self._http_session: aiohttp.ClientSession | None = None
        self._owns_http_session: bool = False

        # Serialise concurrent SpeechStreams over the shared connection.
        # Only one stream may own _on_message_callback at a time.
        self._stream_lock: asyncio.Lock = asyncio.Lock()

        # State-sync bridge (Round 7 G11): orchestrator can signal that the
        # framework's high-level user_state has gone "away", meaning the
        # in-flight stream is operating on noise/echo (not real user speech)
        # and should abort immediately rather than waiting for the 30s safety
        # net. Active SpeechStreams watch this event in their _run() loop.
        #
        # Why an asyncio.Event (not a method):
        #   - Multiple concurrent streams (rare, but possible during
        #     transitions) all need to react to the same signal
        #   - asyncio.Event supports clean cross-task notification
        #   - Idempotent: setting twice is harmless; clear() resets
        self._user_away_event: asyncio.Event = asyncio.Event()

        if not self._config.api_key:
            logger.warning(
                "No SENSEAUDIO_API_KEY provided to SenseTimeSTT. "
                "Set the SENSEAUDIO_API_KEY environment variable or pass api_key explicitly."
            )

        logger.info(
            "[SenseTimeSTT] initialized provider=sensetime model=%s "
            "sample_rate=%s api_url=%s",
            self._config.model,
            self._config.sample_rate,
            self._config.api_url,
        )

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def provider(self) -> str:
        return "sensetime"

    @property
    def label(self) -> str:
        return f"SenseTime STT ({self._config.model})"

    @property
    def api_key(self) -> str:
        return self._config.api_key

    @property
    def api_url(self) -> str:
        return self._config.api_url

    @property
    def language(self) -> str:
        return self._config.language

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    @property
    def conn_options(self) -> APIConnectOptions:
        return self._conn_options

    async def warmup(self) -> None:
        """Open the persistent WebSocket. Idempotent.

        Does NOT send ``task_start`` — SenseAudio STT uses per-utterance tasks,
        so each :class:`SenseTimeSpeechStream._run()` sends its own
        ``task_start`` at the beginning of the utterance and ``task_finish``
        when VAD signals end-of-speech.

        WS-level ping/pong (configured inside ``STTConnection._do_connect``)
        keeps the socket alive between utterances; no application-level
        heartbeat is needed (unlike TTS, which uses empty ``task_continue``).
        """
        async with self._conn_lock:
            if self._conn is not None and self._conn.is_connected:
                logger.info("[SenseTimeSTT] already warmed up, skipping")
                return

            # Tear down any stale connection before re-warmup
            if self._conn is not None:
                await self._conn.disconnect()
                self._conn = None

            logger.info("[SenseTimeSTT] warming up STT connection...")
            conn = STTConnection(
                uri=self._config.api_url,
                api_key=self._config.api_key,
                model=self._config.model,
                sample_rate=self._config.sample_rate,
                language=self._config.language,
                # Round 8 R8.12.a: thread server-side VAD config so
                # operators can tune segmentation aggressiveness.
                silence_duration_ms=self._config.silence_duration_ms,
                min_speech_duration_ms=self._config.min_speech_duration_ms,
                # Round 8 — share aiohttp ClientSession (matches TTS).
                http_session=self._ensure_http_session(),
            )
            connected = await conn.connect()
            if not connected:
                raise SenseTimeSTTError(
                    f"Failed to warm up STT connection at {self._config.api_url}"
                )
            self._conn = conn
            logger.info(
                "[SenseTimeSTT] warmup complete, persistent connection ready "
                "model=%s sample_rate=%d",
                self._config.model,
                self._config.sample_rate,
            )

    async def _ensure_conn(self) -> STTConnection:
        """Return a usable connection, re-warming if it died.

        Called by :class:`SenseTimeSpeechStream._run` inside the
        ``_stream_lock`` so reconnects are serialised with stream usage.
        """
        if self._conn is None or not self._conn.is_connected:
            logger.warning(
                "[SenseTimeSTT] connection dead or missing — re-warmup..."
            )
            await self.warmup()
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # State-sync bridge methods (Round 7 G11)
    #
    # Called by the orchestrator (StreamingPipeline) when the LiveKit
    # framework's high-level user_state changes. These signal in-flight
    # SpeechStreams to react to authoritative state changes, instead of
    # relying on the per-stream 30s safety net.
    # ------------------------------------------------------------------

    def signal_user_away(self) -> None:
        """Notify active streams that user_state -> "away".

        The framework has determined the user is no longer present; any
        audio currently being received is noise/echo and the stream should
        abort immediately. SpeechStream._run() awaits this event in
        parallel with its main exit_event and breaks out cleanly when set.

        Idempotent. Safe to call when no stream is active.
        """
        if not self._user_away_event.is_set():
            logger.info(
                "[SenseTimeSTT] state-sync: user_state -> away, "
                "signaling active streams to abort"
            )
        self._user_away_event.set()

    def signal_user_present(self) -> None:
        """Notify the plugin that user_state has returned from "away".

        Resets the user_away signal so subsequent streams won't be
        prematurely aborted. Called by orchestrator on
        user_state: away -> listening / speaking transitions.

        Idempotent.
        """
        if self._user_away_event.is_set():
            logger.info(
                "[SenseTimeSTT] state-sync: user_state returned from away"
            )
        self._user_away_event.clear()

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> "SenseTimeSpeechStream":
        """Create a streaming transcription session."""
        from .speech_stream import SenseTimeSpeechStream

        return SenseTimeSpeechStream(
            stt=self,
            conn_options=conn_options or self._conn_options,
            sample_rate=self._config.sample_rate,
            language=language or self._config.language,
        )

    async def shutdown(self) -> None:
        """Close the persistent WebSocket connection."""
        async with self._conn_lock:
            if self._conn is not None:
                logger.info("[SenseTimeSTT] shutting down persistent connection")
                await self._conn.disconnect()
                self._conn = None
        # Close own aiohttp session (only if WE created it; framework-provided
        # sessions are managed by framework lifecycle).
        if self._http_session is not None and self._owns_http_session:
            try:
                await self._http_session.close()
            except Exception:
                logger.warning("[SenseTimeSTT] error closing HTTP session")
            self._http_session = None

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Lazy-acquire aiohttp ClientSession.

        Two paths:
        * Inside livekit-agents worker: borrow framework's
          ``utils.http_context.http_session()`` so all our outbound
          HTTP/WS shares the same connection pool / proxy / SSL config
          as the rest of the agent.
        * Outside worker (tests, standalone): create our own session
          and remember to close it at shutdown.
        """
        if self._http_session is not None:
            return self._http_session
        try:
            from livekit.agents import utils
            self._http_session = utils.http_context.http_session()
            self._owns_http_session = False
        except Exception:
            # Outside framework worker context — make our own.
            import aiohttp as _aiohttp
            self._http_session = _aiohttp.ClientSession()
            self._owns_http_session = True
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: "list",
        *,
        language: "Any",
        conn_options: "APIConnectOptions",
    ) -> SpeechEvent:
        """Batch recognition is not supported for streaming STT."""
        raise NotImplementedError("Batch recognition is not supported. Use stream() instead.")
