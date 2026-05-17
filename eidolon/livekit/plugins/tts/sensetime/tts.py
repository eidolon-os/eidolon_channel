"""SenseTime SenseAudio TTS plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import io
import logging
import unicodedata
import uuid
import weakref
from typing import TYPE_CHECKING, Any

import aiohttp

import numpy as np

from livekit.agents.tts import (
    AudioEmitter,
    SynthesizeStream,
    TTS,
    TTSCapabilities,
)
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils.audio import AudioByteStream

from .config import SenseTimeTTSConfig
from .protocol import (
    EVENT_TASK_CONTINUE,
    EVENT_TASK_CONTINUED,
    EVENT_TASK_FAILED,
    EVENT_TASK_FINISHED,
    KEY_AUDIO,
    KEY_BASE_RESP,
    KEY_DATA,
    KEY_STATUS,
    KEY_STATUS_MSG,
)
from .tts_client import SenseTimeTTSClient, SenseTimeTTSError

if TYPE_CHECKING:
    from .config import SenseTimeTTSConfig as ConfigClass

logger = logging.getLogger("sensetime.tts")

CHINESE_PUNCT = frozenset("，。！？；：""''（）【】《》、…—")


def _is_punctuation_only(s: str) -> bool:
    for c in s:
        if (
            not unicodedata.category(c).startswith("P")
            and c not in CHINESE_PUNCT
            and not c.isspace()
        ):
            return False
    return True


class SenseTimeTTS(TTS):
    """SenseTime SenseAudio TTS provider for LiveKit Agents.

    Maintains a **single persistent WebSocket connection** for the lifetime of
    the LiveKit room.  The connection is established and a TTS task started in
    :meth:`warmup`; all conversation turns reuse that connection by sending
    repeated ``task_continue`` messages.  :meth:`shutdown` sends ``task_finish``
    and disconnects.

    Lifecycle::

        warmup()   → connect → connected_success → task_start → heartbeat starts
        stream()   → task_continue(×N)            # no task_finish between turns
        shutdown() → stop heartbeat → task_finish → disconnect

    If the connection dies unexpectedly, :meth:`_ensure_conn` re-runs warmup
    transparently before the next stream begins.
    """

    def __init__(
        self,
        config: SenseTimeTTSConfig | None = None,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        sample_rate: int | None = None,
        speed: float | None = None,
        vol: float | None = None,
        pitch: int | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            # Build kwargs for only the values explicitly passed; unset fields
            # fall back to SenseTimeTTSConfig defaults (which read from env vars).
            kwargs: dict = {}
            if api_url is not None:
                kwargs["api_url"] = api_url
            if api_key is not None:
                kwargs["api_key"] = api_key
            if model is not None:
                kwargs["model"] = model
            if voice is not None:
                kwargs["voice"] = voice
            if sample_rate is not None:
                kwargs["sample_rate"] = sample_rate
            if speed is not None:
                kwargs["speed"] = speed
            if vol is not None:
                kwargs["vol"] = vol
            if pitch is not None:
                kwargs["pitch"] = pitch
            self._config = SenseTimeTTSConfig(**kwargs)
        super().__init__(
            capabilities=TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=self._config.sample_rate,
            num_channels=1,
        )
        self._conn_options = conn_options or APIConnectOptions()
        self._http_session: aiohttp.ClientSession | None = None

        # Round 8 R8.7 — Connection pool architecture.
        #
        # The single ``self._conn`` (persistent across all turns) is replaced
        # by a pool of ``pool_size`` pre-warmed connections. Each turn
        # acquires one, uses it for one reply, then discards it via
        # ``mark_dirty`` regardless of normal completion vs cancel. The pool
        # spawns a background replacement so the next turn finds a warm
        # conn ready (zero-latency cancel — see scripts/spike_tts_pool_vs_
        # reconnect.py: 62% latency reduction vs single-conn-reconnect).
        #
        # The pool is provider-agnostic (``plugins/tts/_pool.py``); SenseTime-
        # specific warm-up (connect + task_start) is in ``_factory_warm_conn``.
        from eidolon.livekit.plugins.tts._pool import TTSConnectionPool
        self._pool: TTSConnectionPool[SenseTimeTTSClient] = TTSConnectionPool(
            factory=self._factory_warm_conn,
            disposer=self._dispose_conn,
            size=self._config.pool_size,
            label="SenseTimeTTSPool",
        )
        # Track the most recently acquired conn for tests/debug parity with
        # the old "self._conn" introspection. Not used by production code.
        self._conn: SenseTimeTTSClient | None = None
        # G21 (2026-05-18): weakref for the in-flight synth stream, so the
        # interrupted-context snapshot can read its ``_pushed_text``.
        self._current_stream: weakref.ReferenceType["SenseTimeSynthesizeStream"] | None = None

        # Serialises stream() callers (defense-in-depth — at most one
        # SynthesizeStream should be active per session under LiveKit).
        self._stream_lock = asyncio.Lock()
        self._stream_active: bool = False

        logger.info(
            "[SenseTimeTTS] initialized provider=sensetime model=%s voice=%s "
            "sample_rate=%s api_url=%s",
            self._config.model,
            self._config.voice,
            self._config.sample_rate,
            self._config.api_url,
        )

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def provider(self) -> str:
        return "sensetime"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _create_connection(self) -> SenseTimeTTSClient:
        """Create a new TTSConnection with current config."""
        return SenseTimeTTSClient(
            uri=self._config.api_url,
            api_key=self._config.api_key,
            model=self._config.model,
            voice_id=self._config.voice,
            sample_rate=self._config.sample_rate,
            speed=self._config.speed,
            vol=self._config.vol,
            pitch=self._config.pitch,
            audio_format=self._config.audio_format or "pcm",
            bitrate=self._config.bitrate,
            http_session=self._ensure_http_session(),
        )

    async def _factory_warm_conn(self) -> SenseTimeTTSClient:
        """Pool factory: open one connection, complete handshake +
        ``task_start``, start its heartbeat. Returns a connection ready to
        accept ``task_continue`` immediately.
        """
        conn = self._create_connection()
        ok = await conn.connect()
        if not ok:
            raise SenseTimeTTSError(
                f"Failed to connect TTS at {self._config.api_url}"
            )
        await conn.send_task_start()
        # Each pooled conn runs its own heartbeat. While the conn sits
        # in the pool warm queue, no real traffic flows so heartbeat
        # is what keeps the SenseAudio server-side idle timer from
        # dropping it. ``stream_active_check`` returns False for a
        # warm conn (this conn is not the active one), True only for
        # the conn currently checked out — but we don't actually need
        # to skip heartbeats on the active conn (task_continue traffic
        # already resets the timer; an extra empty task_continue is
        # cheap and safely no-op).
        conn.start_heartbeat(
            interval=30.0,
            stream_active_check=lambda: False,
        )
        return conn

    async def _dispose_conn(self, conn: SenseTimeTTSClient) -> None:
        """Pool disposer: stop heartbeat + close WS. Best-effort, bounded
        by short timeouts — we're discarding regardless of outcome."""
        try:
            conn.stop_heartbeat()
        except Exception:
            pass
        try:
            await asyncio.wait_for(conn.disconnect(), timeout=2.0)
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("[SenseTimeTTS] dispose: disconnect best-effort: %s", e)

    async def warmup(self) -> None:
        """Open the pool of pre-warmed connections.

        Round 8 R8.7 — replaces the old "single persistent connection"
        warmup. Now opens ``pool_size`` connections in parallel; any
        failure of an individual connection is logged but doesn't fail
        warmup if at least one succeeds.

        Idempotent — calling on an already-warm pool is a no-op.
        """
        if self._pool.warm_count >= self._config.pool_size:
            logger.debug("[SenseTimeTTS] warmup: pool already full, skipping")
            return
        logger.info(
            "[SenseTimeTTS] warming up pool (target size=%d)...",
            self._config.pool_size,
        )
        await self._pool.warmup()
        logger.info(
            "[SenseTimeTTS] warmup complete — pool ready (warm=%d) "
            "model=%s voice=%s sample_rate=%d",
            self._pool.warm_count,
            self._config.model,
            self._config.voice,
            self._config.sample_rate,
        )

    async def _acquire_conn(self) -> SenseTimeTTSClient:
        """Get one warm connection from the pool for use by a stream.

        Replaces the old ``_ensure_conn``. The caller (a SynthesizeStream)
        MUST eventually call ``self._pool.mark_dirty(conn)`` regardless
        of normal completion or cancellation — this is the
        "discard every turn" contract that prevents cross-turn audio
        leakage.
        """
        conn = await self._pool.acquire()
        self._conn = conn  # tracked for debug/test parity only
        return conn

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ):
        """Synthesize text in batch mode (non-streaming)."""
        return self._synthesize_with_stream(
            text, conn_options=conn_options or self._conn_options
        )

    def stream(
        self, *, conn_options: APIConnectOptions | None = None
    ) -> "SenseTimeSynthesizeStream":
        """Create a streaming synthesis session."""
        s = SenseTimeSynthesizeStream(
            tts=self, conn_options=conn_options or self._conn_options
        )
        # G21 (2026-05-18): expose stream weakref for interrupted-context.
        self._current_stream = weakref.ref(s)
        return s

    @property
    def current_pushed_text(self) -> str:
        """G21 (2026-05-18): see BailianTTS.current_pushed_text — same
        contract: the text currently being synthesized by the in-flight
        stream, or empty string if no active stream."""
        if self._current_stream is None:
            return ""
        stream = self._current_stream()
        if stream is None:
            return ""
        return getattr(stream, "_pushed_text", "") or ""

    async def shutdown(self) -> None:
        """Drain the connection pool, dispose all warm conns, close HTTP session."""
        logger.info(
            "[SenseTimeTTS] shutdown: draining pool (warm=%d, in-flight=%d)",
            self._pool.warm_count, self._pool.in_flight_refills,
        )
        try:
            await self._pool.shutdown()
        except Exception as e:
            logger.warning("[SenseTimeTTS] pool shutdown error (ignored): %s", e)
        self._conn = None

        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                logger.warning("[SenseTimeTTS] error closing HTTP session")
            self._http_session = None
        logger.info("[SenseTimeTTS] shutdown complete")


class SenseTimeSynthesizeStream(SynthesizeStream):
    """Streaming TTS using the SenseAudio WebSocket protocol.

    Reuses the persistent connection maintained by :class:`SenseTimeTTS`.
    Each stream sends ``task_continue`` messages and exits when the server
    signals completion (``status=2`` / ``is_final=True``) or after a 200ms
    quiesce following the end of input.  ``task_finish`` is never sent between
    turns — only :meth:`SenseTimeTTS.shutdown` sends it.
    """

    def __init__(
        self, *, tts: SenseTimeTTS, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._config = tts._config

        # Per-stream state
        self._audio_byte_stream: AudioByteStream | None = None
        self._leading_silence_trimmed = False
        self._pcm_total_bytes = 0
        self._log_audio_diag = self._config.log_audio_diag
        self._task_failed = False
        self._task_failed_error: SenseTimeTTSError | None = None
        self._input_done = asyncio.Event()
        self._task_finished_received = False
        # True when at least one task_continue with text was sent this turn
        self._text_sent = False
        # Number of non-empty task_continue messages sent this turn
        self._text_chunks_sent: int = 0
        # Number of "batch-end" markers received (task_continued with empty audio)
        self._batch_ends_received: int = 0
        # True when _receive_loop sent the connection-closed sentinel (None)
        self._conn_closed: bool = False
        # Monotonic time when _input_done was set (used by no-first-audio guard)
        self._input_done_time: float = 0.0
        # Event-driven exit signal — set by _check_exit() when any exit
        # condition (A-F in the design) is satisfied. _run() waits on this.
        self._exit_event: asyncio.Event = asyncio.Event()

    def _convert_audio(self, raw_data: bytes) -> bytes:
        """Decode audio data based on format, returning PCM."""
        fmt = self._audio_format.lower()
        if fmt in ("pcm", "raw", ""):
            return raw_data

        try:
            import av
        except ImportError:
            logger.warning(
                "[SenseTimeSynthesizeStream] PyAV not installed, "
                "cannot decode format=%s (install with: pip install av)",
                fmt,
            )
            return raw_data

        try:
            ioctx = io.BytesIO(raw_data)
            container = av.open(ioctx)
            stream = container.streams.audio[0]
            pcm_frames: list[bytes] = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                if arr.ndim == 2:
                    arr = arr.T.ravel()
                pcm_frames.append(arr.tobytes())
            container.close()
            return b"".join(pcm_frames)
        except Exception as e:
            logger.warning(
                "[SenseTimeSynthesizeStream] failed to decode %s: %s",
                fmt,
                e,
            )
            return raw_data

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """Main synthesis loop: use the persistent connection, emit audio."""
        config = self._config

        # Reset per-invocation state
        self._task_failed = False
        self._task_failed_error = None
        self._pcm_total_bytes = 0
        self._leading_silence_trimmed = False
        self._input_done = asyncio.Event()
        self._task_finished_received = False
        self._text_sent = False
        self._text_chunks_sent = 0
        self._batch_ends_received = 0
        self._conn_closed = False
        self._input_done_time = 0.0
        self._exit_event = asyncio.Event()
        self._audio_format = config.audio_format or "pcm"

        # Initialize output emitter for streaming PCM
        output_emitter.initialize(
            request_id=uuid.uuid4().hex[:16],
            sample_rate=config.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=uuid.uuid4().hex[:16])
        logger.info(
            "[SenseTimeSynthesizeStream] _run: stream initialized "
            "model=%s voice=%s sample_rate=%d speed=%.2f",
            config.model,
            config.voice,
            config.sample_rate,
            config.speed,
        )

        # Round 8 R8.7 — pool architecture.
        # Acquire ONE warm conn from the pool. This conn is *exclusively* this
        # stream's; we ALWAYS discard it (mark_dirty) when the turn ends,
        # whether the turn completed normally or was cancelled. This gives
        # zero-latency cancel response (next turn finds another warm conn
        # already in the pool).
        async with self._tts._stream_lock:
            client = await self._tts._acquire_conn()
            self._tts._stream_active = True

            # Initialize audio byte stream for PCM framing (60ms frames)
            self._audio_byte_stream = AudioByteStream(
                sample_rate=config.sample_rate,
                num_channels=1,
                samples_per_channel=int(config.sample_rate * 60 // 1000),
            )

            # Message channel for bridging callback-style to async
            msg_ch: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def on_message(msg: dict[str, Any] | None) -> None:
                await msg_ch.put(msg)

            client._on_message_callback = on_message

            # Start receive and input loops concurrently
            recv_task = asyncio.create_task(
                self._recv_loop(client, msg_ch, output_emitter)
            )
            input_task = asyncio.create_task(self._input_loop(client))

            # Schedule the no-first-audio safety check (condition F).
            # Fires once 5s after _input_done is set; _check_exit re-evaluates.
            no_audio_guard_task = asyncio.create_task(
                self._no_first_audio_guard()
            )

            # Round 8 R8.12.b: track whether _run was cancelled by the
            # framework (user interrupt, agent close, etc.) vs. exited
            # naturally via _exit_event. The framework's TTSNode sees
            # "got text, produced 0 audio" and emits tts_error → triggers
            # a useless retry that races with the new turn's TTS. We
            # distinguish cancellation in the finally clause and emit a
            # specific log so logs are diagnosable.
            cancelled_by_framework = False
            try:
                # Event-driven exit: _check_exit() sets _exit_event when any of
                # conditions A-F (see _check_exit) is satisfied. The 30s safety
                # net is purely a defence against state-machine bugs — under
                # normal operation _exit_event is set well before that.
                try:
                    await asyncio.wait_for(self._exit_event.wait(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "[SenseTimeSynthesizeStream] exit_event safety net "
                        "triggered (30s) — state machine did not converge "
                        "(text_chunks_sent=%d batch_ends_received=%d "
                        "pcm_total_bytes=%d input_done=%s text_sent=%s "
                        "task_finished=%s task_failed=%s conn_closed=%s)",
                        self._text_chunks_sent,
                        self._batch_ends_received,
                        self._pcm_total_bytes,
                        self._input_done.is_set(),
                        self._text_sent,
                        self._task_finished_received,
                        self._task_failed,
                        self._conn_closed,
                    )
            except asyncio.CancelledError:
                cancelled_by_framework = True
            except BaseException:
                pass
            finally:
                # Make sure _input_done is set so _input_loop exits if still alive
                self._input_done.set()

                for task in (recv_task, input_task, no_audio_guard_task):
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(
                        recv_task, input_task, no_audio_guard_task,
                        return_exceptions=True,
                    )
                except BaseException:
                    pass

                # Flush any remaining partial audio frame before releasing the lock
                if self._audio_byte_stream is not None:
                    for remaining in self._audio_byte_stream.flush():
                        output_emitter.push(remaining.data.tobytes())
                        self._pcm_total_bytes += len(remaining.data)
                    self._audio_byte_stream = None

                client._on_message_callback = None
                self._tts._stream_active = False

                output_emitter.end_segment()
                output_emitter.end_input()

                # Round 8 R8.7 — discard this conn unconditionally.
                # On cancel: server still has queued task_continues; new
                # turn must use a fresh conn so leftover audio can't
                # leak in.
                # On normal completion: same path for architectural
                # symmetry — relying on "task ended cleanly" is fragile
                # and the cost is just a background WS open (~2s, hidden
                # from user-facing latency).
                # Background refill keeps the pool topped up.
                try:
                    await self._tts._pool.mark_dirty(client)
                except Exception:
                    logger.exception(
                        "[SenseTimeSynthesizeStream] mark_dirty failed; "
                        "pool may be left short-staffed"
                    )

        # ``self._tts._conn`` cleanup: this attr is debug-only; the pool
        # has already taken care of the actual conn lifecycle. Reset to
        # None so callers don't see a dangling reference.
        if self._tts._conn is client:
            self._tts._conn = None

        # Propagate server-side task failures if audio was NOT successfully received
        if self._task_failed and self._task_failed_error is not None:
            if self._pcm_total_bytes == 0:
                raise self._task_failed_error from None
            else:
                logger.warning(
                    "[SenseTimeSynthesizeStream] task_failed received after successful audio "
                    "(pcm_total_bytes=%d) - treating as non-fatal: %s",
                    self._pcm_total_bytes,
                    self._task_failed_error,
                )
                self._task_failed = False
                self._task_failed_error = None

        # Round 8 R8.12.b: distinguish cancelled-mid-flight from natural
        # synthesis-complete. The framework's TTSNode treats "0 audio for
        # non-zero text" as an error and retries — but for cancellation,
        # retry is wrong (the new turn is already starting; retry races
        # with it and causes the "speech not done in time after
        # interruption" cascade observed in production logs 2026-05-07).
        #
        # Detection: cancel-mid-flight is signalled by ANY of:
        #   (a) batch_ends_received < text_chunks_sent — server hadn't
        #       finished all sent batches when we exited
        #   (b) cancelled_by_framework — explicit CancelledError caught
        #   (c) we got text input but produced zero audio AND framework
        #       hadn't normally completed (no task_finished received) —
        #       handles the case where aggregator buffered text but
        #       never flushed because cancellation hit during _input_loop
        text_received_without_audio = (
            len(self._pushed_text) > 0
            and self._pcm_total_bytes == 0
            and not self._task_finished_received
        )
        cancelled_mid_flight = (
            self._batch_ends_received < self._text_chunks_sent
            or cancelled_by_framework
            or text_received_without_audio
        )

        if cancelled_mid_flight:
            logger.info(
                "[SenseTimeSynthesizeStream] synthesis_cancelled "
                "(turn interrupted before all audio arrived; "
                "chunks_sent=%d batch_ends=%d pcm_total_bytes=%d "
                "framework_cancel=%s)",
                self._text_chunks_sent,
                self._batch_ends_received,
                self._pcm_total_bytes,
                cancelled_by_framework,
            )
            # Two different exit paths:
            #
            # 1. ``cancelled_by_framework`` — framework's `wait_for`
            #    propagated CancelledError because a new user turn started
            #    while we were synthesising. This is the **normal** turn-
            #    interruption flow; re-raise CancelledError so the framework
            #    treats it as a clean cancellation (no retry, no ERROR-level
            #    session event). Emitting APIError here was the source of
            #    the spurious ``type='tts_error'`` flood seen in production
            #    2026-05-07: 8 cancels = 8 false ERRORs / 2 minutes.
            #
            # 2. We detect mid-flight by counting (batch_ends < chunks_sent)
            #    or text-without-audio, but framework didn't actually cancel
            #    us. That's a genuine anomaly — the WS dropped, the server
            #    misbehaved, etc. Raise APIError(retryable=False) so the
            #    framework records the failure but doesn't retry the same
            #    text on a new stream.
            if cancelled_by_framework:
                raise asyncio.CancelledError()
            from livekit.agents import APIError
            raise APIError(
                "tts synthesis ended mid-flight without framework cancel "
                f"(chunks_sent={self._text_chunks_sent} "
                f"batch_ends={self._batch_ends_received})",
                body=None,
                retryable=False,
            )

        if self._log_audio_diag:
            logger.info(
                "[SenseTimeSynthesizeStream] synthesis_complete "
                "pcm_total_bytes=%d input_text_chars=%d",
                self._pcm_total_bytes,
                len(self._pushed_text),
            )

    def _check_exit(self) -> None:
        """Evaluate exit conditions and set ``_exit_event`` if any holds.

        Called whenever state changes that could trigger an exit:
          - task_continued / task_failed / task_finished received
          - _input_done is set (input loop ended)
          - connection-closed sentinel received
          - no-first-audio guard fires (5s after input_done with zero audio)

        Conditions (any one set → exit):
          A. task_failed
          B. task_finished
          C. _input_done AND not _text_sent              (empty input)
          D. _input_done AND _batch_ends_received >= _text_chunks_sent  (main path)
          E. _conn_closed                                (connection died)
          F. _input_done AND _text_sent AND no audio nor batch-ends for 5s
        """
        if self._exit_event.is_set():
            return

        # A
        if self._task_failed:
            self._exit_event.set()
            return
        # B
        if self._task_finished_received:
            self._exit_event.set()
            return
        # E
        if self._conn_closed:
            logger.debug(
                "[SenseTimeSynthesizeStream] exit: connection closed "
                "(pcm_total_bytes=%d)",
                self._pcm_total_bytes,
            )
            self._exit_event.set()
            return

        if not self._input_done.is_set():
            return

        # C
        if not self._text_sent:
            self._exit_event.set()
            return

        # D — primary state-driven exit
        if (
            self._text_chunks_sent > 0
            and self._batch_ends_received >= self._text_chunks_sent
        ):
            logger.info(
                "[SenseTimeSynthesizeStream] exit: all batches received "
                "(chunks=%d batch_ends=%d pcm_total_bytes=%d)",
                self._text_chunks_sent,
                self._batch_ends_received,
                self._pcm_total_bytes,
            )
            self._exit_event.set()
            return

        # F — handled by _no_first_audio_guard task to keep this method
        # purely state-driven (no time references).

    async def _no_first_audio_guard(self) -> None:
        """Condition F: server is silent after input ended.

        Wait until ``_input_done`` is set, then 5s. If we still have no audio
        and no batch-ends, the server is silent — log error and force exit.
        """
        try:
            await self._input_done.wait()
            await asyncio.sleep(5.0)
            if (
                self._text_sent
                and self._batch_ends_received == 0
                and self._pcm_total_bytes == 0
                and not self._exit_event.is_set()
            ):
                logger.error(
                    "[SenseTimeSynthesizeStream] no_first_audio_guard: "
                    "5s after input_done with zero audio and zero batch-ends "
                    "(text_chunks_sent=%d) — forcing exit",
                    self._text_chunks_sent,
                )
                self._exit_event.set()
        except asyncio.CancelledError:
            pass

    async def _process_message(
        self, msg: dict[str, Any], output_emitter: AudioEmitter
    ) -> None:
        """Process a single message from the server, then re-evaluate exit.

        Updates state (audio buffer, batch-end count, terminal flags) and
        calls ``_check_exit()`` so the event-driven loop can wake.
        """
        event = msg.get("event")
        data = msg.get(KEY_DATA) or {}

        if event in (EVENT_TASK_CONTINUE, EVENT_TASK_CONTINUED):
            await self._handle_audio_chunk(data, output_emitter)
        elif event == EVENT_TASK_FINISHED:
            await self._handle_task_finished(output_emitter)
        elif event == EVENT_TASK_FAILED:
            await self._handle_task_failed(msg)

        self._check_exit()

    async def _recv_loop(
        self,
        client: SenseTimeTTSClient,
        msg_ch: asyncio.Queue[dict[str, Any] | None],
        output_emitter: AudioEmitter,
    ) -> None:
        """Pure message-driven receive loop.

        Blocks on ``msg_ch.get()`` until a message or the close sentinel
        (``None``) arrives. State updates are routed through
        ``_process_message`` → ``_check_exit`` so exit is signalled via
        ``_exit_event`` rather than time-based polling.
        """
        while True:
            try:
                msg = await msg_ch.get()
            except asyncio.CancelledError:
                return

            if msg is None:
                # Connection-closed sentinel from TTSConnection._receive_loop.
                logger.debug(
                    "[SenseTimeSynthesizeStream] _recv_loop: connection closed sentinel"
                )
                self._conn_closed = True
                self._check_exit()
                return

            await self._process_message(msg, output_emitter)

    async def _handle_audio_chunk(
        self, data: dict[str, Any], output_emitter: AudioEmitter
    ) -> None:
        """Decode hex audio (if present) and update batch-end counter.

        SenseAudio sends one task_continued per text segment; the segment
        ends with a "batch-end marker" — a task_continued whose ``audio``
        is empty/missing while ``status`` is present (typically status=0).
        Counting these markers vs ``_text_chunks_sent`` lets us know when
        all submitted text has been synthesised.
        """
        audio_hex = data.get(KEY_AUDIO)

        # Detect batch-end marker: empty/missing audio + status field present.
        if (not audio_hex) and (KEY_STATUS in data):
            self._batch_ends_received += 1
            logger.debug(
                "[SenseTimeSynthesizeStream] batch-end marker received "
                "(received=%d/sent=%d)",
                self._batch_ends_received,
                self._text_chunks_sent,
            )
            # Flush any pending partial frame at the segment boundary.
            byte_stream = self._audio_byte_stream
            if byte_stream is not None:
                for remaining in byte_stream.flush():
                    output_emitter.push(remaining.data.tobytes())
                    self._pcm_total_bytes += len(remaining.data)
            output_emitter.flush()
            return

        if not audio_hex or not isinstance(audio_hex, str):
            return

        try:
            raw = bytes.fromhex(audio_hex)
        except ValueError:
            logger.warning(
                "[SenseTimeSynthesizeStream] invalid hex audio, skipping"
            )
            return

        if not raw:
            return

        # Discard leading silence
        if not self._leading_silence_trimmed:
            if not any(raw):
                return
            self._leading_silence_trimmed = True

        converted = self._convert_audio(raw)

        byte_stream = self._audio_byte_stream
        if byte_stream is None:
            return

        for pcm_frame in byte_stream.push(converted):
            output_emitter.push(pcm_frame.data.tobytes())
            self._pcm_total_bytes += len(pcm_frame.data)

    async def _handle_task_finished(
        self, output_emitter: AudioEmitter
    ) -> None:
        """Handle task_finished (server closed task — typically only at shutdown).

        In persistent connection mode this is unexpected during a stream turn,
        but we handle it gracefully: flush any remaining audio and mark the turn
        ended so _run()'s finalizer skips a second flush.
        """
        self._task_finished_received = True
        byte_stream = self._audio_byte_stream
        if byte_stream:
            for remaining in byte_stream.flush():
                output_emitter.push(remaining.data.tobytes())
                self._pcm_total_bytes += len(remaining.data)
            self._audio_byte_stream = None  # prevent double-flush in _run() finalizer

        if self._task_failed and self._pcm_total_bytes > 0:
            logger.info(
                "[SenseTimeSynthesizeStream] task_finished after audio received, "
                "clearing task_failed (pcm_total_bytes=%d)",
                self._pcm_total_bytes,
            )
            self._task_failed = False
            self._task_failed_error = None

    async def _handle_task_failed(self, msg: dict[str, Any]) -> None:
        """Handle task_failed: store the error for later propagation."""
        logger.error(
            "[SenseTimeSynthesizeStream] task_failed raw_msg: %s",
            msg,
        )
        base_resp = msg.get(KEY_BASE_RESP) or {}
        err = (
            base_resp.get(KEY_STATUS_MSG)
            if isinstance(base_resp, dict)
            else str(base_resp)
        )
        if not err:
            data = msg.get(KEY_DATA) or {}
            base_resp = data.get(KEY_BASE_RESP) or {}
            err = (
                base_resp.get(KEY_STATUS_MSG)
                if isinstance(base_resp, dict)
                else str(data)
            )
        logger.error(
            "[SenseTimeSynthesizeStream] task_failed extracted err=%r from base_resp=%r",
            err,
            base_resp,
        )
        self._task_failed = True
        self._task_failed_error = SenseTimeTTSError(
            f"SenseAudio task failed: {err}", recoverable=False
        )

    async def _input_loop(self, client: SenseTimeTTSClient) -> None:
        """Read from _input_ch and forward text to the server via task_continue.

        Round 8 R8.2: tokens are funneled through a ``SentenceAggregator``
        that batches them into sentence-sized chunks before submission.
        Each ``task_continue`` is its own batch on the SenseAudio server
        (with ~600 ms gap between batches), so sending one token at a
        time produces choppy audio. The aggregator collapses ~12 tokens
        into ~3 sentences, dramatically improving listening experience.

        In persistent-connection mode, ``task_finish`` is NOT sent here;
        only by :meth:`SenseTimeTTS.shutdown`. ``_text_chunks_sent``
        tracks segments sent (post-aggregation) so ``_recv_loop`` can
        match it against received batch-end markers.
        """
        from ._aggregator import SentenceAggregator

        first_token_wait = self._config.first_token_timeout

        async def emit_segment(text: str) -> None:
            """Aggregator callback: clean + send a finalised segment.

            We strip leading/trailing punctuation to avoid weird audio
            at segment boundaries (e.g. "，hello，" gets a leading pause
            that sounds odd). Internal punctuation is kept — that drives
            sentence prosody on the server side.
            """
            cleaned = text.strip()
            # Strip leading punctuation/whitespace.
            while cleaned and (
                _is_punctuation_only(cleaned[0]) or cleaned[0].isspace()
            ):
                cleaned = cleaned[1:]
            # Strip trailing punctuation/whitespace.
            while cleaned and (
                _is_punctuation_only(cleaned[-1]) or cleaned[-1].isspace()
            ):
                cleaned = cleaned[:-1]
            if not cleaned:
                return
            await client.send_task_continue(cleaned)
            self._text_chunks_sent += 1
            self._text_sent = True

        aggregator = SentenceAggregator(
            emit_segment,
            soft_min_chars=self._config.aggregator_soft_min_chars,
            hard_max_chars=self._config.aggregator_hard_max_chars,
            idle_ms=self._config.aggregator_idle_ms,
        )

        try:
            try:
                token = await asyncio.wait_for(
                    self._input_ch.__anext__(), timeout=first_token_wait
                )
                if not isinstance(token, SynthesizeStream._FlushSentinel):
                    if isinstance(token, str):
                        await aggregator.feed(token)
            except asyncio.TimeoutError:
                logger.warning(
                    "[SenseTime TTS] no first token within %.1fs — "
                    "LLM upstream slow? exiting input loop",
                    first_token_wait,
                )
                return

            async for token in self._input_ch:
                if isinstance(token, SynthesizeStream._FlushSentinel):
                    # Framework signals end of a generation segment;
                    # flush whatever's buffered so audio doesn't wait for
                    # the next reply.
                    await aggregator.flush()
                    continue
                if isinstance(token, str):
                    await aggregator.feed(token)

            # Input channel closed naturally — flush whatever's left.
            await aggregator.flush()

        except asyncio.CancelledError:
            raise
        except StopAsyncIteration:
            await aggregator.flush()
        except Exception as e:
            logger.error("[SenseTime TTS] _input_loop error: %s", e)
        finally:
            await aggregator.aclose()
            self._input_done.set()
            self._input_done_time = asyncio.get_event_loop().time()
            # Wake the exit waiter so it can re-evaluate state.
            self._check_exit()
