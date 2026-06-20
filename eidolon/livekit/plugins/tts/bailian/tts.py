"""Bailian CosyVoice TTS plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
import weakref
from typing import Any

import aiohttp
import numpy as np
from livekit.agents import APIError
from livekit.agents.tts import AudioEmitter, SynthesizeStream, TTS, TTSCapabilities
from livekit.agents.types import APIConnectOptions
from livekit.agents.utils.audio import AudioByteStream

from eidolon.livekit.plugins.tts._pool import TTSConnectionPool
from eidolon.livekit.plugins.tts._aggregator import SentenceAggregator

from .config import BailianTTSConfig
from .protocol import EVENT_TASK_FAILED, EVENT_TASK_FINISHED, parse_event
from .tts_client import BailianTTSClient, BailianTTSError

logger = logging.getLogger("bailian.tts")


class BailianTTS(TTS):
    def __init__(
        self,
        config: BailianTTSConfig | None = None,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        sample_rate: int | None = None,
        audio_format: str | None = None,
        speech_rate: float | None = None,
        pool_size: int | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            kwargs: dict[str, Any] = {}
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
            if audio_format is not None:
                kwargs["audio_format"] = audio_format
            if speech_rate is not None:
                kwargs["speech_rate"] = speech_rate
            if pool_size is not None:
                kwargs["pool_size"] = pool_size
            self._config = BailianTTSConfig(**kwargs)

        super().__init__(
            capabilities=TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=self._config.sample_rate,
            num_channels=1,
        )
        self._conn_options = conn_options or APIConnectOptions()
        self._http_session: aiohttp.ClientSession | None = None
        self._pool: TTSConnectionPool[BailianTTSClient] = TTSConnectionPool(
            factory=self._factory_warm_conn,
            disposer=self._dispose_conn,
            size=self._config.pool_size,
            label="BailianTTSPool",
            refill_failure_backoff=self._config.pool_refill_backoff,
            acquire_wait_timeout=self._config.pool_acquire_timeout,
            enable_inline_slow_path=False,
            # F2 (2026-05-16): evict stale conns before handing them out.
            # dashscope server idles WS after ~30s; 25s gives 5s safety margin.
            max_idle_sec=self._config.pool_max_idle_sec,
        )
        self._stream_lock = asyncio.Lock()
        self._stream_active = False
        self._closing = False
        self._conn: BailianTTSClient | None = None
        # G21 (2026-05-18): weakref to the currently-active synth stream so
        # callers (StreamingPipeline._snapshot_interrupted_context) can read
        # ``_pushed_text`` for the in-flight reply — session.history only
        # gets the assistant message AFTER the speech_handle winds down,
        # which is AFTER our cancel snapshot fires.
        self._current_stream: weakref.ReferenceType["BailianSynthesizeStream"] | None = None

    @property
    def provider(self) -> str:
        return "bailian"

    @property
    def model(self) -> str:
        return self._config.model

    def emit_provider_event(self, name: str, **payload: Any) -> None:
        """Emit provider-level TTS timing events for Channel observability.

        Mirrors ``BailianFunASRSTT.emit_provider_event`` so the StreamingPipeline
        can bridge TTS provider truth (request start, first provider audio byte)
        into the per-turn timeline, separately from the agent-state-driven
        ``tts_first_audio_at`` experience mark.
        """

        self.emit(
            "provider_event",
            {
                "provider": self.provider,
                "event": name,
                "timestamp": time.monotonic(),
                "model": self.model,
                **payload,
            },
        )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _create_connection(self) -> BailianTTSClient:
        return BailianTTSClient(
            uri=self._config.api_url,
            api_key=self._config.api_key,
            model=self._config.model,
            voice=self._config.voice,
            audio_format=self._config.audio_format or "pcm",
            sample_rate=self._config.sample_rate,
            rate=self._config.speech_rate,
            volume=self._config.volume,
            pitch=self._config.pitch,
            task_started_timeout=self._config.task_started_timeout,
            task_finished_timeout=self._config.task_finished_timeout,
            http_session=self._ensure_http_session(),
            ws_heartbeat_sec=self._config.ws_heartbeat_sec,
        )

    async def _factory_warm_conn(self) -> BailianTTSClient:
        conn = self._create_connection()
        ok = await conn.connect()
        if not ok:
            raise BailianTTSError(
                f"Failed to connect Bailian TTS: {self._config.api_url}",
                recoverable=True,
            )
        return conn

    async def _dispose_conn(self, conn: BailianTTSClient) -> None:
        try:
            await asyncio.wait_for(conn.disconnect(), timeout=2.0)
        except Exception as e:
            logger.debug("[BailianTTS] dispose best-effort: %s", e)

    async def warmup(self) -> None:
        if self._closing:
            raise RuntimeError("[BailianTTS] cannot warmup after shutdown started")
        if self._pool.warm_count >= self._config.pool_size:
            return
        # F5 (2026-05-16): synchronously open only ``pool_size_bootstrap``
        # conns (default 3); background refill brings pool up to ``pool_size``
        # after warmup returns. Cuts cold-start from ~5s to ~3.5s while still
        # reaching full pool target within ~1-2s of first turn.
        bootstrap = min(self._config.pool_size_bootstrap, self._config.pool_size)
        logger.info(
            "[BailianTTS] warming up pool bootstrap=%d target=%d",
            bootstrap, self._config.pool_size,
        )
        await self._pool.warmup(count=bootstrap)
        # Kick background refill so the pool fills to ``pool_size`` while
        # the first turn is happening. acquire() also triggers this, but
        # being explicit avoids a corner case where warmup completes and
        # nothing else acquires for a while (e.g. silent room).
        if bootstrap < self._config.pool_size:
            self._pool._maybe_refill()  # type: ignore[attr-defined]

    async def shutdown(self) -> None:
        self._closing = True
        await self._pool.shutdown()
        self._conn = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                logger.warning("[BailianTTS] failed to close HTTP session")
            self._http_session = None

    async def _acquire_conn(self) -> BailianTTSClient:
        if self._closing:
            raise BailianTTSError("[BailianTTS] shutting down", recoverable=False)
        try:
            conn = await self._pool.acquire()
        except RuntimeError as e:
            raise BailianTTSError(str(e), recoverable=not self._closing) from e
        self._conn = conn
        return conn

    def synthesize(self, text: str, *, conn_options: APIConnectOptions | None = None):
        return self._synthesize_with_stream(
            text, conn_options=conn_options or self._conn_options
        )

    def stream(
        self, *, conn_options: APIConnectOptions | None = None
    ) -> "BailianSynthesizeStream":
        s = BailianSynthesizeStream(
            tts=self,
            conn_options=conn_options or self._conn_options,
        )
        # G21 (2026-05-18): register weakref so _snapshot_interrupted_context
        # can find this stream's _pushed_text even before session.history
        # gets the assistant message.
        self._current_stream = weakref.ref(s)
        return s

    @property
    def current_pushed_text(self) -> str:
        """G21 (2026-05-18): the text currently being synthesized by the
        in-flight synth stream, if any. Empty string if no active stream
        or the stream has been GC'd. Used by the interrupted-context
        snapshot to capture exactly what the agent was saying at the
        moment of cancellation, rather than the older session.history
        which only commits the message after speech winds down."""
        if self._current_stream is None:
            return ""
        stream = self._current_stream()
        if stream is None:
            return ""
        return getattr(stream, "_pushed_text", "") or ""


class BailianSynthesizeStream(SynthesizeStream):
    def __init__(
        self,
        *,
        tts: BailianTTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._config = tts._config
        self._audio_byte_stream: AudioByteStream | None = None
        self._exit_event = asyncio.Event()
        self._task_failed_error: Exception | None = None
        self._task_finished = False
        self._text_sent = False
        self._input_done = asyncio.Event()
        self._conn_closed = False
        self._pcm_total_bytes = 0
        self._log_audio_diag = self._config.log_audio_diag
        self._provider_task_started = False
        # G12 (2026-05-17): wall-clock marker for the no-audio guard.
        # Set on first successful send_continue. None means "no text sent
        # yet" — the guard ignores this state.
        self._first_send_continue_time: float | None = None
        # Emit tts_provider_first_audio exactly once per stream.
        self._first_provider_audio_emitted = False
        # Stall-watchdog clock (reset per-run); see _await_stream_completion.
        self._last_stream_activity_at = 0.0

    def _convert_audio(self, raw_data: bytes) -> bytes:
        fmt = self._config.audio_format.lower()
        if fmt in ("pcm", "raw", ""):
            return raw_data
        try:
            import av
        except ImportError:
            logger.warning("[BailianSynthesizeStream] PyAV not installed")
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
            logger.warning("[BailianSynthesizeStream] decode failed: %s", e)
            return raw_data

    async def _run(self, output_emitter: AudioEmitter) -> None:
        self._exit_event = asyncio.Event()
        self._task_failed_error = None
        self._task_finished = False
        self._text_sent = False
        self._input_done = asyncio.Event()
        self._conn_closed = False
        self._pcm_total_bytes = 0
        self._first_send_continue_time = None  # G12 reset per-stream
        self._provider_task_started = False
        # Reset per-attempt so a framework retry re-evaluates "did THIS attempt
        # produce audio" — drives the recoverable classification in
        # _handle_json_event and the first-audio provider event.
        self._first_provider_audio_emitted = False
        # Stall-watchdog clock: last time the stream made progress (provider
        # message received or text token sent). See _await_stream_completion.
        self._last_stream_activity_at = time.monotonic()

        output_emitter.initialize(
            request_id=uuid.uuid4().hex[:16],
            sample_rate=self._config.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=uuid.uuid4().hex[:16])

        if self._tts._closing:
            output_emitter.end_segment()
            output_emitter.end_input()
            logger.debug("[BailianSynthesizeStream] ignored stream during shutdown")
            return

        self._tts.emit_provider_event("tts_stream_started")
        async with self._tts._stream_lock:
            client = await self._tts._acquire_conn()
            self._tts.emit_provider_event("tts_connection_acquired")
            self._tts._stream_active = True
            self._audio_byte_stream = AudioByteStream(
                sample_rate=self._config.sample_rate,
                num_channels=1,
                samples_per_channel=int(self._config.sample_rate * 60 // 1000),
            )
            msg_ch: asyncio.Queue[dict[str, Any] | bytes | None] = asyncio.Queue()

            async def on_message(msg: dict[str, Any]) -> None:
                # Mark at the WS-receive layer (before the downstream push) so a
                # full-duplex duck that backpressures playback is never mistaken
                # for a provider stall.
                self._mark_stream_activity()
                await msg_ch.put(msg)

            async def on_binary(data: bytes) -> None:
                self._mark_stream_activity()
                await msg_ch.put(data)

            async def on_closed() -> None:
                await msg_ch.put(None)

            client._on_message_callback = on_message
            client._on_binary_callback = on_binary
            client._on_closed_callback = on_closed

            recv_task = asyncio.create_task(self._recv_loop(msg_ch, output_emitter))
            input_task = asyncio.create_task(self._input_loop(client))
            no_audio_guard_task = asyncio.create_task(self._no_first_audio_guard())
            try:
                await self._await_stream_completion()
            finally:
                self._input_done.set()
                for t in (recv_task, input_task, no_audio_guard_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(
                    recv_task, input_task, no_audio_guard_task, return_exceptions=True
                )
                byte_stream = self._audio_byte_stream
                if byte_stream is not None:
                    for remaining in byte_stream.flush():
                        output_emitter.push(remaining.data.tobytes())
                        self._pcm_total_bytes += len(remaining.data)
                    self._audio_byte_stream = None
                output_emitter.end_segment()
                output_emitter.end_input()

                client._on_message_callback = None
                client._on_binary_callback = None
                client._on_closed_callback = None
                self._tts._stream_active = False
                await self._tts._pool.mark_dirty(client)

        if self._tts._conn is client:
            self._tts._conn = None

        if self._task_failed_error is not None:
            retryable = bool(getattr(self._task_failed_error, "recoverable", False))
            raise APIError(str(self._task_failed_error), body=None, retryable=retryable)

        if self._conn_closed and not self._task_finished:
            raise APIError(
                "bailian tts connection closed unexpectedly",
                body=None,
                retryable=True,
            )

        if self._log_audio_diag:
            logger.info(
                "[BailianSynthesizeStream] complete text_chars=%d pcm_bytes=%d",
                len(self._pushed_text),
                self._pcm_total_bytes,
            )

    def _mark_stream_activity(self) -> None:
        """Record that the stream just made progress (provider sent something,
        or we sent text). Drives the stall watchdog in _await_stream_completion."""
        self._last_stream_activity_at = time.monotonic()

    async def _await_stream_completion(self) -> None:
        """Wait for the TTS stream to finish, aborting only on a genuine stall.

        Replaces the old fixed total-lifetime cap, which truncated long replies
        whose text the LLM streams over many seconds (the stream was healthy —
        audio kept flowing — but the blanket ceiling expired anyway). Liveness
        here is "is the stream still making progress?": activity is marked on
        every provider message and every text token (``_mark_stream_activity``).

        Because activity is marked at the WebSocket-receive layer, a full-duplex
        barge-in duck that backpressures downstream playback does NOT look like a
        stall (the provider is still feeding us; the mixer is just holding the
        audio). A confirmed barge-in cancels this coroutine via CancelledError —
        not a stall. We abort only when nothing flows for
        ``stream_stall_timeout_sec``; the first-token / inter-token / no-first-
        audio guards still cover the startup phase. 0 / negative disables.
        """
        stall = self._config.stream_stall_timeout_sec
        if stall <= 0:
            await self._exit_event.wait()
            return
        while not self._exit_event.is_set():
            remaining = stall - (time.monotonic() - self._last_stream_activity_at)
            if remaining <= 0:
                idle = time.monotonic() - self._last_stream_activity_at
                raise APIError(
                    f"bailian tts stream stalled (no audio/text for {idle:.0f}s)",
                    body=None,
                    retryable=True,
                )
            try:
                # Wait until the deadline; if activity arrives meanwhile, the
                # deadline moves forward and we simply re-arm on the next loop.
                await asyncio.wait_for(self._exit_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                continue

    async def _recv_loop(
        self,
        msg_ch: asyncio.Queue[dict[str, Any] | bytes | None],
        output_emitter: AudioEmitter,
    ) -> None:
        while True:
            try:
                msg = await msg_ch.get()
            except asyncio.CancelledError:
                return
            if msg is None:
                self._conn_closed = True
                self._exit_event.set()
                return
            if isinstance(msg, bytes):
                await self._handle_audio_chunk(msg, output_emitter)
                continue
            await self._handle_json_event(msg)
            if self._task_finished or self._task_failed_error is not None:
                self._exit_event.set()
                return

    async def _handle_json_event(self, msg: dict[str, Any]) -> None:
        event = parse_event(msg)
        if event == EVENT_TASK_FINISHED:
            self._task_finished = True
            return
        if event == EVENT_TASK_FAILED:
            header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
            error_code = header.get("error_code", "Unknown")
            error_message = header.get("error_message", "task failed")
            err_str = f"[{error_code}] {error_message}"
            logger.error("[BailianSynthesizeStream] task-failed: %s", err_str)
            # A task-failed that arrives BEFORE any audio has been produced is a
            # setup/handshake-phase failure — most commonly a stale pre-warmed
            # pool connection. CosyVoice rejects ``run-task`` ("Invalid
            # action('run-task')! Please follow the protocol!") when the socket
            # has sat idle between connect and the first run-task (the pool warms
            # connections eagerly, and run-task is only sent once the first LLM
            # tokens are aggregated — so a slow first token can span that gap).
            # These are safe to retry: the framework's SynthesizeStream replays
            # the buffered input through a fresh ``_run`` (→ a fresh pooled
            # connection, run-task sent immediately since the text is already
            # buffered). Once audio has flowed, a failure is mid-synthesis and
            # NOT recoverable (retrying would re-emit speech — the framework's
            # pushed_duration guard also blocks it).
            self._task_failed_error = BailianTTSError(
                err_str,
                recoverable=not self._first_provider_audio_emitted,
            )

    async def _handle_audio_chunk(
        self, chunk: bytes, output_emitter: AudioEmitter
    ) -> None:
        if not chunk:
            return
        if not self._first_provider_audio_emitted:
            self._first_provider_audio_emitted = True
            self._tts.emit_provider_event(
                "tts_provider_first_audio", bytes=len(chunk)
            )
        converted = self._convert_audio(chunk)
        byte_stream = self._audio_byte_stream
        if byte_stream is None:
            return
        for frame in byte_stream.push(converted):
            output_emitter.push(frame.data.tobytes())
            self._pcm_total_bytes += len(frame.data)

    async def _input_loop(self, client: BailianTTSClient) -> None:
        first_token_wait = self._config.first_token_timeout

        async def emit_segment(text: str) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            # Guard: if the connection was already closed (e.g. by an
            # interrupt while the aggregator still had buffered text),
            # skip the send silently instead of surfacing a fatal error.
            # This race is expected during interruptions — the framework
            # cancels TTS generation, which closes the WebSocket, but
            # the aggregator's last flush may already be in-flight.
            if self._conn_closed or self._exit_event.is_set():
                logger.debug(
                    "[BailianSynthesizeStream] emit_segment skipped "
                    "(conn_closed=%s, exit=%s): %r",
                    self._conn_closed, self._exit_event.is_set(),
                    cleaned[:40],
                )
                return
            try:
                if not self._provider_task_started:
                    self._tts.emit_provider_event("tts_request_started")
                    await client.start_task()
                    self._provider_task_started = True

                async def send_text_part(part: str) -> None:
                    await client.send_continue(part)
                    # Feeding text is progress too — keeps the stall watchdog
                    # happy through the LLM-streaming phase before audio starts.
                    self._mark_stream_activity()
                    # G12 (2026-05-17): wall-clock marker for the no-audio
                    # watchdog. Set on FIRST send_continue only — subsequent
                    # ones don't shift the deadline.
                    if self._first_send_continue_time is None:
                        self._first_send_continue_time = time.monotonic()
                        self._tts.emit_provider_event(
                            "tts_first_text_sent",
                            chars=len(part),
                        )

                # DashScope restricts per-continue-task text size.
                if len(cleaned) > 20000:
                    for i in range(0, len(cleaned), 20000):
                        await send_text_part(cleaned[i : i + 20000])
                else:
                    await send_text_part(cleaned)
                self._text_sent = True
            except BailianTTSError as e:
                if e.recoverable and (self._conn_closed or self._exit_event.is_set()):
                    # Connection closed during interrupt — expected, not an error.
                    logger.debug(
                        "[BailianSynthesizeStream] send_continue skipped "
                        "(connection closed during interrupt)"
                    )
                else:
                    raise

        # G5 (2026-05-16): optionally override first-sentence behaviour
        # to flush more eagerly on fast LLMs. ``0`` (default) → leave unset
        # → aggregator uses its standard soft_min for all sentences.
        first_min = self._config.aggregator_first_sentence_soft_min_chars or None
        aggregator = SentenceAggregator(
            emit_segment,
            soft_min_chars=self._config.aggregator_soft_min_chars,
            hard_max_chars=self._config.aggregator_hard_max_chars,
            idle_ms=self._config.aggregator_idle_ms,
            first_sentence_soft_min_chars=first_min,
            first_sentence_flush_any_punct=(
                self._config.aggregator_first_sentence_flush_any_punct
            ),
        )
        try:
            try:
                token = await asyncio.wait_for(
                    self._input_ch.__anext__(),
                    timeout=first_token_wait,
                )
                if isinstance(token, str):
                    await aggregator.feed(token)
            except asyncio.TimeoutError:
                logger.warning("[BailianSynthesizeStream] first token timeout")
                return

            # G11 (2026-05-17): inter-token timeout. Previously the
            # ``async for`` had no timeout — if the framework's input_ch
            # didn't propagate ``StopAsyncIteration`` after the LLM stream
            # ended, this loop hung indefinitely (observed in 2026-05-17
            # round-2: input_loop blocked, _input_done never set, the
            # no-audio guard waited on it forever).
            inter_token_timeout = self._config.inter_token_timeout
            while True:
                try:
                    token = await asyncio.wait_for(
                        self._input_ch.__anext__(),
                        timeout=inter_token_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[BailianSynthesizeStream] inter-token timeout "
                        "(%.1fs) — assuming LLM stream ended, force-flushing",
                        inter_token_timeout,
                    )
                    break
                except StopAsyncIteration:
                    break
                if isinstance(token, SynthesizeStream._FlushSentinel):
                    await aggregator.flush()
                    continue
                if isinstance(token, str):
                    await aggregator.feed(token)
            await aggregator.flush()
            await client.send_finish()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._task_failed_error = e
            self._exit_event.set()
        finally:
            await aggregator.aclose()
            if not self._text_sent and self._task_failed_error is None:
                self._exit_event.set()
            self._input_done.set()

    async def _no_first_audio_guard(self) -> None:
        """G12 (2026-05-17): wall-clock watchdog. Previously waited on
        ``self._input_done``, which never fires when ``_input_loop`` hangs
        (the exact failure mode of the round-2 incident). Now uses an
        independent wall-clock timer: as soon as the FIRST send_continue
        succeeds, we have ``no_first_audio_timeout`` seconds to see at
        least one PCM byte. If not — abort and let framework retry.
        """
        try:
            no_audio_timeout = self._config.no_first_audio_timeout
            while not self._exit_event.is_set():
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    return
                if self._first_send_continue_time is None:
                    continue  # No text sent yet
                if self._task_finished or self._task_failed_error is not None:
                    return  # Already concluded
                if self._pcm_total_bytes > 0:
                    return  # Got audio — watchdog done
                elapsed = time.monotonic() - self._first_send_continue_time
                if elapsed > no_audio_timeout:
                    logger.warning(
                        "[BailianSynthesizeStream] no audio %.1fs after "
                        "first send_continue (likely dashscope task death) "
                        "— aborting",
                        elapsed,
                    )
                    self._task_failed_error = BailianTTSError(
                        f"no audio {elapsed:.1f}s after first text sent",
                        recoverable=True,
                    )
                    self._exit_event.set()
                    return
        except asyncio.CancelledError:
            return
