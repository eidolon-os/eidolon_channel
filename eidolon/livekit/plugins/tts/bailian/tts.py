"""Bailian CosyVoice TTS plugin for LiveKit Agents."""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
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
        self._conn: BailianTTSClient | None = None

    @property
    def provider(self) -> str:
        return "bailian"

    @property
    def model(self) -> str:
        return self._config.model

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
        )

    async def _factory_warm_conn(self) -> BailianTTSClient:
        conn = self._create_connection()
        ok = await conn.connect()
        if not ok:
            raise BailianTTSError(
                f"Failed to connect Bailian TTS: {self._config.api_url}",
                recoverable=True,
            )
        await conn.start_task()
        return conn

    async def _dispose_conn(self, conn: BailianTTSClient) -> None:
        try:
            await asyncio.wait_for(conn.disconnect(), timeout=2.0)
        except Exception as e:
            logger.debug("[BailianTTS] dispose best-effort: %s", e)

    async def warmup(self) -> None:
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
        await self._pool.shutdown()
        self._conn = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                logger.warning("[BailianTTS] failed to close HTTP session")
            self._http_session = None

    async def _acquire_conn(self) -> BailianTTSClient:
        try:
            conn = await self._pool.acquire()
        except RuntimeError as e:
            raise BailianTTSError(str(e), recoverable=True) from e
        self._conn = conn
        return conn

    def synthesize(self, text: str, *, conn_options: APIConnectOptions | None = None):
        return self._synthesize_with_stream(
            text, conn_options=conn_options or self._conn_options
        )

    def stream(
        self, *, conn_options: APIConnectOptions | None = None
    ) -> "BailianSynthesizeStream":
        return BailianSynthesizeStream(
            tts=self,
            conn_options=conn_options or self._conn_options,
        )


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

        output_emitter.initialize(
            request_id=uuid.uuid4().hex[:16],
            sample_rate=self._config.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=uuid.uuid4().hex[:16])

        async with self._tts._stream_lock:
            client = await self._tts._acquire_conn()
            self._tts._stream_active = True
            self._audio_byte_stream = AudioByteStream(
                sample_rate=self._config.sample_rate,
                num_channels=1,
                samples_per_channel=int(self._config.sample_rate * 60 // 1000),
            )
            msg_ch: asyncio.Queue[dict[str, Any] | bytes | None] = asyncio.Queue()

            async def on_message(msg: dict[str, Any]) -> None:
                await msg_ch.put(msg)

            async def on_binary(data: bytes) -> None:
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
                await asyncio.wait_for(self._exit_event.wait(), timeout=45.0)
            except asyncio.TimeoutError:
                raise APIError("bailian tts stream timeout", body=None, retryable=False)
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
            raise APIError(str(self._task_failed_error), body=None, retryable=False)

        if self._conn_closed and not self._task_finished:
            raise APIError("bailian tts connection closed unexpectedly", body=None, retryable=False)

        if self._log_audio_diag:
            logger.info(
                "[BailianSynthesizeStream] complete text_chars=%d pcm_bytes=%d",
                len(self._pushed_text),
                self._pcm_total_bytes,
            )

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
            self._task_failed_error = BailianTTSError(err_str, recoverable=False)

    async def _handle_audio_chunk(
        self, chunk: bytes, output_emitter: AudioEmitter
    ) -> None:
        if not chunk:
            return
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
                # DashScope restricts per-continue-task text size.
                if len(cleaned) > 20000:
                    for i in range(0, len(cleaned), 20000):
                        await client.send_continue(cleaned[i : i + 20000])
                else:
                    await client.send_continue(cleaned)
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

        aggregator = SentenceAggregator(
            emit_segment,
            soft_min_chars=self._config.aggregator_soft_min_chars,
            hard_max_chars=self._config.aggregator_hard_max_chars,
            idle_ms=self._config.aggregator_idle_ms,
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

            async for token in self._input_ch:
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
        finally:
            await aggregator.aclose()
            self._input_done.set()

    async def _no_first_audio_guard(self) -> None:
        try:
            await self._input_done.wait()
            await asyncio.sleep(self._config.no_first_audio_timeout)
            if (
                self._text_sent
                and self._pcm_total_bytes == 0
                and not self._task_finished
                and self._task_failed_error is None
            ):
                self._task_failed_error = BailianTTSError(
                    "no audio received before timeout",
                    recoverable=True,
                )
                self._exit_event.set()
        except asyncio.CancelledError:
            return
