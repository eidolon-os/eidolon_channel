"""Streaming implementation for Bailian FunASR STT.

BailianFunASRSpeechStream bridges the LiveKit RecognizeStream contract with
the Bailian FunASR WebSocket protocol.

Architecture:
- Inherits from ``RecognizeStream`` so the framework manages the stream lifecycle.
- ``RecognizeStream.flush()`` sends a ``_FlushSentinel`` into ``_input_ch``,
  which causes ``send_loop`` to call ``conn.finish()`` and exit — allowing
  FunASR to return the final transcript for this segment.
- ``RecognizeStream.end_input()`` calls ``flush()`` then closes ``_input_ch``.
  The framework calls ``end_input()`` when a user turn ends, ensuring
  ``send_loop`` always terminates cleanly.
- ``send_loop`` drains ``_input_ch`` and sends audio to the FunASR WebSocket.
- ``recv_loop`` receives transcriptions and emits :class:`SpeechEvent` objects
  back to the caller through the event channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .connection_manager import BailianConnectionError, BailianConnectionManager
from .models import (
    FunASREventType,
    FunASRResultGenerated,
    FunASRSentence,
    FunASRTaskFailed,
    FunASRTaskFinished,
    parse_funasr_message,
)

from livekit.agents import stt as lk_stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.stt.stt import RecognizeStream
from livekit.agents.utils import aio

if TYPE_CHECKING:
    from livekit.agents.types import APIConnectOptions
    from .stt import BailianFunASRSTT

logger = logging.getLogger("bailian.stt_stream")

# 100 ms of 16 kHz mono PCM = 1600 samples
_CHUNK_SAMPLES = 1600


class BailianFunASRSpeechStream(lk_stt.RecognizeStream):
    """LiveKit RecognizeStream implementation for Bailian FunASR.

    Inherits from ``RecognizeStream`` so the framework manages the stream
    lifecycle automatically via ``flush()`` / ``end_input()``.

    Usage::

        stream = stt.stream()
        stream.push_frame(audio_frame)
        # framework calls flush() when VAD detects end of speech
        # framework calls end_input() when user turn ends
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                print(ev.alternatives[0].text)
    """

    def __init__(
        self,
        stt: "BailianFunASRSTT",
        *,
        conn_options: "APIConnectOptions | None" = None,
        sample_rate: int = 16000,
        language: str | None = None,
    ):
        from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType

        actual_conn_options = (
            conn_options if conn_options is not None else stt.conn_options
        )
        actual_sample_rate = sample_rate
        actual_language = language or stt.language

        # Parent init: creates _input_ch, _event_ch, _task, _resampler, etc.
        super().__init__(
            stt=stt,
            conn_options=actual_conn_options,
            sample_rate=actual_sample_rate,
        )

        # Stash types on self for use in handlers
        self._SpeechData = SpeechData
        self._SpeechEvent = SpeechEvent
        self._SpeechEventType = SpeechEventType

        # Aliases to match previous code
        self._stt_ref = stt
        self._sample_rate = actual_sample_rate
        self._language = actual_language

        # AudioByteStream accumulates raw frames into fixed-size chunks (100 ms)
        from livekit.agents.utils.audio import AudioByteStream

        self._audio_buf = AudioByteStream(
            sample_rate=actual_sample_rate,
            num_channels=1,
            samples_per_channel=_CHUNK_SAMPLES,
        )

        # Per-utterance state — reset on each _run()
        self._parser_confirmed: list[FunASRSentence] = []
        self._last_interim_text: str = ""
        self._finished: bool = False
        self._conn: "BailianConnectionManager | None" = None

    # ------------------------------------------------------------------
    # _run: called by RecognizeStream._main_task with retry logic
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main body of the receive loop. Called by _main_task in a retry loop."""
        self._parser_confirmed.clear()
        self._last_interim_text = ""
        self._finished = False

        conn = BailianConnectionManager(
            api_url=self._stt_ref.api_url,
            api_key=self._stt_ref.api_key,
            model=self._stt_ref.model,
            sample_rate=self._sample_rate,
            itn=self._stt_ref.itn,
            language_hints=self._language,
        )
        self._conn = conn

        # _FlushSentinel is a nested class of RecognizeStream.
        # Get the type so we can use isinstance() in send_loop.
        from livekit.agents.stt.stt import RecognizeStream

        flush_sentinel_type: type = RecognizeStream._FlushSentinel  # type: ignore[attr-defined]

        async def send_loop() -> None:
            frames_sent = 0
            logger.info("[Bailian STT] send_loop started")
            try:
                async for item in self._input_ch:  # type: ignore[attr-defined]
                    if self._finished:
                        logger.info(
                            "[Bailian STT] send_loop exiting: _finished=True frames_sent=%d",
                            frames_sent,
                        )
                        break

                    if isinstance(item, flush_sentinel_type):
                        remaining = self._audio_buf.flush()
                        flushed_bytes = 0
                        for chunk in remaining:
                            d = bytes(chunk.data.tobytes())
                            flushed_bytes += len(d)
                            await conn.send_audio(d)
                        logger.info(
                            "[Bailian STT] flush sentinel received — flushed %d audio chunks "
                            "(%d bytes), calling finish()",
                            len(remaining),
                            flushed_bytes,
                        )
                        await conn.finish()
                        break

                    frame: rtc.AudioFrame = item
                    pcm = frame.data.tobytes()
                    chunks = self._audio_buf.push(pcm)
                    for chunk in chunks:
                        await conn.send_audio(bytes(chunk.data.tobytes()))
                        frames_sent += 1
                    # Only log every 50 frames to avoid log spam
                    if frames_sent % 50 == 0:
                        logger.info(
                            "[Bailian STT] send_loop: pushed frame size=%d bytes total_frames=%d",
                            len(pcm),
                            frames_sent,
                        )
            except asyncio.CancelledError:
                logger.info(
                    "[Bailian STT] send_loop cancelled (frames_sent=%d)", frames_sent
                )

        async def recv_loop() -> None:
            messages_received = 0
            try:
                async for raw in conn._ws:  # noqa: SLF001
                    if self._finished:
                        logger.info(
                            "[Bailian STT] recv_loop exiting: _finished=True messages=%d",
                            messages_received,
                        )
                        break
                    messages_received += 1
                    if isinstance(raw, str):
                        try:
                            data: dict[str, Any] = dict[Any, Any](**json.loads(raw))  # type: ignore[arg-type]
                        except Exception:
                            logger.warning(
                                "[Bailian STT] recv_loop: ignoring non-dict JSON: %s",
                                raw[:100],
                            )
                            continue
                        logger.info(
                            "[Bailian STT] recv_loop: msg #%d event=%s data=%s",
                            messages_received,
                            data.get("event", "unknown"),
                            json.dumps(data)[:200],
                        )
                        await self._handle_message(data)
            except asyncio.CancelledError:
                logger.info(
                    "[Bailian STT] recv_loop cancelled (messages=%d)",
                    messages_received,
                )
            except Exception as e:
                if not self._finished:
                    logger.exception("[Bailian STT] recv_loop error")
                    self._emit_error(e, recoverable=True)

        try:
            logger.info(
                "[Bailian STT] _run: connecting to %s model=%s sample_rate=%d",
                self._stt_ref.api_url,
                self._stt_ref.model,
                self._sample_rate,
            )
            await conn.connect()
            send_task = asyncio.create_task(send_loop())
            recv_task = asyncio.create_task(recv_loop())

            logger.info(
                "[Bailian STT] _run: send_loop + recv_loop started, waiting for completion..."
            )
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            logger.info("[Bailian STT] _run: both loops completed")

        except BailianConnectionError as e:
            logger.error(
                "[Bailian STT] _run: BailianConnectionError %s recoverable=%s",
                e,
                e.recoverable,
            )
            self._emit_error(e, recoverable=e.recoverable)
        except Exception as e:
            logger.exception("[Bailian STT] _run: unexpected error")
            self._emit_error(e, recoverable=True)
        finally:
            await conn.close()
            logger.info("[Bailian STT] _run: connection closed")

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Dispatch a FunASR JSON message to the appropriate handler."""
        try:
            event_name, parsed = parse_funasr_message(data)
        except ValueError:
            logger.debug("Unrecognised FunASR message: %s", data)
            return

        if event_name == FunASREventType.RESULT_GENERATED.value:
            await self._handle_result(parsed)  # type: ignore[arg-type]
        elif event_name == FunASREventType.TASK_FINISHED.value:
            await self._handle_task_finished(parsed)  # type: ignore[arg-type]
        elif event_name == FunASREventType.TASK_FAILED.value:
            await self._handle_task_failed(parsed)  # type: ignore[arg-type]
        elif event_name in (
            FunASREventType.TASK_STARTED.value,
            FunASREventType.HEARTBEAT.value,
        ):
            pass
        else:
            logger.warning("[Bailian STT] unhandled FunASR event: %s", event_name)

    async def _handle_result(self, result: FunASRResultGenerated) -> None:
        """Map a FunASR result-generated event to LiveKit SpeechEvent types."""
        if not result.sentences:
            logger.warning("[Bailian STT] _handle_result: no sentences in result")
            return

        latest = result.latest_sentence()
        if latest is None:
            logger.warning("[Bailian STT] _handle_result: latest_sentence() returned None")
            return

        text = latest.text
        if not text:
            logger.warning("[Bailian STT] _handle_result: empty text in latest sentence")
            return

        if latest.sentence_end:
            self._parser_confirmed.append(latest)
            confirmed_text = "".join(s.text for s in self._parser_confirmed)
            logger.debug("[STT-DBG] FINAL confirmed_text=%r", confirmed_text)
            words = self._build_timed_strings(*self._parser_confirmed)
            start_s = (
                self._parser_confirmed[0].begin_time / 1000.0
                if self._parser_confirmed
                else 0.0
            )
            end_s = latest.end_time / 1000.0
            logger.info(
                "[Bailian STT] FINAL_TRANSCRIPT text=%r start=%.3fs end=%.3fs",
                confirmed_text,
                start_s,
                end_s,
            )
            self._push_event(
                self._SpeechEvent(
                    type=self._SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        self._SpeechData(
                            language=self._language,
                            text=confirmed_text,
                            start_time=start_s,
                            end_time=end_s,
                            confidence=1.0,
                            words=words,
                        )
                    ],
                )
            )
            self._parser_confirmed.clear()
        else:
            if text != self._last_interim_text:
                self._last_interim_text = text
                logger.debug("[STT-DBG] INTERIM text=%r", text)
                logger.info(
                    "[Bailian STT] INTERIM_TRANSCRIPT text=%r", text
                )
                self._push_event(
                    self._SpeechEvent(
                        type=self._SpeechEventType.INTERIM_TRANSCRIPT,
                        alternatives=[
                            self._SpeechData(
                                language=self._language,
                                text=text,
                                start_time=latest.begin_time / 1000.0,
                                end_time=latest.end_time / 1000.0,
                                confidence=1.0,
                                words=self._build_timed_strings(latest),
                            )
                        ],
                    )
                )

    async def _handle_task_finished(self, _: FunASRTaskFinished) -> None:
        """Server confirmed end of task."""
        logger.info("[Bailian STT] TASK_FINISHED received")
        self._finished = True
        self._push_event(self._SpeechEvent(type=self._SpeechEventType.END_OF_SPEECH))

    async def _handle_task_failed(self, failed: FunASRTaskFailed) -> None:
        """Server reported a task-level error."""
        logger.error("[Bailian STT] TASK_FAILED: %s (code=%s)", failed.error_message, failed.error_code)
        err = BailianConnectionError(
            f"Task {failed.task_id} failed: {failed.error_message} "
            f"(code={failed.error_code})",
            recoverable=True,
        )
        self._finished = True
        self._emit_error(err, recoverable=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_timed_strings(*sentences: FunASRSentence) -> list:
        """Convert FunASR word-level data to LiveKit SpeechData word list."""
        result = []
        for sentence in sentences:
            for word in sentence.words:
                result.append(
                    dict(
                        language="",
                        text=word.text,
                        start_time=word.begin_time / 1000.0,
                        end_time=word.end_time / 1000.0,
                        confidence=1.0,
                    )
                )
        return result

    def _push_event(self, event: "lk_stt.SpeechEvent") -> None:
        try:
            self._event_ch.send_nowait(event)  # type: ignore[attr-defined]
        except asyncio.QueueFull:
            logger.warning("SpeechStream event queue full, dropping event")

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        from livekit.agents.stt import STTError

        self._stt_ref.emit(
            "error",
            STTError(
                timestamp=time.time(),
                label=self._stt_ref.label,
                error=error,
                recoverable=recoverable,
            ),
        )
