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

from livekit.agents import APIError
from livekit.agents import stt as lk_stt

if TYPE_CHECKING:
    from livekit.agents.types import APIConnectOptions
    from .stt import BailianFunASRSTT

logger = logging.getLogger("bailian.stt_stream")

# 100 ms of 16 kHz mono PCM = 1600 samples
_CHUNK_SAMPLES = 1600

# Preemptive generation support: when an interim transcript has been stable
# (unchanged) for this long and is at least this many chars, emit a
# PREFLIGHT_TRANSCRIPT so the framework can start brain generation before the
# (late) FINAL arrives. Bailian's FINAL lands ~1s after speech stop, so this is
# what actually hides that wait. The FINAL may still differ (punctuation/ITN);
# the framework then discards the speculative turn and regenerates — clean, just
# without the speedup for that turn.
_PREFLIGHT_STABLE_MS = 320
_PREFLIGHT_MIN_CHARS = 2


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
        self._stream_id = f"bailian-{id(self):x}"
        self._first_audio_sent = False
        self._first_partial_seen = False
        self._first_final_seen = False
        self._turn_audio_observation: dict[str, Any] | None = None
        # Preflight (preemptive-generation) state.
        self._preflight_task: asyncio.Task | None = None
        self._preflight_sent_text: str | None = None
        # G7 (2026-05-17): track recv_loop errors so we can re-raise from
        # _run() and let the framework's RecognizeStream._main_task retry
        # mechanism kick in. Without this, an unexpected WS close exits
        # both loops silently and the whole session loses STT for good.
        self._recv_error: Exception | None = None
        # G16 (2026-05-17): VAD-gated audio forwarding for cost reduction.
        # Only constructed if config.gate_enabled. None when disabled means
        # send_loop falls back to the pre-G16 passthrough path.
        self._gate = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # _run: called by RecognizeStream._main_task with retry logic
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main body of the receive loop. Called by _main_task in a retry loop.

        G7 (2026-05-17): converts unexpected WS close into ``APIError``
        re-raise so the framework's RecognizeStream._main_task retry loop
        kicks in. Without that, server-side close (which DOES happen on
        DashScope FunASR — observed in production after AEC warmup
        windows) silently kills STT for the rest of the session.
        Clean termination paths (``self._finished=True`` via TASK_FINISHED
        or framework CancelledError) still return normally.
        """
        self._parser_confirmed.clear()
        self._last_interim_text = ""
        self._finished = False
        self._recv_error = None
        self._cancel_preflight()
        self._preflight_sent_text = None

        conn = BailianConnectionManager(
            api_url=self._stt_ref.api_url,
            api_key=self._stt_ref.api_key,
            model=self._stt_ref.model,
            sample_rate=self._sample_rate,
            itn=self._stt_ref.itn,
            language_hints=self._language,
            max_sentence_silence_ms=self._stt_ref.max_sentence_silence_ms,
        )
        self._conn = conn

        # _FlushSentinel is a nested class of RecognizeStream.
        # Get the type so we can use isinstance() in send_loop.
        from livekit.agents.stt.stt import RecognizeStream

        flush_sentinel_type: type = RecognizeStream._FlushSentinel  # type: ignore[attr-defined]

        # G16 (2026-05-17): construct VAD gate if enabled. The gate buffers
        # audio during silence (~1Hz keepalive only) and forwards real audio
        # only when VAD detects speech. Provides 55-65% cost reduction with
        # zero first-word latency cost (pre-roll buffer).
        gate_cfg = getattr(self._stt_ref, "_config", None)
        gate_enabled = bool(
            gate_cfg is not None and getattr(gate_cfg, "gate_enabled", False)
        )
        if gate_enabled:
            from ._gate import SttGate

            async def _send_audio_with_event(data: bytes) -> None:
                self._emit_provider_event_once(
                    "stt_first_audio_sent",
                    "_first_audio_sent",
                    bytes=len(data),
                )
                self._emit_turn_first_audio_if_needed(bytes_len=len(data))
                await conn.send_audio(data)

            self._gate = SttGate(
                sample_rate=self._sample_rate,
                sender=_send_audio_with_event,
                preroll_ms=gate_cfg.gate_preroll_ms,
                tail_window_ms=gate_cfg.gate_tail_window_ms,
                keepalive_interval_sec=gate_cfg.gate_keepalive_interval_sec,
                keepalive_frame_ms=gate_cfg.gate_keepalive_frame_ms,
                chunk_ms=int(_CHUNK_SAMPLES * 1000 / self._sample_rate),
                vad_high_threshold=gate_cfg.gate_vad_high_threshold,
                vad_low_threshold=gate_cfg.gate_vad_low_threshold,
                rms_threshold=gate_cfg.gate_rms_threshold,
            )
            await self._gate.start()
            logger.info(
                "[Bailian STT] G16 VAD gate enabled (preroll=%dms, tail=%dms, "
                "keepalive=%.1fHz, vad_high=%.2f, vad_low=%.2f, rms_thresh=%.0f)",
                gate_cfg.gate_preroll_ms,
                gate_cfg.gate_tail_window_ms,
                1.0 / gate_cfg.gate_keepalive_interval_sec,
                gate_cfg.gate_vad_high_threshold,
                gate_cfg.gate_vad_low_threshold,
                gate_cfg.gate_rms_threshold,
            )
        else:
            self._gate = None

        async def send_audio(data: bytes) -> None:
            self._emit_provider_event_once(
                "stt_first_audio_sent",
                "_first_audio_sent",
                bytes=len(data),
            )
            self._emit_turn_first_audio_if_needed(bytes_len=len(data))
            await conn.send_audio(data)

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
                            # G16: flush bypasses the gate — at stream end we
                            # want every last byte sent regardless of VAD.
                            await send_audio(d)
                        logger.info(
                            "[Bailian STT] flush sentinel received — flushed %d audio chunks "
                            "(%d bytes), calling finish()",
                            len(remaining),
                            flushed_bytes,
                        )
                        await conn.finish()
                        self._emit_provider_event(
                            "stt_flush_sent",
                            flushed_chunks=len(remaining),
                            flushed_bytes=flushed_bytes,
                        )
                        break

                    frame: rtc.AudioFrame = item
                    pcm = frame.data.tobytes()
                    chunks = self._audio_buf.push(pcm)
                    for chunk in chunks:
                        chunk_bytes = bytes(chunk.data.tobytes())
                        # G16: route through gate if enabled. Gate decides
                        # whether to buffer (GATED) or forward (FORWARDING).
                        if self._gate is not None:
                            await self._gate.feed(chunk_bytes)
                        else:
                            await send_audio(chunk_bytes)
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
                    # G7 (2026-05-17): record the error so _run can re-raise
                    # and trigger framework retry. The legacy _emit_error
                    # call only sent an "error" event but kept the run
                    # silently completing — framework saw success and never
                    # retried.
                    self._recv_error = e

        try:
            self._emit_provider_event("stt_stream_started")
            logger.info(
                "[Bailian STT] _run: connecting to %s model=%s sample_rate=%d",
                self._stt_ref.api_url,
                self._stt_ref.model,
                self._sample_rate,
            )
            await conn.connect()
            self._emit_provider_event("stt_ws_connected")
            send_task = asyncio.create_task(send_loop())
            recv_task = asyncio.create_task(recv_loop())

            logger.info("[Bailian STT] _run: send_loop + recv_loop started")
            pending: set[asyncio.Task[None]] = {send_task, recv_task}
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    await task

                if recv_task in done:
                    # In a streaming session the input side may stay open
                    # indefinitely. If the server-side read loop ended first,
                    # cancel the writer so recv errors can surface immediately.
                    if send_task in pending:
                        send_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                        pending.discard(send_task)
                    break
            logger.info("[Bailian STT] _run: both loops completed")

        except BailianConnectionError as e:
            logger.error(
                "[Bailian STT] _run: BailianConnectionError %s recoverable=%s",
                e,
                e.recoverable,
            )
            # G7 (2026-05-17): wrap as APIError and re-raise so framework's
            # RecognizeStream._main_task retry loop kicks in. Recoverable
            # → retryable=True triggers retry; non-recoverable → False
            # bypasses retry and surfaces immediately.
            try:
                await conn.close()
            except Exception:
                pass
            raise APIError(
                str(e),
                body=None,
                retryable=bool(e.recoverable),
            ) from e
        except asyncio.CancelledError:
            # Framework asked us to stop (e.g. session closing). Let it bubble.
            await conn.close()
            raise
        except Exception as e:
            logger.exception("[Bailian STT] _run: unexpected error")
            try:
                await conn.close()
            except Exception:
                pass
            raise APIError(str(e), body=None, retryable=True) from e
        else:
            # Loops completed without exception. Decide clean vs recoverable
            # based on whether we got server's TASK_FINISHED (self._finished)
            # or recv_loop recorded an error mid-flight.
            try:
                await conn.close()
            except Exception:
                pass
            logger.info("[Bailian STT] _run: connection closed")
            if self._recv_error is not None and not self._finished:
                # G7 (2026-05-17): server-side WS close mid-session — without
                # this re-raise, framework saw _run return normally and
                # never retried, killing STT for the rest of the session.
                err = self._recv_error
                logger.warning(
                    "[Bailian STT] _run: unclean exit (recv_error=%s) — "
                    "raising APIError to trigger framework retry",
                    err,
                )
                raise APIError(
                    f"Bailian STT WS closed unexpectedly: {err}",
                    body=None,
                    retryable=True,
                ) from err
        # G16: cleanup gate's keepalive task (idempotent / safe if absent).
        if self._gate is not None:
            try:
                metrics = self._gate.get_metrics()
                logger.info("[Bailian STT] G16 gate final metrics: %s", metrics)
                await self._gate.stop()
            except Exception as e:
                logger.warning("[Bailian STT] gate cleanup failed: %s", e)
            self._gate = None

    # G16: VAD signal bridge ------------------------------------------
    def notify_vad_state(self, probability: float, rms: float) -> None:
        """Push VAD inference output to the gate (called per-frame from the
        STT instance, which is called from streaming.py's VAD callback).
        No-op if the gate is disabled or not yet started."""
        if self._gate is not None:
            self._gate.notify_vad_state(probability, rms)

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
            self._cancel_preflight()
            self._emit_provider_event_once(
                "stt_provider_final",
                "_first_final_seen",
                text_preview=text[:80],
            )
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
                self._emit_provider_event_once(
                    "stt_provider_first_partial",
                    "_first_partial_seen",
                    text_preview=text[:80],
                )
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
                # Interim changed -> (re)arm the stability timer. If it goes
                # quiet for _PREFLIGHT_STABLE_MS we emit a PREFLIGHT_TRANSCRIPT
                # so the framework can start the brain before the late FINAL.
                self._arm_preflight(text, latest)

    def _arm_preflight(self, text: str, latest: FunASRSentence) -> None:
        self._cancel_preflight()
        if len(text.strip()) < _PREFLIGHT_MIN_CHARS or text == self._preflight_sent_text:
            return
        self._preflight_task = asyncio.create_task(
            self._emit_preflight_after_stable(text, latest)
        )

    async def _emit_preflight_after_stable(
        self, text: str, latest: FunASRSentence
    ) -> None:
        try:
            await asyncio.sleep(_PREFLIGHT_STABLE_MS / 1000.0)
        except asyncio.CancelledError:
            return
        # Still the latest interim, not finalized, not already sent.
        if (
            self._finished
            or text != self._last_interim_text
            or text == self._preflight_sent_text
        ):
            return
        self._preflight_sent_text = text
        logger.info("[Bailian STT] PREFLIGHT_TRANSCRIPT text=%r", text)
        self._push_event(
            self._SpeechEvent(
                type=self._SpeechEventType.PREFLIGHT_TRANSCRIPT,
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

    def _cancel_preflight(self) -> None:
        task = self._preflight_task
        self._preflight_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _handle_task_finished(self, _: FunASRTaskFinished) -> None:
        """Server confirmed end of task."""
        logger.info("[Bailian STT] TASK_FINISHED received")
        self._finished = True
        self._cancel_preflight()
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

    def _emit_provider_event(self, name: str, **payload: Any) -> None:
        self._stt_ref.emit_provider_event(
            name,
            stream_id=self._stream_id,
            language=self._language,
            **payload,
        )

    def _emit_provider_event_once(
        self,
        name: str,
        flag_name: str,
        **payload: Any,
    ) -> None:
        if getattr(self, flag_name):
            return
        setattr(self, flag_name, True)
        self._emit_provider_event(name, **payload)

    def observe_next_audio_for_turn(
        self,
        *,
        turn_id: str,
        speech_started_at: float,
    ) -> None:
        self._turn_audio_observation = {
            "turn_id": turn_id,
            "speech_started_at": speech_started_at,
        }

    def _emit_turn_first_audio_if_needed(self, *, bytes_len: int) -> None:
        observation = self._turn_audio_observation
        if observation is None:
            return
        self._turn_audio_observation = None
        self._emit_provider_event(
            "stt_turn_first_audio_sent",
            turn_id=observation["turn_id"],
            speech_started_at=observation["speech_started_at"],
            bytes=bytes_len,
        )

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
