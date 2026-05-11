"""Streaming implementation for SenseTime SenseAudio STT.

Bridges the LiveKit ``RecognizeStream`` contract with the SenseAudio STT
WebSocket protocol. **One stream covers the entire AgentSession** (Round
8 architecture alignment):

    _run() acquires _stream_lock          → serialises stream usage on shared WS
       ↓
    send_task_start                       → server: task_started
       ↓
    send_audio_binary ×N (100ms PCM)      → server: result_final ×M
       ↓                                              ↓
    VAD FlushSentinel: flush buffer       → emit FINAL_TRANSCRIPT per utterance
    (does NOT send task_finish — server VAD            (one task, many utterances)
     segments utterances inside one task)
       ↓
    framework closes audio_ch / user away → _exit_event set
       ↓
    finally: send_task_finish, await ack  → server: task_finished
       ↓
    _run() returns

Architectural principles (Round 8):

- **Session-long task, not per-utterance.** SenseAudio STT supports many
  result_finals inside one task_start/task_finish pair. LiveKit framework
  creates one ``stt.stream()`` per session and never recreates it (see
  ``audio_recognition._STTPipeline``). Mismatch between these two layers
  was the multi-turn dead-stream bug fixed in Round 8.

- **Pure event-driven exit, no wall-clock guesses.** Exit conditions
  (Round 8 R8.8 — three remain after removing user_away):
    A. ``_task_failed`` — fatal server error (caller decides recovery)
    B. ``_conn_closed`` — WS died (R8.3 reconnect logic kicks in higher up)
    C. ``_input_done`` — framework closed ``_input_ch`` (session ending)

  No 30s safety net (it was based on the wrong "per-utterance task" model
  and would fire on healthy long sessions, killing multi-turn).

  Round 7 G11's user_away exit was REMOVED in R8.8: ``user_state="away"``
  is a 15 s wall-clock timer (mutual silence detection), not a "user
  actually left" signal. Exiting on it killed session-long streams with
  no way to revive (framework's ``_STTPipeline`` doesn't recreate
  streams). The user_away signal is still observed via the watcher but
  treated as informational only.

- **Cleanup ack uses 5s timeout.** After exit, ``finally`` sends
  ``task_finish`` and awaits ``task_finished`` for ≤5s before forcibly
  closing. This timeout is a cleanup deadline, not a state-machine guess.

- **Binary audio frames.** Per the SenseAudio STT protocol, audio is sent
  as raw WebSocket binary frames, not JSON-wrapped hex.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from livekit import rtc

from livekit.agents import stt as lk_stt
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.stt.stt import RecognizeStream
from livekit.agents.utils.audio import AudioByteStream

from .connection import SenseTimeSTTError, STTConnection
from .protocol import (
    EVENT_RESULT_FINAL,
    EVENT_TASK_FAILED,
    EVENT_TASK_FINISHED,
    KEY_BASE_RESP,
    KEY_DATA,
    KEY_IS_FINAL,
    KEY_STATUS_MSG,
    KEY_TEXT,
)

if TYPE_CHECKING:
    from .stt import SenseTimeSTT

logger = logging.getLogger("sensetime.stt_stream")

# 100 ms of 16 kHz mono 16-bit PCM = 1600 samples = 3200 bytes
_CHUNK_SAMPLES = 1600


class SenseTimeSpeechStream(RecognizeStream):
    """LiveKit RecognizeStream — session-long; one stream serves all
    utterances in the AgentSession.

    Lifecycle:
        framework.push_frame ─┐
                              ├→ _input_ch  ──→ _send_loop  ──→ STT WS (binary)
        framework.flush ──────┤                                          │
                              └→ FlushSentinel → flush audio buffer      │
                                 (server VAD handles utterance bounds)   │
                                                                          ↓
                              _on_message_callback ←── result_final ×M
                                       ↓
                                  _process_message → emit FINAL_TRANSCRIPT
                                                     per utterance

        framework closes ──→ _input_done → _check_exit → _exit_event set
        audio_ch                                              ↓
                              finally: send_task_finish, await task_finished (5s)
                                                              ↓
                                                         _run() returns
    """

    def __init__(
        self,
        stt: "SenseTimeSTT",
        *,
        conn_options: "lk_stt.APIConnectOptions | None" = None,
        sample_rate: int = 16000,
        language: str = "zh",
    ) -> None:
        actual_conn_options = (
            conn_options if conn_options is not None else stt.conn_options
        )

        super().__init__(
            stt=stt,
            conn_options=actual_conn_options,
            sample_rate=sample_rate,
        )

        self._stt_ref = stt
        self._config = stt._config
        self._SpeechData = SpeechData
        self._SpeechEvent = SpeechEvent
        self._SpeechEventType = SpeechEventType
        self._sample_rate = sample_rate
        self._language = language

        # Per-utterance state — initialised here, reset at start of _run().
        # Most are also set in _run(); listing here makes the contract explicit.
        self._audio_buf: AudioByteStream | None = None
        self._task_failed: bool = False
        self._task_failed_error: SenseTimeSTTError | None = None
        self._task_finished_received: bool = False
        self._final_received: bool = False  # Got at least one result_final
        self._conn_closed: bool = False  # WS death sentinel
        # Round 8 R8.8: ``user_state == "away"`` is a wall-clock signal
        # (15 s of mutual silence — see livekit/agents agent_session.py:
        # 1446-1448). It does NOT mean "user has actually left"; it just
        # means "user has been quiet while agent was quiet for 15 s".
        # Round 7's G11 originally exited the STT stream on user_away
        # (correct for the per-utterance model: avoid wasting 30 s safety
        # net on noise). R8.1 made the stream session-long, which inverted
        # the cost: exiting kills the WHOLE session's STT — there's no way
        # to revive it because LiveKit's _STTPipeline never recreates streams.
        # R8.8: keep user_away as an INFORMATIONAL flag (logged for
        # diagnostics) but DO NOT use it to terminate the stream.
        self._user_away_received: bool = False
        self._pcm_total_bytes: int = 0
        self._input_done: asyncio.Event = asyncio.Event()
        self._exit_event: asyncio.Event = asyncio.Event()
        # Cleanup-phase signal: set when server returns task_finished after
        # we send task_finish in the finally clause. ``_run``'s cleanup
        # awaits this with a 5 s timeout to know the server has finished
        # processing all in-flight audio.
        self._task_finished_event: asyncio.Event = asyncio.Event()
        # Round 8 R8.3: distinguishes "framework closed audio_ch" (natural
        # end-of-session, no retry) from "WS died mid-stream" (recoverable,
        # retry inline). Set in ``_send_loop`` when its ``async for``
        # exits without an exception.
        self._input_ch_closed: bool = False

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Top-level entry: retry-on-conn-drop wrapper around ``_run_once``.

        Round 8 R8.3 — when the WS dies mid-session (network blip,
        server restart, transient task_failed), we attempt to reconnect
        inline rather than letting the stream end (the framework's
        ``_STTPipeline`` never re-creates streams). On budget exhaustion,
        emit an STT error and exit so the framework can close the
        AgentSession cleanly (client then reconnects).
        """
        config = self._config
        max_retries = config.max_stream_retries
        backoffs = config.stream_retry_backoffs

        # Cross-attempt state — preserved across reconnects.
        self._final_received = False
        self._user_away_received = False
        self._pcm_total_bytes = 0
        self._input_ch_closed = False
        self._audio_buf = AudioByteStream(
            sample_rate=config.sample_rate,
            num_channels=1,
            samples_per_channel=_CHUNK_SAMPLES,
        )

        logger.info(
            "[SenseTimeSpeechStream] _run: session-long stream initialized "
            "model=%s sample_rate=%d language=%s max_retries=%d",
            config.model,
            config.sample_rate,
            self._language,
            max_retries,
        )

        attempt = 0
        while True:
            attempt += 1
            await self._run_once(attempt=attempt)

            # Decide: retry or exit?
            if self._task_failed:
                # Fatal — propagate via existing logic in _run_once.
                # If recoverable+retry is desired in future, classify
                # task_failed status_codes here.
                logger.info(
                    "[SenseTimeSpeechStream] _run exiting on task_failed",
                )
                return

            if self._input_ch_closed:
                logger.info(
                    "[SenseTimeSpeechStream] _run exiting on natural session end "
                    "(input_ch closed; pcm_total_bytes=%d, finals_received=%s)",
                    self._pcm_total_bytes, self._final_received,
                )
                return

            if not self._conn_closed:
                # Unexpected: exited without conn_closed and without natural
                # end. Don't retry — something's wrong with our state machine.
                logger.warning(
                    "[SenseTimeSpeechStream] _run exiting on unknown condition "
                    "(input_done=%s conn_closed=%s); not retrying",
                    self._input_done.is_set(), self._conn_closed,
                )
                return

            # Mid-session WS drop. Retry?
            if attempt > max_retries:
                logger.error(
                    "[SenseTimeSpeechStream] _run: retry budget exhausted "
                    "after %d attempt(s); ending stream "
                    "(framework will close session)",
                    attempt,
                )
                self._emit_error(
                    SenseTimeSTTError(
                        "STT stream retry budget exhausted",
                        recoverable=False,
                    ),
                    recoverable=False,
                )
                return

            wait = backoffs[min(attempt - 1, len(backoffs) - 1)]
            logger.warning(
                "[SenseTimeSpeechStream] _run: mid-stream WS drop on "
                "attempt %d/%d, retrying in %.1fs "
                "(pcm_total_bytes=%d, finals_so_far=%s)",
                attempt, max_retries + 1, wait,
                self._pcm_total_bytes, self._final_received,
            )
            await asyncio.sleep(wait)
            # Force the shared conn to be re-established on next attempt.
            self._stt_ref._conn = None

    async def _run_once(self, *, attempt: int) -> None:
        """Single attempt: connect → task_start → loop → cleanup.

        Cross-attempt state (``_input_done``, ``_input_ch_closed``,
        ``_user_away_received``, ``_final_received``, ``_pcm_total_bytes``)
        is preserved by the caller; per-attempt state (``_task_failed``,
        ``_conn_closed``, ``_exit_event``, etc) is reset here.
        """
        config = self._config

        # Per-attempt state reset
        self._task_failed = False
        self._task_failed_error = None
        self._task_finished_received = False
        self._conn_closed = False
        self._exit_event = asyncio.Event()
        self._task_finished_event = asyncio.Event()
        # _input_done is per-attempt: it tracks "this attempt's _send_loop
        # has stopped". The framework's audio_ch lifecycle is reflected by
        # _input_ch_closed instead.
        self._input_done = asyncio.Event()

        if attempt > 1:
            logger.info(
                "[SenseTimeSpeechStream] _run_once: retry attempt %d", attempt,
            )

        # Acquire stream_lock — at most one stream may use the shared
        # connection at a time. In session-long mode there's typically only
        # one stream per session anyway, but the lock guards against the
        # framework somehow calling stream() twice.
        async with self._stt_ref._stream_lock:
            client = await self._stt_ref._ensure_conn()

            # Clear stale user_away signal from a previous stream/session.
            self._stt_ref._user_away_event.clear()

            # Session-long task: ONE task_start at stream entry. task_finish
            # is sent only in the finally clause when the session ends.
            await client.send_task_start()

            # Bridge callback-style server messages to an asyncio.Queue
            msg_ch: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def on_message(msg: dict[str, Any] | None) -> None:
                await msg_ch.put(msg)

            client._on_message_callback = on_message

            send_task = asyncio.create_task(self._send_loop(client))
            recv_task = asyncio.create_task(self._recv_loop(client, msg_ch))
            # State-sync bridge: watch the plugin-level _user_away_event set
            # by orchestrator on user_state -> "away" (Round 7 G11).
            user_away_watcher = asyncio.create_task(self._watch_user_away())

            try:
                # Pure event-driven exit. No wall-clock timeout — the stream
                # is session-long. Exit conditions (set in _check_exit):
                #   A. _task_failed       — fatal server error
                #   B. _conn_closed       — WS died
                #   C. _user_away_received — orchestrator says user gone
                #   D. _input_done        — framework closed _input_ch
                await self._exit_event.wait()
            except asyncio.CancelledError:
                logger.info("[SenseTimeSpeechStream] _run cancelled")
            except BaseException:
                logger.exception("[SenseTimeSpeechStream] _run unexpected error")
            finally:
                # Stop pushing new audio first.
                self._input_done.set()
                if not send_task.done():
                    send_task.cancel()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.gather(send_task, return_exceptions=True)),
                        timeout=2.0,
                    )
                except (asyncio.TimeoutError, BaseException):
                    pass

                # Cleanup: send task_finish so the server can flush its
                # state. Wait briefly for ack so the conn is fully drained
                # before we move on. Skip on broken-conn paths (task_failed
                # / conn_closed mean the WS is already dead).
                if (
                    not self._task_failed
                    and not self._conn_closed
                    and client.is_connected
                ):
                    try:
                        await client.send_task_finish()
                        try:
                            await asyncio.wait_for(
                                self._task_finished_event.wait(),
                                timeout=5.0,
                            )
                            logger.info(
                                "[SenseTimeSpeechStream] cleanup: "
                                "task_finished received "
                                "(pcm_total_bytes=%d, finals=%s, "
                                "user_was_away=%s)",
                                self._pcm_total_bytes,
                                self._final_received,
                                self._user_away_received,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "[SenseTimeSpeechStream] cleanup: "
                                "task_finished ack timeout (5s); "
                                "forcing close (pcm_total_bytes=%d)",
                                self._pcm_total_bytes,
                            )
                    except Exception as e:
                        logger.warning(
                            "[SenseTimeSpeechStream] cleanup: "
                            "task_finish send failed: %s",
                            e,
                        )

                # Cancel remaining loops.
                for t in (recv_task, user_away_watcher):
                    if not t.done():
                        t.cancel()
                try:
                    await asyncio.gather(
                        send_task, recv_task, user_away_watcher,
                        return_exceptions=True,
                    )
                except BaseException:
                    pass

                client._on_message_callback = None

        # Per-attempt cleanup of shared conn — if the WS died this
        # attempt, clear so _ensure_conn reconnects on the next attempt
        # (or the next stream() call).
        if (
            self._stt_ref._conn is client
            and not client.is_connected
        ):
            self._stt_ref._conn = None

        # Propagate fatal task_failed in-attempt (the retry loop in
        # _run inspects _task_failed and exits cleanly without retry).
        # Only raise if no transcripts have ever been received — else
        # downgrade to a warning, the partial result is still useful.
        if self._task_failed and self._task_failed_error is not None:
            if not self._final_received:
                raise self._task_failed_error from None
            else:
                logger.warning(
                    "[SenseTimeSpeechStream] task_failed received after "
                    "transcripts (pcm_total_bytes=%d) — treating as non-fatal: %s",
                    self._pcm_total_bytes,
                    self._task_failed_error,
                )
                self._task_failed = False
                self._task_failed_error = None

        if config.log_audio_diag:
            logger.info(
                "[SenseTimeSpeechStream] stream_complete (attempt=%d) "
                "pcm_total_bytes=%d final_received=%s conn_closed=%s "
                "input_ch_closed=%s",
                attempt,
                self._pcm_total_bytes,
                self._final_received,
                self._conn_closed,
                self._input_ch_closed,
            )

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def _send_loop(self, client: STTConnection) -> None:
        """Read audio frames from _input_ch, send 100 ms binary chunks.

        Session-long mode: this loop runs for the entire session. On VAD
        ``_FlushSentinel`` we flush the partial audio buffer (so any
        residual sub-100ms tail reaches the server) but do NOT send
        ``task_finish`` — that's reserved for session-end cleanup in
        ``_run``'s finally clause. Server-side VAD splits utterances and
        emits one ``result_final`` each.

        Exits naturally when ``_input_ch`` closes (framework session end)
        or on cancellation (we're being torn down).
        """
        flush_sentinel_type: type = RecognizeStream._FlushSentinel  # type: ignore[attr-defined]

        chunks_sent = 0
        natural_end = False
        try:
            async for item in self._input_ch:  # type: ignore[attr-defined]
                if isinstance(item, flush_sentinel_type):
                    # VAD endpoint: flush partial PCM but keep the task open.
                    # Server VAD will close the utterance and emit result_final.
                    assert self._audio_buf is not None
                    flushed_bytes = 0
                    for chunk in self._audio_buf.flush():
                        data = bytes(chunk.data.tobytes())
                        flushed_bytes += len(data)
                        await client.send_audio_binary(data)
                    if flushed_bytes:
                        logger.debug(
                            "[SenseTimeSpeechStream] flush sentinel — "
                            "flushed %d bytes (utterance boundary; task stays open)",
                            flushed_bytes,
                        )
                    continue

                frame: rtc.AudioFrame = item
                pcm = frame.data.tobytes()
                self._pcm_total_bytes += len(pcm)

                assert self._audio_buf is not None
                for chunk in self._audio_buf.push(pcm):
                    await client.send_audio_binary(bytes(chunk.data.tobytes()))
                    chunks_sent += 1

                if chunks_sent and chunks_sent % 50 == 0:
                    logger.info(
                        "[SenseTimeSpeechStream] _send_loop: chunks=%d total_bytes=%d",
                        chunks_sent,
                        self._pcm_total_bytes,
                    )
            # If we exit the for-loop without an exception, _input_ch
            # closed naturally — framework signaled session end.
            natural_end = True

        except asyncio.CancelledError:
            logger.info(
                "[SenseTimeSpeechStream] _send_loop cancelled (chunks_sent=%d)",
                chunks_sent,
            )
            raise
        except SenseTimeSTTError as e:
            logger.warning("[SenseTimeSpeechStream] _send_loop send failed: %s", e)
            self._conn_closed = True
            self._check_exit()
        except Exception:
            logger.exception("[SenseTimeSpeechStream] _send_loop unexpected error")
        finally:
            if natural_end:
                # Framework closed audio_ch — distinguishes "session end"
                # from "WS died" so the retry layer above doesn't reconnect.
                self._input_ch_closed = True
            self._input_done.set()
            self._check_exit()

    async def _recv_loop(
        self,
        client: STTConnection,
        msg_ch: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        """Pure message-driven receive loop.

        Blocks on ``msg_ch.get()``; ``_process_message`` updates state and
        calls ``_check_exit`` so ``_run`` is woken via ``_exit_event``.
        """
        try:
            while True:
                msg = await msg_ch.get()
                if msg is None:
                    # Connection-closed sentinel
                    logger.debug(
                        "[SenseTimeSpeechStream] _recv_loop: connection closed"
                    )
                    self._conn_closed = True
                    self._check_exit()
                    return
                await self._process_message(msg)
        except asyncio.CancelledError:
            return

    async def _watch_user_away(self) -> None:
        """Round 8 R8.8: informational watcher — does NOT exit the stream.

        Background:
          Round 7 G11 added this as a fast-exit path: when framework's
          ``user_state`` went to "away", we'd kill the stream immediately
          to avoid waiting on the 30 s safety net for noise/echo.

          That made sense in the per-utterance model. R8.1 made the
          stream session-long, which inverted the equation: killing the
          stream on a 15 s silence timer kills the entire session's STT
          (no way to revive — framework's ``_STTPipeline`` doesn't
          recreate streams). The user reappearing 16 s later got no
          STT response — observed in production 2026-05-06.

          Architectural correction: ``user_state == "away"`` is a
          wall-clock heuristic, NOT a "user actually left" signal. It
          fires after 15 s of mutual silence. A user thinking about a
          question for 16 s would also trigger it.

          The session-long stream cost of "keep streaming during away"
          is near zero — server-side VAD on the SenseAudio side handles
          silence/noise without producing transcripts.

        New behavior: log the transition, set the diagnostic flag, but
        do NOT touch ``_exit_event``. Loop forever (until cancelled in
        finally) so transitions away→present can also be logged.
        """
        try:
            event = self._stt_ref._user_away_event
            while True:
                await event.wait()
                self._user_away_received = True
                logger.info(
                    "[SenseTimeSpeechStream] user_state -> away signaled "
                    "(stream stays alive — R8.8 architectural fix); "
                    "pcm_total_bytes=%d final_received=%s",
                    self._pcm_total_bytes,
                    self._final_received,
                )
                # Wait until orchestrator clears the event (user came back)
                # to avoid a tight loop.
                while event.is_set():
                    await asyncio.sleep(0.5)
                logger.info(
                    "[SenseTimeSpeechStream] user_state returned from "
                    "away — stream continuing normally"
                )
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # State machine: _check_exit + _process_message
    # ------------------------------------------------------------------

    def _check_exit(self) -> None:
        """Pure state predicate; sets ``_exit_event`` if any condition holds.

        Per-attempt exit conditions (Round 8 R8.8 — three remain after
        removing user_away):
          A. ``_task_failed``        → fatal server error (retry layer
                                       won't retry; propagates up)
          B. ``_conn_closed``        → WS dead (retry layer may retry)
          C. ``_input_done``         → this attempt's _send_loop stopped
                                       (could be natural session end OR
                                       cancellation of the attempt; the
                                       retry layer in ``_run`` decides
                                       based on ``_input_ch_closed``)

        Removed in R8.8: ``_user_away_received`` was an exit trigger in
        Round 7 G11. In the session-long model (R8.1), exiting on this
        15 s wall-clock signal kills the whole session's STT with no way
        to revive. The flag is now informational only — see
        ``_watch_user_away``.

        ``_task_finished_received`` is NOT an exit trigger; in session-long
        mode it only fires during cleanup in response to our own
        ``send_task_finish``, signalling ``_task_finished_event`` for the
        cleanup waiter.
        """
        if self._exit_event.is_set():
            return

        if self._task_failed:
            self._exit_event.set()
            return
        if self._conn_closed:
            logger.debug(
                "[SenseTimeSpeechStream] exit: connection closed "
                "(pcm_total_bytes=%d)",
                self._pcm_total_bytes,
            )
            self._exit_event.set()
            return
        if self._input_done.is_set():
            # _send_loop stopped this attempt. _run inspects
            # _input_ch_closed afterwards to know whether to retry.
            self._exit_event.set()
            return

    async def _process_message(self, msg: dict[str, Any]) -> None:
        """Route a single server message → update state → check exit."""
        event = msg.get("event")

        if event == EVENT_RESULT_FINAL:
            await self._handle_result_final(msg)
        elif event == EVENT_TASK_FINISHED:
            await self._handle_task_finished()
        elif event == EVENT_TASK_FAILED:
            await self._handle_task_failed(msg)
        # Unknown events are silently ignored for forward compatibility.

        self._check_exit()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_result_final(self, msg: dict[str, Any]) -> None:
        """Handle a ``result_final`` server message → emit FINAL_TRANSCRIPT."""
        data = msg.get(KEY_DATA) or {}
        text = data.get(KEY_TEXT, "")
        is_final = bool(data.get(KEY_IS_FINAL, True))

        if not text:
            return

        # Per protocol, result_final.is_final is always true. Emit as
        # FINAL_TRANSCRIPT (no INTERIM in this protocol).
        event_type = (
            self._SpeechEventType.FINAL_TRANSCRIPT
            if is_final
            else self._SpeechEventType.INTERIM_TRANSCRIPT
        )

        logger.info(
            "[SenseTimeSpeechStream] %s text=%r is_final=%s",
            event_type.name,
            text,
            is_final,
        )

        self._push_event(
            self._SpeechEvent(
                type=event_type,
                alternatives=[
                    self._SpeechData(
                        language=self._language,
                        text=text,
                        start_time=0.0,
                        end_time=0.0,
                        confidence=1.0,
                        words=[],
                    )
                ],
            )
        )
        if is_final:
            self._final_received = True
            # Server-side VAD already segmented this utterance — surface
            # the per-utterance END_OF_SPEECH so the framework can
            # commit the user turn (more accurate than relying solely on
            # client-side VAD endpointing).
            self._push_event(
                self._SpeechEvent(type=self._SpeechEventType.END_OF_SPEECH)
            )

    async def _handle_task_finished(self) -> None:
        """Handle ``task_finished``: cleanup ack (NOT an exit trigger).

        In session-long mode this only fires during ``_run`` finally
        cleanup, after we send ``task_finish``. We signal the event so
        the cleanup waiter can proceed; we do NOT emit END_OF_SPEECH
        (the framework's own VAD handles per-utterance endpointing —
        emitting it here would be redundant and potentially out-of-order).
        """
        logger.info("[SenseTimeSpeechStream] task_finished received (cleanup ack)")
        self._task_finished_received = True
        self._task_finished_event.set()

    async def _handle_task_failed(self, msg: dict[str, Any]) -> None:
        """Handle ``task_failed``: store the error for later propagation."""
        logger.error(
            "[SenseTimeSpeechStream] task_failed raw_msg: %s",
            msg,
        )
        # SenseAudio puts base_resp at the top level for task_failed
        base_resp = msg.get(KEY_BASE_RESP) or (msg.get(KEY_DATA) or {}).get(KEY_BASE_RESP) or {}
        err = (
            base_resp.get(KEY_STATUS_MSG)
            if isinstance(base_resp, dict)
            else str(base_resp)
        )
        if not err:
            err = str(msg.get(KEY_DATA) or msg)
        logger.error("[SenseTimeSpeechStream] task_failed extracted err=%r", err)
        self._task_failed = True
        self._task_failed_error = SenseTimeSTTError(
            f"SenseAudio STT task failed: {err}", recoverable=False
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
