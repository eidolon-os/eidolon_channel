# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""STT transcript gate — cross-turn contamination filter.

============================================================================
WHY (G23, 2026-05-18)
============================================================================

LiveKit Agents' ``audio_recognition.AudioRecognition`` accumulates STT
events into a single ``user_turn`` until ``commit_user_turn`` fires. Two
specific lines in ``audio_recognition.py`` cause cross-turn contamination
when an STT plugin uses a single long-lived stream that spans multiple
user utterances (Bailian FunASR, SenseTime, Azure realtime, Google
streaming all behave this way):

  * line 715-730 — at commit time, any pending INTERIM transcript is
    appended onto the accumulated FINAL transcript before being sent to
    the LLM. If the user just started a new sentence, that sentence's
    first interim ("你有") gets glued onto the previous sentence's
    final ("…睡不着。") yielding ``"…睡不着。 你有"``. The LLM sees
    garbled input and either replies nonsensically or wastes tokens on
    a quickly-cancelled response.

  * line 839 — every FINAL event is concatenated onto ``_audio_transcript``.
    If two FINALs arrive before commit (rare with Bailian since VAD-end
    typically intervenes; more likely with SenseTime which only emits
    FINALs, no INTERIMs), they're merged into a single user message.

This module decorates any ``lk_stt.STT`` plugin with a time-window gate:
within ``suppress_window_ms`` of a FINAL event, subsequent INTERIM/FINAL
events are dropped before reaching the framework. Subsequent transcripts
for the new utterance come through normally after the window expires
(and the next VAD-end triggers a fresh ``commit_user_turn`` for them).

============================================================================
WHY THIS IS SAFE (i.e., what's lost?)
============================================================================

Bailian/SenseTime use cumulative interim transcripts — each interim
contains the FULL current sentence, not a delta. So if we drop the first
interim "你有" within the window, the next interim 100-200ms later will
contain "你有什么" anyway. Nothing semantic is lost — at most, the user's
on-screen transcript is delayed by ``suppress_window_ms`` (200 ms by
default) and the LLM call for the second utterance fires at the next
VAD-end as expected.

============================================================================
WHY DECORATE INSTEAD OF MODIFY FRAMEWORK
============================================================================

The framework's behaviour at line 715/839 has a legitimate use-case — STTs
that emit interim transcripts WITHOUT eventual finals (slow / overloaded
servers) need the "promote last interim to final" path to function.
Patching framework would break that. By decorating at the STT level
upstream of framework, we keep the framework's logic intact while
shielding it from cross-utterance interim leakage.

============================================================================
INTEGRATION
============================================================================

See ``eidolon/livekit/agent/factory.py`` for wiring. The gate is opt-in
via ``EIDOLON_STT_TRANSCRIPT_GATE_ENABLED=true``; default off until the
real-world regression confirms ``suppressed_count`` distribution is
healthy (expected: 0 per typical session, 1 per "two sentences with
short pause" session).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from livekit.agents import stt as lk_stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("plugins.stt.transcript_gate")


class STTTranscriptGate(lk_stt.STT):
    """Decorator over an ``lk_stt.STT`` plugin that filters cross-turn
    transcript events.

    The gate's contract:

      * All non-transcript events pass through (START_OF_SPEECH,
        END_OF_SPEECH, RECOGNITION_USAGE).
      * INTERIM_TRANSCRIPT and FINAL_TRANSCRIPT events are dropped if
        emitted within ``suppress_window_ms`` of a prior FINAL_TRANSCRIPT.
      * The first FINAL of the stream always passes through.
      * ``push_frame`` / ``flush`` / ``end_input`` are forwarded to the
        inner plugin unchanged.
      * Metadata (``label``, ``model``, ``provider``, ``capabilities``)
        is forwarded from the inner plugin.

    Args:
        inner: The concrete STT plugin to decorate (BailianFunASRSTT,
            SenseTimeSTT, etc.). MUST be a subclass of ``lk_stt.STT``.
        suppress_window_ms: How long after a FINAL event subsequent
            transcript events are dropped. 200 ms is the sweet spot for
            Bailian FunASR (observed cross-utterance first-INTERIM
            latency ~128 ms) and SenseTime (similar). Set to 0 to
            disable suppression (equivalent to bypassing the gate).
    """

    def __init__(
        self,
        inner: lk_stt.STT,
        *,
        suppress_window_ms: int = 200,
    ) -> None:
        # Forward inner's capabilities so the framework sees the same
        # streaming/interim support as the underlying plugin.
        super().__init__(capabilities=inner.capabilities)
        self._inner = inner
        self._suppress_window_ms = max(0, int(suppress_window_ms))
        # Override the auto-generated label so log lines clearly show
        # the decoration layer.
        self._label = f"STTTranscriptGate({inner.label})"

        logger.info(
            "[STTTranscriptGate] wrapped %s (suppress_window=%dms)",
            inner.label,
            self._suppress_window_ms,
        )

    # ------------------------------------------------------------------
    # Metadata forwarding (decorator pattern)
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    @property
    def inner(self) -> lk_stt.STT:
        """Expose the wrapped plugin for tests / introspection / metrics.
        Production code should never need this."""
        return self._inner

    @property
    def suppress_window_ms(self) -> int:
        return self._suppress_window_ms

    # ------------------------------------------------------------------
    # Plugin lifecycle (warmup / shutdown) — forward if present
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Forward to inner if it supports warmup. SttStage's warmup path
        already does ``hasattr(self._stt, "warmup")`` so we provide the
        method unconditionally to keep that check happy."""
        if hasattr(self._inner, "warmup"):
            await self._inner.warmup()  # type: ignore[attr-defined]

    async def shutdown(self) -> None:
        if hasattr(self._inner, "shutdown"):
            await self._inner.shutdown()  # type: ignore[attr-defined]

    async def aclose(self) -> None:
        """Close both ourselves and the inner plugin."""
        try:
            await super().aclose()
        finally:
            await self._inner.aclose()

    # ------------------------------------------------------------------
    # Pass-through methods used by some downstream code
    # ------------------------------------------------------------------

    def notify_vad_state(self, probability: float, rms: float) -> None:
        """G16 bridge — StreamingPipeline calls this on the STT plugin
        to feed per-frame VAD state into the STT VAD gate. Forward to
        inner if it exposes the hook (Bailian does, SenseTime doesn't)."""
        if hasattr(self._inner, "notify_vad_state"):
            self._inner.notify_vad_state(probability, rms)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Core STT abstract methods
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        """Batch recognize is single-shot — no cross-turn issue to filter.
        Delegate directly to inner."""
        return await self._inner._recognize_impl(
            buffer, language=language, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_FilteringRecognizeStream":
        """Construct the inner stream and wrap it with our filter."""
        # Accept either canonical kwargs (language/conn_options) or the
        # plugin-specific NOT_GIVEN sentinel. Forward whichever the inner
        # plugin's signature accepts.
        # NOTE: Bailian / SenseTime accept ``language=None`` and
        # ``conn_options=None`` rather than the NOT_GIVEN sentinel; pass
        # None when NOT_GIVEN to match.
        inner_kwargs: dict = {}
        if language is not NOT_GIVEN:
            inner_kwargs["language"] = language
        if conn_options is not DEFAULT_API_CONNECT_OPTIONS:
            inner_kwargs["conn_options"] = conn_options

        inner_stream = self._inner.stream(**inner_kwargs)
        return _FilteringRecognizeStream(
            stt=self,
            inner=inner_stream,
            suppress_window_ms=self._suppress_window_ms,
        )


class _FilteringRecognizeStream(lk_stt.RecognizeStream):
    """The stream object handed to framework's audio_recognition.

    Composed of two cooperating coroutines (both run from ``_run``):

      1. ``_forward_input``: receives audio frames + flush sentinels from
         the OUTER ``_input_ch`` (where framework pushes audio) and
         forwards them to the INNER stream via ``push_frame`` / ``flush``.

      2. The main ``_run`` body: iterates events emitted by the INNER
         stream, applies the time-window filter, re-emits passing events
         to the OUTER ``_event_ch`` for the framework to consume.

    The two coroutines decouple audio input flow from transcript event
    flow, mirroring the base ``RecognizeStream``'s own bidirectional
    pattern.
    """

    def __init__(
        self,
        *,
        stt: STTTranscriptGate,
        inner: lk_stt.RecognizeStream,
        suppress_window_ms: int,
    ) -> None:
        # Use the inner stream's conn_options so retry behaviour matches.
        super().__init__(stt=stt, conn_options=inner._conn_options)
        self._inner: lk_stt.RecognizeStream = inner
        self._suppress_window_sec: float = max(0, suppress_window_ms) / 1000.0

        # Time of the last FINAL_TRANSCRIPT that was PASSED through.
        # Used to compute whether subsequent transcript events fall
        # inside the suppression window.
        self._last_final_pass_time: float | None = None

        # Diagnostic counter — incremented every time we drop an event.
        # Exposed via ``suppressed_count`` for tests and observability.
        self._suppressed_count: int = 0

    @property
    def suppressed_count(self) -> int:
        """Number of INTERIM/FINAL events dropped during this stream's
        lifetime. Production target: 0–1 per typical conversation; high
        values (≥3) indicate the suppression window may be too large
        or the user habitually pauses 100-200 ms between sentences."""
        return self._suppressed_count

    @property
    def last_final_pass_time(self) -> float | None:
        return self._last_final_pass_time

    # ------------------------------------------------------------------
    # Abstract method implementation — runs as ``self._task`` in base
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        forward_task = asyncio.create_task(
            self._forward_input(), name="STTTranscriptGate._forward_input"
        )

        try:
            async for ev in self._inner:
                if not self._should_pass(ev):
                    self._suppressed_count += 1
                    text = ""
                    if ev.alternatives:
                        text = ev.alternatives[0].text
                    logger.debug(
                        "[STTTranscriptGate] suppressed %s within %.0fms of FINAL: %r",
                        ev.type.name,
                        self._suppress_window_sec * 1000,
                        text[:60],
                    )
                    continue

                # Track FINAL events that pass — these reset the
                # suppression window's reference point.
                if ev.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                    self._last_final_pass_time = time.monotonic()

                self._event_ch.send_nowait(ev)
        finally:
            forward_task.cancel()
            try:
                await asyncio.gather(forward_task, return_exceptions=True)
            except Exception:
                pass
            # Inner may already be closing if we exited because inner's
            # events stream ended. aclose is idempotent in the base
            # class so a double-close is safe.
            try:
                await self._inner.aclose()
            except Exception:
                logger.debug(
                    "[STTTranscriptGate] inner.aclose() raised during cleanup",
                    exc_info=True,
                )

    async def _forward_input(self) -> None:
        """Pump frames + flush sentinels from outer _input_ch to inner."""
        try:
            async for item in self._input_ch:
                if isinstance(item, lk_stt.RecognizeStream._FlushSentinel):
                    # Convert our sentinel into inner's flush() call —
                    # we can't push our sentinel into inner's _input_ch
                    # because the two channels have distinct sentinel
                    # types (compared by identity inside base class).
                    self._inner.flush()
                else:
                    self._inner.push_frame(item)
        except asyncio.CancelledError:
            raise
        finally:
            # Signal end-of-input to inner so its _run loop can finalise
            # any in-flight transcripts. Safe even if inner.flush()
            # already saw a sentinel.
            try:
                self._inner.end_input()
            except RuntimeError:
                # Already ended — benign.
                pass

    # ------------------------------------------------------------------
    # Filter logic
    # ------------------------------------------------------------------

    def _should_pass(self, ev: lk_stt.SpeechEvent) -> bool:
        # All non-transcript events pass unconditionally:
        # START_OF_SPEECH, END_OF_SPEECH, RECOGNITION_USAGE, etc.
        # These are control events that the framework needs for state
        # machine bookkeeping.
        if ev.type not in (
            lk_stt.SpeechEventType.INTERIM_TRANSCRIPT,
            lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
        ):
            return True

        # Suppression window disabled (window_ms=0) → never suppress.
        if self._suppress_window_sec <= 0:
            return True

        # First-ever transcript — no prior FINAL to anchor against.
        if self._last_final_pass_time is None:
            return True

        elapsed = time.monotonic() - self._last_final_pass_time
        return elapsed >= self._suppress_window_sec


__all__ = ["STTTranscriptGate"]
