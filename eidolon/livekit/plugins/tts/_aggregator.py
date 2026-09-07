# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sentence aggregator for streaming TTS input.

============================================================================
ARCHITECTURE DECISION (ADR — why not livekit-agents' SentenceTokenizer?)
============================================================================

The framework provides ``livekit.agents.tokenize.SentenceTokenizer`` /
``StreamAdapter`` for sentence segmentation (default: Blingfire). We do
NOT use them because:

1. **StreamAdapter assumes non-streaming TTS** — it wraps a one-shot
   ``synthesize(text)`` plugin into a streaming one by tokenising into
   whole sentences and submitting one at a time. Our TTS providers are
   ALREADY streaming at the token level via ``continue-task``.
   Inserting StreamAdapter would force every sentence to wait for the
   full LLM sentence to complete before TTS starts, undoing the
   streaming benefit.

2. **Blingfire's segmentation is linguistic** (statistically learned),
   not driven by punctuation + min-length + idle-timer. Our heuristic
   is intentionally simpler and tunable; e.g. we flush at soft
   punctuation (",", "，") above a threshold so the user hears the
   first phrase fast, even if Blingfire would still wait for a full
   sentence.

3. **Empirical fit** — Round 8 R8.2 spike showed our 3-trigger
   aggregator drops the per-reply ``continue-task`` count from ~12
   (per-token) to ~3 (per sentence) on real Chinese LLM output, with
   audible smoothness improvement. We didn't measure Blingfire
   side-by-side because it was the wrong tool (per #1).

If a future framework version adds a streaming-friendly tokeniser, this
module should be re-evaluated.

============================================================================
DESIGN
============================================================================

Buffers individual LLM tokens into sentence-sized chunks before
submission to TTS. Round 8 R8.2: production logs showed the LLM
emitting 12 tokens for "Hey there I'm ready to help. What can I do
for you?" — each one became a separate ``continue-task``, and the
TTS synthesizes each ``continue-task`` as an independent batch with
~600 ms gaps between them. End result: choppy "几个字蹦一次" audio.
The aggregator buffers tokens until a flush condition fires, dropping
the per-batch count from ~12 to ~3 (one per sentence).

Flush triggers (any one):
  1. Hard sentence-ending punctuation: ``.!?。！？；;\\n``
  2. Soft punctuation + min length: ``,，、 :：—`` with buffer ≥ ``soft_min_chars``
  3. Buffer reaches ``hard_max_chars`` (force flush, prevents giant batches)
  4. ``idle_ms`` elapsed since last token (LLM stalled — flush what we have)
  5. ``flush()`` called explicitly (input ended / framework FlushSentinel)

This module is **TTS-provider-agnostic**. Both BailianTTS and SenseTimeTTS
use the same aggregation logic — the only difference is the callback
(``send_continue`` vs ``send_task_continue``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("tts.aggregator")


# Hard sentence-ending punctuation: flush immediately after any of these.
# Includes both ASCII (.!?;) and Chinese full-width (。！？；) variants
# plus newline.
_HARD_PUNCT = ".!?;。！？；\n"
# Soft punctuation: flush only when buffer is long enough. Avoids
# splitting "Hi, ..." into "Hi," + "..." (sounds choppier than waiting
# for a real sentence).
_SOFT_PUNCT = "，,、:：—"


class SentenceAggregator:
    """Buffer LLM tokens into sentence-sized chunks for TTS submission.

    Usage::

        async def emit(text: str) -> None:
            await client.send_task_continue(text)

        agg = SentenceAggregator(emit)
        async for token in llm_stream:
            await agg.feed(token)
        await agg.flush()       # flush whatever's left
        await agg.aclose()      # cancel idle timer

    The callback ``on_segment`` is invoked with the aggregated text
    (already stripped of leading/trailing whitespace). Empty strings
    are never emitted.
    """

    def __init__(
        self,
        on_segment: Callable[[str], Awaitable[None]],
        *,
        soft_min_chars: int = 12,
        hard_max_chars: int = 60,
        idle_ms: int = 300,
        first_sentence_soft_min_chars: int | None = None,
        first_sentence_flush_any_punct: bool = False,
    ) -> None:
        if soft_min_chars <= 0 or hard_max_chars <= 0:
            raise ValueError("char thresholds must be positive")
        if soft_min_chars > hard_max_chars:
            raise ValueError("soft_min_chars must be ≤ hard_max_chars")
        if idle_ms <= 0:
            raise ValueError("idle_ms must be positive")
        if first_sentence_soft_min_chars is not None and first_sentence_soft_min_chars <= 0:
            raise ValueError("first_sentence_soft_min_chars must be positive when set")

        self._on_segment = on_segment
        self._soft_min_chars = soft_min_chars
        self._hard_max_chars = hard_max_chars
        self._idle_sec = idle_ms / 1000.0
        # G5 (2026-05-16): first-sentence aggressive flush — when LLM body
        # arrives in fast bursts (sub-second TTFB on fast endpoints), the
        # default 12-char soft_min misses early punct opportunities and the
        # whole reply ends up as one ``explicit`` flush, defeating the
        # streaming TTS design. These two knobs are scoped to the FIRST
        # flush of the stream so subsequent sentences keep clean batching.
        self._first_sentence_soft_min_chars = first_sentence_soft_min_chars
        self._first_sentence_flush_any_punct = first_sentence_flush_any_punct
        self._first_emit_done: bool = False

        self._buf: list[str] = []
        self._buf_len: int = 0
        self._idle_task: Optional[asyncio.Task[None]] = None
        self._closed: bool = False
        # Lock to ensure feed / flush / idle-timer don't race on the buffer.
        self._lock: asyncio.Lock = asyncio.Lock()

    # ── public API ───────────────────────────────────────────────

    async def feed(self, token: str) -> None:
        """Push one token into the buffer; may trigger a flush."""
        if self._closed:
            raise RuntimeError("SentenceAggregator already aclose()d")
        if not token:
            return

        async with self._lock:
            self._buf.append(token)
            self._buf_len += len(token)

            # Decide: flush, partial-flush at punct, or wait?
            decision = self._classify_buffer()

            if decision == "hard_flush":
                await self._flush_locked(reason="hard_punct_or_max")
            elif decision == "soft_flush":
                await self._flush_locked(reason="soft_punct_with_min_len")
            else:
                # Hold and reset idle timer.
                self._reset_idle_timer()

    async def flush(self) -> None:
        """Flush any pending buffer immediately."""
        async with self._lock:
            if self._buf_len > 0:
                await self._flush_locked(reason="explicit")

    async def aclose(self) -> None:
        """Cancel the idle timer and mark closed. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, BaseException):
                pass
        self._idle_task = None

    # ── internals ────────────────────────────────────────────────

    def _classify_buffer(self) -> str:
        """Return ``"hard_flush"``, ``"soft_flush"``, or ``"hold"``."""
        joined = "".join(self._buf)
        # Streaming chunks need not end at punctuation: the SDK's Markdown
        # filter can put a boundary at the start of the next chunk. Keep the
        # existing whole-buffer batching, but recognize boundaries anywhere.
        hard_boundary = any(char in _HARD_PUNCT for char in joined)
        soft_boundary = max(joined.rfind(char) for char in _SOFT_PUNCT)

        # G5 (2026-05-16): first-sentence aggressive mode. Only applies
        # BEFORE the first successful flush of this stream; after that,
        # standard thresholds resume so subsequent sentences batch cleanly.
        if not self._first_emit_done:
            if self._first_sentence_flush_any_punct and (hard_boundary or soft_boundary >= 0):
                return "hard_flush"
            soft_min = (
                self._first_sentence_soft_min_chars
                if self._first_sentence_soft_min_chars is not None
                else self._soft_min_chars
            )
        else:
            soft_min = self._soft_min_chars

        if hard_boundary:
            return "hard_flush"
        if self._buf_len >= self._hard_max_chars:
            return "hard_flush"
        if soft_boundary + 1 >= soft_min:
            return "soft_flush"
        return "hold"

    async def _flush_locked(self, *, reason: str) -> None:
        """Emit the buffer (caller already holds ``self._lock``)."""
        text = "".join(self._buf).strip()
        self._buf.clear()
        self._buf_len = 0
        # Cancel any pending idle timer — buffer is empty now.
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

        if not text:
            return

        logger.debug(
            "[SentenceAggregator] flush reason=%s text=%r", reason, text,
        )
        # G5 (2026-05-16): mark first emit done so subsequent classifications
        # revert to standard thresholds. Done BEFORE the callback so even if
        # the callback raises (F2 propagation), we don't replay the aggressive
        # mode for the retry — it had its chance.
        self._first_emit_done = True
        # F2 (2026-05-16): propagate on_segment failures instead of swallowing.
        # Previously this except-Exception caught BailianTTSError("WebSocket
        # disconnected") and silently dropped the segment text — the framework
        # had no way to know the TTS turn failed, so the user heard nothing
        # for the rest of the reply. By raising, the failure surfaces to the
        # SynthesizeStream's _task_failed_error, which raises APIError, which
        # at minimum gets logged at ERROR level and lets the framework cancel
        # the doomed audio segment cleanly.
        await self._on_segment(text)

    def _reset_idle_timer(self) -> None:
        """Restart the idle timer; schedules a flush if no new feed within ``idle_ms``."""
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_flush_after())

    async def _idle_flush_after(self) -> None:
        try:
            await asyncio.sleep(self._idle_sec)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._buf_len > 0 and not self._closed:
                await self._flush_locked(reason="idle_timer")
