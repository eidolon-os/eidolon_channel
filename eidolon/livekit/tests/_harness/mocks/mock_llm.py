# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MockLLM — deterministic, in-process LLM stand-in.

Conforms to ``livekit.agents.llm.LLM``. Yields ``ChatChunk`` events
identical in shape to what real providers emit, so downstream
plumbing (LlmStage, AgentSession's chat orchestration) can't tell
the difference.

Three modes:
  * **scripted**: list of ``ScriptedReply`` matched against the latest
    user message (substring or callable predicate). Most-specific match
    wins; falls back to a default reply.
  * **echo**: replies with the latest user text verbatim. Useful for
    "did the message reach the LLM" scenarios without caring about
    content.
  * **error**: every ``chat()`` raises a configured ``Exception``.
    Used to test pipeline error paths.

Token-stream pacing: replies are yielded one character at a time by
default (mimicking streaming providers); ``chunk_size`` and
``chunk_delay_ms`` configure this. Non-zero ``chunk_delay_ms`` is the
only place this mock will await — keep it 0 in unit tests, raise it
in scenarios that exercise interrupt-mid-LLM-stream.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Optional

from livekit.agents import APIConnectOptions
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    CompletionUsage,
    LLMStream,
)
from livekit.agents.llm.tool_context import Tool
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

# Type alias for clarity — predicate takes the latest user text and
# returns True on match.
_Predicate = Callable[[str], bool]


@dataclass
class ScriptedReply:
    """One scripted (trigger → reply) entry."""

    reply: str
    """The text the LLM will emit when this entry matches."""

    when: str | _Predicate | None = None
    """Match the latest user message:
      * ``str``  — case-insensitive substring match
      * callable — custom predicate
      * ``None`` — wildcard (used as default fallback)
    """

    raises: Optional[Exception] = None
    """If set, this Exception is raised inside ``_run`` instead of
    yielding the reply (simulates LLM API failure mid-stream)."""

    delay_first_chunk_ms: int = 0
    """Sleep this long before yielding the first chunk (simulate slow
    first-token latency / TTFT)."""

    def matches(self, latest_user_text: str) -> bool:
        if self.when is None:
            return True
        if callable(self.when):
            return bool(self.when(latest_user_text))
        # str → case-insensitive substring
        return self.when.lower() in latest_user_text.lower()


def _extract_latest_user_text(chat_ctx: ChatContext) -> str:
    """Return the most recent user-role message's text, or "".

    livekit-agents ChatContext stores ChatMessage with ``role`` and
    ``text_content`` (joined str). We walk in reverse for O(turns)
    worst case but typically O(1) — the user message is at or near
    the end.
    """
    for item in reversed(chat_ctx.items):
        # ChatItem can be ChatMessage / FunctionCall / etc. We only
        # care about user-role chat messages.
        if getattr(item, "type", None) != "message":
            continue
        if getattr(item, "role", None) != "user":
            continue
        text = getattr(item, "text_content", None)
        if text:
            return text
    return ""


class MockLLM(LLM):
    """Deterministic LLM mock for scenario tests.

    Examples:
        >>> # Scripted patterns
        >>> llm = MockLLM.scripted([
        ...     ScriptedReply(when="你好", reply="你好！很高兴见到你。"),
        ...     ScriptedReply(when=None, reply="我没听清楚。"),  # default
        ... ])

        >>> # Echo mode — repeats user input verbatim
        >>> llm = MockLLM.echo()

        >>> # Always-error mode
        >>> llm = MockLLM.errors_with(RuntimeError("upstream down"))
    """

    def __init__(
        self,
        *,
        scripts: Iterable[ScriptedReply] | None = None,
        mode: Literal["scripted", "echo"] = "scripted",
        default_reply: str = "好的。",
        chunk_size: int = 1,
        chunk_delay_ms: int = 0,
        always_raises: Optional[Exception] = None,
    ) -> None:
        super().__init__()
        self._scripts: list[ScriptedReply] = list(scripts or [])
        self._mode = mode
        self._default_reply = default_reply
        self._chunk_size = max(1, chunk_size)
        self._chunk_delay_ms = max(0, chunk_delay_ms)
        self._always_raises = always_raises
        # Recorded prompts for assertions (test-only).
        self.prompts: list[ChatContext] = []
        self.call_count: int = 0

    # ─────────────────────────────────────────────── factories

    @classmethod
    def scripted(
        cls,
        replies: Iterable[ScriptedReply | tuple[str, str]],
        *,
        default_reply: str = "好的。",
        **kwargs: Any,
    ) -> "MockLLM":
        """Build from a list of (when, reply) pairs or ScriptedReply.

        Tuple shorthand: ``[("你好", "你好啊"), ("再见", "拜拜")]``.
        """
        normalized: list[ScriptedReply] = []
        for r in replies:
            if isinstance(r, ScriptedReply):
                normalized.append(r)
            elif isinstance(r, tuple) and len(r) == 2:
                normalized.append(ScriptedReply(when=r[0], reply=r[1]))
            else:
                raise TypeError(
                    f"scripted entry must be ScriptedReply or (str, str), got {type(r)}"
                )
        return cls(scripts=normalized, default_reply=default_reply, **kwargs)

    @classmethod
    def echo(cls, *, prefix: str = "", **kwargs: Any) -> "MockLLM":
        """Echo the user's latest message verbatim (optional ``prefix``)."""
        return cls(mode="echo", default_reply=prefix, **kwargs)

    @classmethod
    def errors_with(cls, exc: Exception, **kwargs: Any) -> "MockLLM":
        """Every chat() raises ``exc`` from inside the stream."""
        return cls(always_raises=exc, **kwargs)

    # ─────────────────────────────────────────────── protocol

    @property
    def model(self) -> str:
        return "mock-llm"

    @property
    def provider(self) -> str:
        return "eidolon-test"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Any] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        # Snapshot the chat context for assertions. ChatContext is
        # mutable in livekit-agents, so save the items list directly.
        self.prompts.append(ChatContext(list(chat_ctx.items)))
        self.call_count += 1

        reply_text = self._resolve_reply(chat_ctx)
        first_chunk_delay_ms = 0
        raise_after_resolve: Exception | None = self._always_raises

        # Look up scripted entry (if any) for richer behavior.
        if self._mode == "scripted" and not raise_after_resolve:
            user_text = _extract_latest_user_text(chat_ctx)
            for entry in self._scripts:
                if entry.matches(user_text):
                    raise_after_resolve = entry.raises
                    first_chunk_delay_ms = entry.delay_first_chunk_ms
                    break

        return _MockLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            reply_text=reply_text,
            chunk_size=self._chunk_size,
            chunk_delay_ms=self._chunk_delay_ms,
            first_chunk_delay_ms=first_chunk_delay_ms,
            raises=raise_after_resolve,
        )

    # ─────────────────────────────────────────────── internals

    def _resolve_reply(self, chat_ctx: ChatContext) -> str:
        """Compute the reply text based on mode + scripts."""
        user_text = _extract_latest_user_text(chat_ctx)
        if self._mode == "echo":
            return f"{self._default_reply}{user_text}" if self._default_reply else user_text
        # scripted
        for entry in self._scripts:
            if entry.matches(user_text):
                return entry.reply
        return self._default_reply


class _MockLLMStream(LLMStream):
    """Internal streaming impl. Yields the reply char-by-char (or in
    ``chunk_size`` slices) wrapped in proper ``ChatChunk`` events."""

    def __init__(
        self,
        *,
        llm: LLM,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        reply_text: str,
        chunk_size: int,
        chunk_delay_ms: int,
        first_chunk_delay_ms: int,
        raises: Optional[Exception],
    ) -> None:
        self._reply_text = reply_text
        self._chunk_size = chunk_size
        self._chunk_delay_ms = chunk_delay_ms
        self._first_chunk_delay_ms = first_chunk_delay_ms
        self._raises = raises
        self._request_id = f"mock-{uuid.uuid4().hex[:8]}"
        super().__init__(
            llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )

    async def _run(self) -> None:
        if self._raises is not None:
            raise self._raises
        if self._first_chunk_delay_ms:
            await asyncio.sleep(self._first_chunk_delay_ms / 1000.0)
        text = self._reply_text
        if not text:
            # Even an empty reply yields a single chunk with empty
            # content + final usage so the stream closes cleanly.
            self._event_ch.send_nowait(
                ChatChunk(
                    id=self._request_id,
                    delta=ChoiceDelta(role="assistant", content=""),
                )
            )
        else:
            for i in range(0, len(text), self._chunk_size):
                slice_ = text[i : i + self._chunk_size]
                self._event_ch.send_nowait(
                    ChatChunk(
                        id=self._request_id,
                        delta=ChoiceDelta(role="assistant", content=slice_),
                    )
                )
                if self._chunk_delay_ms:
                    await asyncio.sleep(self._chunk_delay_ms / 1000.0)

        # Final usage chunk (livekit-agents convention — providers emit
        # usage at end of stream).
        self._event_ch.send_nowait(
            ChatChunk(
                id=self._request_id,
                usage=CompletionUsage(
                    completion_tokens=len(text),
                    prompt_tokens=0,
                    total_tokens=len(text),
                ),
            )
        )
