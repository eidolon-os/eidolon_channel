"""Language Model (LLM) stage for the voice pipeline.

Wraps :class:`livekit.agents.llm.LLM` to provide both
non-streaming (manual mode) and streaming (streaming mode) generation.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator

from livekit.agents.types import APIConnectOptions

logger = logging.getLogger("pipeline.llm")


@dataclass
class LlmInput:
    """Input to the LLM stage."""

    text: str


@dataclass
class LlmOutput:
    """Output from the LLM stage."""

    text: str


@dataclass
class LlmParams:
    """Parameters for the LLM stage."""

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


class LlmStage(ABC):
    """Abstract base for the LLM stage."""

    @abstractmethod
    async def chat(
        self,
        input: LlmInput,
        conn_options: "APIConnectOptions | None" = None,
    ) -> LlmOutput:
        """Non-streaming chat — used in manual mode."""

    @abstractmethod
    async def generate(
        self,
        input: LlmInput,
        conn_options: "APIConnectOptions | None" = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming token generation — used in streaming mode.

        Yields tokens as they are generated. The caller must consume
        the generator fully or cancel the awaiting task to stop generation.
        """


class LivekitLlmStage(LlmStage):
    """LLM stage wrapping a :class:`livekit.agents.llm.LLM` instance.

    In manual mode, uses ``chat()`` for one-shot response.
    In streaming mode, uses the LLM's streaming API and dispatches tokens
    to the EOT detector and TTS simultaneously.
    """

    def __init__(
        self,
        llm: "lk_llm.LLM",
        params: LlmParams | None = None,
        fnc_ctx: "lk_llm.FunctionContext | None" = None,
    ) -> None:
        from livekit.agents import llm as lk_llm
        from livekit.agents.llm import ChatContext

        self._llm = llm
        self._params = params or LlmParams()
        self._fnc_ctx = fnc_ctx
        self._chat_ctx = ChatContext()
        logger.info(
            "[LivekitLlmStage] initialized with llm=%s", type(llm).__name__
        )

    @property
    def llm(self) -> "lk_llm.LLM":
        """The underlying LiveKit LLM instance."""
        return self._llm

    async def chat(
        self,
        input: LlmInput,
        conn_options: "APIConnectOptions | None" = None,
    ) -> LlmOutput:
        """Non-streaming chat for manual mode."""
        from livekit.agents import llm as lk_llm
        from livekit.agents.llm import ChatContext, ChatMessage

        logger.debug("[LivekitLlmStage] chat() input=%r", input.text[:100])

        # Add user message to context
        self._chat_ctx.add_message(role="user", content=[input.text])

        stream = self._llm.chat(
            chat_ctx=self._chat_ctx,
            tools=self._fnc_ctx.tools if self._fnc_ctx else None,
            conn_options=conn_options or APIConnectOptions(),
        )

        full_text_parts: list[str] = []
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                full_text_parts.append(chunk.delta.content)

        full_text = "".join(full_text_parts)

        # Add assistant message to context
        self._chat_ctx.add_message(role="assistant", content=[full_text])

        logger.debug("[LivekitLlmStage] chat() output=%r", full_text[:100])
        return LlmOutput(text=full_text)

    async def generate(
        self,
        input: LlmInput,
        conn_options: "APIConnectOptions | None" = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming token generation for streaming mode."""
        from livekit.agents import llm as lk_llm
        from livekit.agents.llm import ChatContext, ChatMessage

        logger.debug("[LivekitLlmStage] generate() input=%r", input.text[:100])

        # Build a fresh chat context for this turn
        turn_ctx = ChatContext()
        for msg in self._chat_ctx.items:
            turn_ctx.add_message(role=msg.role, content=msg.content)
        turn_ctx.add_message(role="user", content=[input.text])

        stream = self._llm.chat(
            chat_ctx=turn_ctx,
            tools=self._fnc_ctx.tools if self._fnc_ctx else None,
            conn_options=conn_options or APIConnectOptions(),
        )

        full_text_parts: list[str] = []
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                token = chunk.delta.content
                full_text_parts.append(token)
                yield token

        full_text = "".join(full_text_parts)
        self._chat_ctx.add_message(role="assistant", content=[full_text])
        logger.debug("[LivekitLlmStage] generate() done, total=%d chars", len(full_text))

    def clear_history(self) -> None:
        """Clear the conversation history."""
        from livekit.agents.llm import ChatContext

        self._chat_ctx = ChatContext()
