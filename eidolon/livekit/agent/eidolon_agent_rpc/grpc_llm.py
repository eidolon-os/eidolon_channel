"""LiveKit ``LLM`` implementation backed by ``EidolonAgent.Chat`` (gRPC).

Each :class:`EidolonAgentGrpcLlm` instance is scoped to a single LiveKit job and
holds one :class:`EidolonAgentSession` (one long-lived bidi stream shared by
every turn). ``chat()`` opens a new logical turn on the same stream and
forwards ``DELTA`` events to the LiveKit pipeline as ``ChatChunk`` deltas.

Cancellation (e.g. user barge-in) writes a ``CancelTurn`` on the same stream
without tearing it down.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError
from livekit.agents.llm import ChatContext, ToolChoice
from livekit.agents.llm.chat_context import ChatMessage
from livekit.agents.llm.tool_context import Tool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from eidolon.livekit.agent.eidolon_agent_rpc.session import (
    EidolonAgentSession,
    TurnError,
)

logger = logging.getLogger("eidolon_agent_rpc.grpc_llm")


def _last_user_text(chat_ctx: ChatContext) -> str:
    """Pull the most recent user utterance out of the LiveKit ChatContext."""
    for item in reversed(chat_ctx.items):
        if isinstance(item, ChatMessage) and item.role == "user":
            text = item.text_content
            if text:
                return text
            parts = [c for c in item.content if isinstance(c, str)]
            return "\n".join(parts) if parts else ""
    return ""


class EidolonAgentGrpcLlm(llm.LLM):
    """Routes ``LLM.chat`` to ``EidolonAgent.Chat`` over gRPC."""

    def __init__(
        self,
        *,
        target: str,
        device_token: str,
        conversation_id: str,
        display_model: str = "eidolon_agent",
    ) -> None:
        super().__init__()
        if not target.strip():
            raise ValueError("EidolonAgentGrpcLlm: target must be non-empty")
        if not device_token.strip():
            raise ValueError(
                "EidolonAgentGrpcLlm: device_token is required. "
                "Run scripts/provision_eidolon_token.py to obtain one and set "
                "REMOTE_AGENT_RPC_DEVICE_TOKEN in your .env."
            )
        self._target = target.strip()
        self._device_token = device_token.strip()
        self._conversation_id = conversation_id
        self._display_model = display_model
        self._session: EidolonAgentSession | None = None
        self._session_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._display_model

    @property
    def provider(self) -> str:
        return "eidolon_agent_rpc"

    async def _get_session(self) -> EidolonAgentSession:
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    self._session = EidolonAgentSession(
                        target=self._target,
                        device_token=self._device_token,
                    )
        return self._session

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        if tools:
            logger.debug(
                "[EidolonAgentGrpcLlm] tools are not forwarded over eidolon.agent.v1"
            )
        return EidolonAgentGrpcLlmStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None


class EidolonAgentGrpcLlmStream(llm.LLMStream):
    async def _run(self) -> None:
        llm_v: EidolonAgentGrpcLlm = self._llm  # type: ignore[assignment]
        session = await llm_v._get_session()
        user_text = _last_user_text(self._chat_ctx)
        turn_id, deltas = await session.start_turn(
            text=user_text,
            conversation_id=llm_v._conversation_id,
        )
        req_id = f"eidolon-{turn_id}"
        try:
            async for chunk in deltas:
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=req_id,
                        delta=llm.ChoiceDelta(content=chunk),
                    )
                )
        except asyncio.CancelledError:
            # Barge-in or job teardown: tell the brain to stop generating
            # without closing the underlying bidi stream. Fire-and-forget so
            # the cancel write survives the current task's cancellation.
            asyncio.create_task(
                session.cancel_turn(turn_id),
                name=f"eidolon-cancel-{turn_id}",
            )
            raise
        except TurnError as exc:
            raise APIConnectionError(
                f"eidolon_agent error {exc.code}: {exc}",
                retryable=not exc.fatal,
            ) from exc
        except Exception as exc:
            raise APIConnectionError(
                f"eidolon_agent stream failed: {exc}",
                retryable=True,
            ) from exc
