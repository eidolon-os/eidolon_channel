"""LiveKit ``LLM`` implementation backed by ``RemoteAgent.Session`` (gRPC).

Each ``chat()`` opens a short-lived bidi stream: ``ClientConfig`` + ``UserTurn``,
then consumes ``AssistantDelta`` / ``AssistantFinal`` / ``ServerError``.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import grpc.aio
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

logger = logging.getLogger("remote_agent_rpc.grpc_llm")

_PROTOCOL_VERSION = 1


def _last_user_text(chat_ctx: ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if isinstance(item, ChatMessage) and item.role == "user":
            text = item.text_content
            if text:
                return text
            parts = [c for c in item.content if isinstance(c, str)]
            return "\n".join(parts) if parts else ""
    return ""


class RemoteAgentGrpcLlm(llm.LLM):
    """Routes ``LLM.chat`` to a remote agent over gRPC (UDS or TCP)."""

    def __init__(
        self,
        *,
        target: str,
        session_id: str,
        locale: str = "zh",
        display_model: str = "remote_agent_rpc",
    ) -> None:
        super().__init__()
        self._target = target.strip()
        self._session_id = session_id
        self._locale = locale
        self._display_model = display_model
        self._turn_ids = itertools.count(1)

    @property
    def model(self) -> str:
        return self._display_model

    @property
    def provider(self) -> str:
        return "remote_agent_rpc"

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
                "[RemoteAgentGrpcLlm] tools are not forwarded over remote_agent_rpc v1"
            )
        return RemoteAgentGrpcLlmStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        return


class RemoteAgentGrpcLlmStream(llm.LLMStream):
    async def _run(self) -> None:
        from eidolon.channel.livekit.agent.remote_agent_rpc.v1.grpc_gen import (
            remote_agent_rpc_pb2 as pb2,
        )
        from eidolon.channel.livekit.agent.remote_agent_rpc.v1.grpc_gen import (
            remote_agent_rpc_pb2_grpc as pb2_grpc,
        )

        turn_id = next(self._llm._turn_ids)  # type: ignore[attr-defined]
        llm_v: RemoteAgentGrpcLlm = self._llm  # type: ignore[assignment]
        req_id = f"rar-{turn_id}"

        channel = grpc.aio.insecure_channel(llm_v._target)
        try:
            stub = pb2_grpc.RemoteAgentStub(channel)
            call = stub.Session()
            env = pb2.Envelope(
                protocol_version=_PROTOCOL_VERSION,
                session_id=llm_v._session_id,
                turn_id=turn_id,
                trace_id="",
            )
            await call.write(
                pb2.ClientMessage(
                    envelope=env,
                    config=pb2.ClientConfig(locale=llm_v._locale),
                )
            )
            user_text = _last_user_text(self._chat_ctx)
            await call.write(
                pb2.ClientMessage(
                    envelope=env,
                    user_turn=pb2.UserTurn(user_text=user_text, is_final=True),
                )
            )
            await call.done_writing()

            saw_delta = False
            while True:
                msg = await call.read()
                if msg is grpc.aio.EOF:
                    break
                which = msg.WhichOneof("payload")
                if which == "assistant_delta" and msg.assistant_delta.text:
                    saw_delta = True
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=req_id,
                            delta=llm.ChoiceDelta(content=msg.assistant_delta.text),
                        )
                    )
                elif which == "assistant_final":
                    # Avoid duplicating text already streamed as deltas.
                    if msg.assistant_final.full_text and not saw_delta:
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=req_id,
                                delta=llm.ChoiceDelta(content=msg.assistant_final.full_text),
                            )
                        )
                elif which == "error":
                    err = msg.error
                    raise APIConnectionError(
                        f"remote_agent_rpc error {err.code}: {err.message}",
                        retryable=not err.fatal,
                    ) from None
                elif which == "pong":
                    continue
        finally:
            await channel.close()
