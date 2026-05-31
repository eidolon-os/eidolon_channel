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
import time
from typing import Any, Callable

from livekit.agents import llm
from livekit.agents._exceptions import APIConnectionError, APIStatusError
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
    CitationPayload,
    DeltaPayload,
    EidolonAgentSession,
    HandoffPayload,
    StatePayload,
    TlsConfig,
    ToolCallPayload,
    TurnError,
    UsagePayload,
    _build_channel_credentials,
)

logger = logging.getLogger("eidolon_agent_rpc.grpc_llm")


# Brain ERROR.code → LiveKit exception mapping (A4, plan Phase A).
#
# Without this mapping, every TurnError became APIConnectionError(retryable=True)
# and the LiveKit framework would retry up to max_retry attempts — including
# auth failures that will never succeed by retry, burning quota and adding
# latency. The table below maps the known brain codes to HTTP-shaped status
# errors so the framework can short-circuit on permanent failures.
#
# Anything unknown / `internal` / `unknown` keeps the previous
# APIConnectionError behavior (respect the brain's `fatal` flag for retryability).
_ERROR_CODE_MAP: dict[str, tuple[int, bool]] = {
    # code: (HTTP status code, retryable)
    "unauthenticated": (401, False),
    "permission_denied": (403, False),
    "tenant_not_found": (404, False),
    "user_not_found": (404, False),
    "rate_limited": (429, True),
}


def _map_turn_error(exc: "TurnError") -> Exception:
    """Translate a brain ERROR event into a LiveKit framework-friendly exception."""
    mapped = _ERROR_CODE_MAP.get(exc.code)
    if mapped is not None:
        status_code, retryable = mapped
        return APIStatusError(
            f"eidolon_agent {exc.code}: {exc}",
            status_code=status_code,
            retryable=retryable,
        )
    return APIConnectionError(
        f"eidolon_agent error {exc.code}: {exc}",
        retryable=not exc.fatal,
    )


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
        conversation_id: str | Callable[[], str],
        display_model: str = "eidolon_agent",
        tls: TlsConfig | None = None,
    ) -> None:
        """Construct an EidolonAgent LLM adapter.

        ``conversation_id`` accepts either:
          - a static string ("livekit:my-room") — eagerly fixed at construction.
          - a zero-arg callable returning a string — resolved lazily on each
            chat() call. This is the path used in production so the brain
            sees a participant-aware id (e.g.
            "livekit:<participant_identity>:<room_name>"), since LiveKit
            participants are only known after session.start() — strictly
            later than factory.from_config() runs.

        D1, plan Phase D.
        """
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
        self._conversation_id: str | Callable[[], str] = conversation_id
        self._display_model = display_model
        # D2: validate TLS config eagerly at construction (fail loud) — same
        # contract as the device_token-empty raise above. We discard the
        # credentials object here; EidolonAgentSession will recompute it when
        # it lazily opens the channel.
        if tls is not None and tls.mode != "off":
            _build_channel_credentials(tls)
        self._tls = tls
        self._session: EidolonAgentSession | None = None
        self._session_lock = asyncio.Lock()
        self._pending_turn_control_metadata: dict[str, Any] | None = None

    def emit_provider_event(self, name: str, **payload: Any) -> None:
        """Emit provider-level timing events for Channel observability."""

        self.emit(
            "provider_event",
            {
                "provider": self.provider,
                "event": name,
                "timestamp": time.monotonic(),
                **payload,
            },
        )

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
                        tls=self._tls,
                    )
        return self._session

    def set_turn_control_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach channel-side turn-control metadata to the next StartTurn."""
        self._pending_turn_control_metadata = dict(metadata)

    def pop_turn_control_metadata(self) -> dict[str, Any] | None:
        metadata = self._pending_turn_control_metadata
        self._pending_turn_control_metadata = None
        return metadata

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
        # D1: resolve conversation_id at chat() time so the resolver can
        # consult LiveKit room state (participant identity etc.) that wasn't
        # available when the adapter was constructed by the factory.
        cid_src = llm_v._conversation_id
        if callable(cid_src):
            try:
                conversation_id = cid_src()
            except Exception as exc:
                logger.warning(
                    "[EidolonAgentGrpcLlm] conversation_id resolver raised; "
                    "falling back to literal display_model. exc=%r",
                    exc,
                )
                conversation_id = f"livekit:{llm_v._display_model}"
        else:
            conversation_id = cid_src
        llm_v.emit_provider_event(
            "brain_request_started",
            conversation_id=conversation_id,
        )
        turn_id, payloads = await session.start_turn(
            text=user_text,
            conversation_id=conversation_id,
            metadata={"turn_control": turn_control}
            if (turn_control := llm_v.pop_turn_control_metadata())
            else None,
        )
        req_id = f"eidolon-{turn_id}"
        llm_v.emit_provider_event(
            "brain_request_sent",
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_id=req_id,
        )
        first_delta_seen = False
        try:
            async for payload in payloads:
                # Dispatch by payload type. Adding a new brain event kind only
                # needs an elif here + a payload dataclass in session.py.
                if isinstance(payload, DeltaPayload):
                    if not first_delta_seen:
                        first_delta_seen = True
                        llm_v.emit_provider_event(
                            "brain_first_delta",
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            request_id=req_id,
                        )
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=req_id,
                            delta=llm.ChoiceDelta(content=payload.text),
                        )
                    )
                elif isinstance(payload, UsagePayload):
                    # LiveKit's _metrics_monitor_task aggregates ChatChunk.usage
                    # into LLMMetrics; surfacing this brings token/cost metrics
                    # back into the framework's observability pipeline.
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=req_id,
                            usage=llm.CompletionUsage(
                                prompt_tokens=payload.prompt_tokens,
                                completion_tokens=payload.completion_tokens,
                                total_tokens=payload.total_tokens,
                            ),
                        )
                    )
                elif isinstance(payload, StatePayload):
                    # UX hook (future): pipeline can subscribe to drive a
                    # "thinking..." indicator. For now we just log so the
                    # signal isn't lost.
                    logger.info(
                        "[EidolonAgentGrpcLlmStream] state=%s turn=%s",
                        payload.state, turn_id,
                    )
                elif isinstance(payload, (ToolCallPayload, CitationPayload, HandoffPayload)):
                    # Channel doesn't surface tools / citations / handoff yet
                    # — log at DEBUG so future work has a hook to grep for.
                    logger.debug(
                        "[EidolonAgentGrpcLlmStream] %s turn=%s payload=%r",
                        type(payload).__name__, turn_id, payload,
                    )
                # else: unknown payload type — ignore (forward-compat with new
                # session.py additions).
            llm_v.emit_provider_event(
                "brain_done",
                conversation_id=conversation_id,
                turn_id=turn_id,
                request_id=req_id,
            )
        except asyncio.CancelledError:
            # Barge-in, preemptive-generation discard, or job teardown: tell the
            # brain to stop generating without closing the underlying bidi
            # stream. Use session.spawn rather than raw asyncio.create_task —
            # keeps a strong ref so the task isn't GC'd mid-flight, and routes
            # any unexpected failure through the session's centralized
            # done-callback diagnostic.
            llm_v.emit_provider_event(
                "brain_cancelled",
                conversation_id=conversation_id,
                turn_id=turn_id,
                request_id=req_id,
            )
            session.spawn(
                session.cancel_turn(turn_id),
                name=f"eidolon-cancel-{turn_id}",
            )
            raise
        except TurnError as exc:
            raise _map_turn_error(exc) from exc
        except Exception as exc:
            raise APIConnectionError(
                f"eidolon_agent stream failed: {exc}",
                retryable=True,
            ) from exc
