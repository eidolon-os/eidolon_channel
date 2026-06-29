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
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from eidolon.livekit.agent.eidolon_agent_rpc.proactive import (
        ProactiveHandler,
        ProactiveSubscriber,
    )

from eidolon_sdk.core.grpc import build_channel_credentials, resolve_token_source
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
    ToolResultPayload,
    TurnError,
    UsagePayload,
)

logger = logging.getLogger("eidolon_agent_rpc.grpc_llm")
_USER_TEXT_OVERRIDE_TTL_SEC = 10.0


# ``device_token`` must be a zero-arg sync/async callable that mints the
# runtime token from the LiveKit participant identity. It is resolved once at
# ``_get_session`` time (first chat() call) and cached for the LLM instance
# lifetime.
DeviceTokenSource = Callable[[], str | Awaitable[str]]


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
        device_token: DeviceTokenSource,
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

        ``device_token`` must be a zero-arg sync OR async callable returning
        the token. It is resolved once at first ``_get_session`` call (which
        happens on first chat()) and cached. The resolver reads
        ``participant.metadata`` for kind + identity, queries admin's
        ``/api/resolve``, then signs a fresh JWT.

        D1, plan Phase D.
        """
        super().__init__()
        if not target.strip():
            raise ValueError("EidolonAgentGrpcLlm: target must be non-empty")
        if not callable(device_token):
            raise TypeError("EidolonAgentGrpcLlm: device_token must be a callable resolver")
        self._device_token_source: DeviceTokenSource = device_token
        self._target = target.strip()
        # Concrete token resolved lazily — see ``_resolve_device_token``.
        self._device_token: str | None = None
        self._conversation_id: str | Callable[[], str] = conversation_id
        self._display_model = display_model
        # D2: validate TLS config eagerly at construction (fail loud) — same
        # contract as the device_token-empty raise above. We discard the
        # credentials object here; EidolonAgentSession will recompute it when
        # it lazily opens the channel.
        if tls is not None and tls.mode != "off":
            build_channel_credentials(tls)
        self._tls = tls
        self._session: EidolonAgentSession | None = None
        self._session_lock = asyncio.Lock()
        self._pending_turn_control_metadata: dict[str, Any] | None = None
        self._pending_user_text_override: dict[str, Any] | None = None

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

    async def _resolve_device_token(self) -> str:
        """Resolve the per-session token resolver once and cache it."""
        if self._device_token is not None:
            return self._device_token
        source = self._device_token_source
        try:
            self._device_token = await resolve_token_source(source)
        except Exception as exc:
            raise APIConnectionError(
                f"device_token resolver failed: {exc}"
            ) from exc
        return self._device_token

    async def _get_session(self) -> EidolonAgentSession:
        if self._session is None:
            async with self._session_lock:
                if self._session is None:
                    token = await self._resolve_device_token()
                    self._session = EidolonAgentSession(
                        target=self._target,
                        device_token=token,
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

    def set_next_user_text(self, text: str, *, source: str) -> None:
        """Override the next StartTurn text with Eidolon's canonical user turn.

        LiveKit still owns the framework lifecycle and ChatContext, but Eidolon
        owns the product-level turn assembly.  This one-shot override is the
        narrow adapter boundary that lets ``UserTurnCoordinator`` supply the
        final transcript without mutating LiveKit internals.
        """
        stripped = text.strip()
        self._pending_user_text_override = (
            {
                "text": stripped,
                "source": source,
                "created_at": time.monotonic(),
            }
            if stripped
            else None
        )

    def _pop_next_user_text_for_chat(
        self,
        *,
        framework_user_text: str,
    ) -> dict[str, Any] | None:
        override = self._pending_user_text_override
        if override is None:
            return None
        created_at = override.get("created_at")
        if isinstance(created_at, (int, float)):
            if time.monotonic() - float(created_at) > _USER_TEXT_OVERRIDE_TTL_SEC:
                self._pending_user_text_override = None
                return None
        if not framework_user_text.strip():
            return None
        self._pending_user_text_override = None
        return override

    def _resolve_conversation_id_for_chat(self) -> str:
        # Resolve once per logical chat stream. LiveKit may retry _run() for
        # transient provider errors; retries must not silently move the turn to
        # a different conversation.
        cid_src = self._conversation_id
        if callable(cid_src):
            try:
                return cid_src()
            except Exception as exc:
                logger.warning(
                    "[EidolonAgentGrpcLlm] conversation_id resolver raised; "
                    "falling back to literal display_model. exc=%r",
                    exc,
                )
                return f"livekit:{self._display_model}"
        return cid_src

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
        framework_user_text = _last_user_text(chat_ctx)
        return EidolonAgentGrpcLlmStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            turn_control_metadata=self.pop_turn_control_metadata(),
            user_text_override=self._pop_next_user_text_for_chat(
                framework_user_text=framework_user_text,
            ),
            framework_user_text=framework_user_text,
            conversation_id=self._resolve_conversation_id_for_chat(),
        )

    async def open_proactive_subscriber(
        self,
        *,
        on_event: "ProactiveHandler",
        instance_id: str = "",
    ) -> "ProactiveSubscriber":
        """Build a proactive-report subscriber sharing this LLM's connection config.

        Resolves the device token the same way ``chat()`` does (so the server
        sees the same identity), then hands back a not-yet-running subscriber.
        The pipeline owns the ``run()``/``aclose()`` lifecycle so the stream is
        bounded to the LiveKit job, not the LLM instance.
        """
        from eidolon.livekit.agent.eidolon_agent_rpc.proactive import (
            ProactiveSubscriber,
        )

        token = await self._resolve_device_token()
        return ProactiveSubscriber(
            target=self._target,
            device_token=token,
            on_event=on_event,
            tls=self._tls,
            instance_id=instance_id,
        )

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.aclose()
            self._session = None


class EidolonAgentGrpcLlmStream(llm.LLMStream):
    def __init__(
        self,
        llm: EidolonAgentGrpcLlm,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        turn_control_metadata: dict[str, Any] | None,
        user_text_override: dict[str, Any] | None,
        framework_user_text: str,
        conversation_id: str,
    ) -> None:
        self._turn_control_metadata = turn_control_metadata
        self._user_text_override = user_text_override
        self._framework_user_text = framework_user_text
        self._conversation_id = conversation_id
        self._attempt_index = 0
        super().__init__(
            llm,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )

    async def _run(self) -> None:
        llm_v: EidolonAgentGrpcLlm = self._llm  # type: ignore[assignment]
        self._attempt_index += 1
        attempt = self._attempt_index
        framework_user_text = self._framework_user_text
        override = self._user_text_override
        if override is not None and override["text"].strip():
            user_text = override["text"]
            text_source = override.get("source") or "override"
        else:
            user_text = framework_user_text
            text_source = "framework_chat_context"
        llm_v.emit_provider_event(
            "brain_request_started",
            attempt=attempt,
            text_chars=len(user_text),
            framework_text_chars=len(framework_user_text),
            user_text_source=text_source,
            text_overridden=user_text != framework_user_text,
        )
        timeout = max(float(getattr(self._conn_options, "timeout", 10.0) or 10.0), 0.1)
        try:
            session = await asyncio.wait_for(llm_v._get_session(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise APIConnectionError(
                f"eidolon_agent session open timed out after {timeout:.1f}s",
                retryable=True,
            ) from exc
        conversation_id = self._conversation_id
        try:
            turn_id, payloads = await asyncio.wait_for(
                session.start_turn(
                    text=user_text,
                    conversation_id=conversation_id,
                    metadata={"turn_control": self._turn_control_metadata}
                    if self._turn_control_metadata
                    else None,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise APIConnectionError(
                f"eidolon_agent StartTurn timed out after {timeout:.1f}s",
                retryable=True,
            ) from exc
        req_id = f"eidolon-{turn_id}"
        llm_v.emit_provider_event(
            "brain_request_sent",
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_id=req_id,
            attempt=attempt,
        )
        first_delta_seen = False
        # Roles of non-answer status deltas already spoken this turn, so a
        # preamble is rendered at most once even if the brain repeats it.
        spoken_preamble_roles: set[str] = set()
        try:
            payload_iter = payloads.__aiter__()
            first_delta_deadline = asyncio.get_running_loop().time() + timeout
            while True:
                try:
                    if first_delta_seen:
                        payload = await payload_iter.__anext__()
                    else:
                        remaining = first_delta_deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        payload = await asyncio.wait_for(
                            payload_iter.__anext__(),
                            timeout=remaining,
                        )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    message = (
                        "eidolon_agent first delta timed out after "
                        f"{timeout:.1f}s"
                    )
                    llm_v.emit_provider_event(
                        "brain_error",
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        request_id=req_id,
                        attempt=attempt,
                        code="first_delta_timeout",
                        message=message,
                        fatal=False,
                    )
                    session.spawn(
                        session.cancel_turn(turn_id),
                        name=f"eidolon-first-delta-timeout-cancel-{turn_id}",
                    )
                    raise APIConnectionError(message, retryable=True) from exc

                # Dispatch by payload type. Adding a new brain event kind only
                # needs an elif here + a payload dataclass in session.py.
                if isinstance(payload, DeltaPayload):
                    # Non-answer status lines (e.g. tool preambles) are spoken at
                    # most once per turn and never accumulated as answer content.
                    # The brain already de-dupes per turn; this is the channel's
                    # role-aware guarantee on top of that.
                    if payload.role != "answer":
                        if payload.role in spoken_preamble_roles:
                            continue
                        spoken_preamble_roles.add(payload.role)
                        llm_v.emit_provider_event(
                            "brain_tool_preamble",
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            request_id=req_id,
                            attempt=attempt,
                            role=payload.role,
                        )
                    if not first_delta_seen:
                        first_delta_seen = True
                        llm_v.emit_provider_event(
                            "brain_first_delta",
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            request_id=req_id,
                            attempt=attempt,
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
                    llm_v.emit_provider_event(
                        "brain_state",
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        request_id=req_id,
                        attempt=attempt,
                        state=payload.state,
                    )
                    logger.info(
                        "[EidolonAgentGrpcLlmStream] state=%s turn=%s",
                        payload.state, turn_id,
                    )
                elif isinstance(payload, ToolCallPayload):
                    # The channel neither forwards nor executes tools — the
                    # brain runs its own tool loop. Surface at INFO so the tool
                    # name/args are visible alongside the matching TOOL_RESULT.
                    logger.info(
                        "[EidolonAgentGrpcLlmStream] tool_call name=%s turn=%s",
                        payload.name, turn_id,
                    )
                    llm_v.emit_provider_event(
                        "brain_tool_call",
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        request_id=req_id,
                        attempt=attempt,
                        tool_name=payload.name,
                    )
                elif isinstance(payload, ToolResultPayload):
                    # Surface ok/error at INFO so an operator can tell a failing
                    # tool backend (ok=false, repeated) apart from a result that
                    # never reached the brain's loop. The channel takes no action
                    # on the result itself.
                    logger.info(
                        "[EidolonAgentGrpcLlmStream] tool_result name=%s ok=%s "
                        "error=%s summary=%s turn=%s",
                        payload.name, payload.ok, payload.error or "-",
                        payload.summary or "-", turn_id,
                    )
                    llm_v.emit_provider_event(
                        "brain_tool_result",
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        request_id=req_id,
                        attempt=attempt,
                        tool_name=payload.name,
                        ok=payload.ok,
                        error=payload.error,
                    )
                elif isinstance(payload, (CitationPayload, HandoffPayload)):
                    # Citations / handoff not surfaced yet — DEBUG hook for
                    # future work to grep.
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
                attempt=attempt,
            )
        except APIConnectionError:
            raise
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
                attempt=attempt,
            )
            session.spawn(
                session.cancel_turn(turn_id),
                name=f"eidolon-cancel-{turn_id}",
            )
            raise
        except TurnError as exc:
            llm_v.emit_provider_event(
                "brain_error",
                conversation_id=conversation_id,
                turn_id=turn_id,
                request_id=req_id,
                attempt=attempt,
                code=exc.code,
                message=str(exc),
                fatal=exc.fatal,
            )
            raise _map_turn_error(exc) from exc
        except Exception as exc:
            raise APIConnectionError(
                f"eidolon_agent stream failed: {exc}",
                retryable=True,
            ) from exc
