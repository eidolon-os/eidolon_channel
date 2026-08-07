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

from eidolon_sdk.biz.chat_stream import DeltaRole
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
        ``participant.metadata`` for kind + identity, resolves Kernel Mount and
        System Data runtime facts, then signs a fresh narrow JWT.

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
        self._pending_trace_id: str | None = None
        self._warmer: Any = None  # PreemptiveWarmer, lazily bound to the session

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
            raise APIConnectionError(f"device_token resolver failed: {exc}") from exc
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

    def set_turn_trace_id(self, trace_id: str) -> None:
        """Use the Channel turn id as the next brain turn's cross-hop trace."""

        self._pending_trace_id = trace_id.strip() or None

    def pop_turn_trace_id(self) -> str | None:
        trace_id = self._pending_trace_id
        self._pending_trace_id = None
        return trace_id

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
            logger.debug("[EidolonAgentGrpcLlm] tools are not forwarded over eidolon.agent.v1")
        user_text = _last_user_text(chat_ctx)
        return EidolonAgentGrpcLlmStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            turn_control_metadata=self.pop_turn_control_metadata(),
            trace_id=self.pop_turn_trace_id(),
            user_text=user_text,
            conversation_id=self._resolve_conversation_id_for_chat(),
        )

    async def open_proactive_subscriber(
        self,
        *,
        on_event: "ProactiveHandler",
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
        )

    async def warm(self, text: str) -> None:
        """Preemptively warm the brain on a partial transcript (best effort).

        Called from the turn-detection layer while the user is still speaking;
        fires one ephemeral speculative turn so the real turn's first response
        lands sooner. Superseded (discarded) when the real turn starts.
        """
        try:
            session = await self._get_session()
        except Exception:  # noqa: BLE001 — warming must never break the turn
            return
        if self._warmer is None:
            from eidolon.livekit.agent.eidolon_agent_rpc.preemptive import (
                PreemptiveWarmer,
            )

            self._warmer = PreemptiveWarmer(session, spawn=session.spawn)
        await self._warmer.warm(text, conversation_id=self._resolve_conversation_id_for_chat())

    async def discard_warm(self) -> None:
        """Cancel any in-flight speculative warm-up (real turn supersedes it)."""
        if self._warmer is not None:
            await self._warmer.discard()

    async def aclose(self) -> None:
        if self._warmer is not None:
            await self._warmer.discard()
            self._warmer = None
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
        trace_id: str | None,
        user_text: str,
        conversation_id: str,
    ) -> None:
        self._turn_control_metadata = turn_control_metadata
        self._trace_id = trace_id
        self._user_text = user_text
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
        user_text = self._user_text
        llm_v.emit_provider_event(
            "brain_request_started",
            attempt=attempt,
            text_chars=len(user_text),
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
        # The real turn supersedes any preemptive warm-up for this session.
        await llm_v.discard_warm()
        try:
            turn_id, payloads = await asyncio.wait_for(
                session.start_turn(
                    text=user_text,
                    conversation_id=conversation_id,
                    metadata={"turn_control": self._turn_control_metadata}
                    if self._turn_control_metadata
                    else None,
                    trace_id=self._trace_id,
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
            trace_id=self._trace_id,
        )
        # A complete tool call is valid model activity and must release the
        # first-output deadline, but it is not an audible answer delta. Keep
        # these states separate so long-running tools are not cancelled after
        # 10s and silent-answer fallback still remains accurate.
        first_model_activity_seen = False
        first_delta_seen = False
        first_answer_delta_seen = False
        tool_call_seen = False
        # Roles of non-answer status deltas already spoken this turn, so a
        # preamble is rendered at most once even if the brain repeats it.
        spoken_preamble_roles: set[str] = set()
        try:
            payload_iter = payloads.__aiter__()
            first_delta_deadline = asyncio.get_running_loop().time() + timeout
            while True:
                try:
                    if first_model_activity_seen:
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
                    message = f"eidolon_agent first delta timed out after {timeout:.1f}s"
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
                    # The brain has already accepted this logical turn and may
                    # have compiled context or started provider work. Retrying
                    # at LiveKit's LLMStream layer would create a second Agent
                    # turn for the same user utterance, duplicating persistence
                    # and potentially replaying tools. Transport failures before
                    # StartTurn is accepted remain retryable; an accepted turn
                    # that never produces a usable delta is terminal here.
                    raise APIConnectionError(message, retryable=False) from exc

                # Dispatch by payload type. Adding a new brain event kind only
                # needs an elif here + a payload dataclass in session.py.
                if isinstance(payload, DeltaPayload):
                    # Non-answer status lines are status chrome by default: emit
                    # them as provider events for UI/observability, but keep them
                    # out of TTS. A slow-tool hint is the one non-answer role that
                    # is intentionally spoken, and is still de-duped per turn.
                    if payload.role != DeltaRole.ANSWER.value:
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
                        if payload.role != DeltaRole.SLOW_TOOL_HINT.value:
                            continue
                    if not first_delta_seen:
                        if not first_model_activity_seen:
                            first_model_activity_seen = True
                            llm_v.emit_provider_event(
                                "brain_first_model_activity",
                                conversation_id=conversation_id,
                                turn_id=turn_id,
                                request_id=req_id,
                                attempt=attempt,
                                kind="answer_delta",
                            )
                        first_delta_seen = True
                        llm_v.emit_provider_event(
                            "brain_first_delta",
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            request_id=req_id,
                            attempt=attempt,
                        )
                    if payload.role == DeltaRole.ANSWER.value and not first_answer_delta_seen:
                        first_answer_delta_seen = True
                        llm_v.emit_provider_event(
                            "brain_first_answer_delta",
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
                        payload.state,
                        turn_id,
                    )
                elif isinstance(payload, ToolCallPayload):
                    # The channel neither forwards nor executes tools — the
                    # brain runs its own tool loop. Surface at INFO so the tool
                    # name/args are visible alongside the matching TOOL_RESULT.
                    tool_call_seen = True
                    if not first_model_activity_seen:
                        first_model_activity_seen = True
                        llm_v.emit_provider_event(
                            "brain_first_model_activity",
                            conversation_id=conversation_id,
                            turn_id=turn_id,
                            request_id=req_id,
                            attempt=attempt,
                            kind="tool_call",
                        )
                    logger.info(
                        "[EidolonAgentGrpcLlmStream] tool_call name=%s turn=%s",
                        payload.name,
                        turn_id,
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
                        payload.name,
                        payload.ok,
                        payload.error or "-",
                        payload.summary or "-",
                        turn_id,
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
                        type(payload).__name__,
                        turn_id,
                        payload,
                    )
                # else: unknown payload type — ignore (forward-compat with new
                # session.py additions).
            if not first_answer_delta_seen and not tool_call_seen:
                message = "eidolon_agent completed without answer or tool call"
                llm_v.emit_provider_event(
                    "brain_error",
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    request_id=req_id,
                    attempt=attempt,
                    code="no_usable_output",
                    message=message,
                    fatal=False,
                )
                raise APIConnectionError(message, retryable=False)
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
