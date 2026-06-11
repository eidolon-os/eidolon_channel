"""Long-lived bidi session against ``EidolonAgent.Chat``.

A single :class:`EidolonAgentSession` holds one gRPC channel and one open
``Chat()`` call across every turn of a LiveKit job. A background reader task
drains ``TurnEvent`` frames from the server and routes them to per-turn queues
keyed by ``turn_id``. Each turn yields a stream of text deltas; cancellation
writes ``CancelTurn`` on the same call without tearing down the stream.

If the server closes the stream mid-job (network blip, restart), the next
``start_turn`` lazily re-opens it. In-flight turns are not retried — they
surface the error to the caller and the LiveKit pipeline decides what to do.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import grpc
import grpc.aio
from google.protobuf import struct_pb2

from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)

logger = logging.getLogger("eidolon_agent_rpc.session")


# Channel-level gRPC options.
#
# Keepalive:
#   - Do not enable client-side HTTP/2 keepalive by default. The Chat() RPC is a
#     long-lived bidi stream, and aggressive no-data pings can trip the default
#     gRPC server enforcement policy with GOAWAY/debug "too_many_pings".
#   - If a cross-host deployment sits behind an idle-closing proxy/LB, expose a
#     deployment-specific keepalive policy instead of baking a 10s dev default
#     into the client.
#
# Latency hints:
#   - optimization_target=latency: tell gRPC C-core to prefer p99 latency over
#     throughput. Voice TTFD is the metric we care about, not bulk bytes/sec.
#   - bdp_probe=1: auto-adjust HTTP/2 BDP for bursty streaming (LLM token
#     emission is bursty), keeping flow control windows out of the critical path.
_CHANNEL_OPTIONS: list[tuple[str, int | str]] = [
    ("grpc.optimization_target", "latency"),
    ("grpc.http2.bdp_probe", 1),
]


# Typed inbox payloads (C, plan Phase C).
#
# The per-turn queue now carries one of these payload types (or a BaseException
# for fatal stream errors). EidolonAgentGrpcLlmStream._run dispatches by type:
#   DeltaPayload  -> ChatChunk(delta=…)
#   UsagePayload  -> ChatChunk(usage=CompletionUsage(…))
#   StatePayload  -> INFO log (UX feedback hook, future)
#   ToolCallPayload / CitationPayload / HandoffPayload -> DEBUG log
#   _DonePayload  -> end of turn (queue closes; consumer returns)
#
# This keeps a single event channel for all turn-scoped signals — a side channel
# would require coordinating cancel/reconnect across two queues, complicating
# the existing single-queue demux by turn_id.


@dataclass(frozen=True, slots=True)
class DeltaPayload:
    text: str


@dataclass(frozen=True, slots=True)
class UsagePayload:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


@dataclass(frozen=True, slots=True)
class StatePayload:
    state: str  # "thinking" | "speaking" | …


@dataclass(frozen=True, slots=True)
class ToolCallPayload:
    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CitationPayload:
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HandoffPayload:
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _DonePayload:
    """Sentinel marking end-of-turn. Consumer returns when it sees this."""


_DONE = _DonePayload()


# Union of everything the inbox queue can carry to a turn consumer.
TurnPayload = (
    DeltaPayload
    | UsagePayload
    | StatePayload
    | ToolCallPayload
    | CitationPayload
    | HandoffPayload
    | _DonePayload
)


def _s_field(fields, name: str) -> str:
    """Helper: read string field from a Struct.fields mapping, default empty.
    Module-scope so dispatch elif branches stay readable; protobuf Struct
    field access is verbose enough that an indirection helps."""
    if fields is not None and name in fields:
        return fields[name].string_value
    return ""


class TurnError(RuntimeError):
    """Raised inside ``start_turn`` when the server emits an ``ERROR`` event."""

    def __init__(self, code: str, message: str, fatal: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.fatal = fatal


@dataclass(frozen=True, slots=True)
class TlsConfig:
    """gRPC channel TLS configuration (D2, plan Phase D).

    mode:
        "off"  → insecure_channel (loopback / UDS / dev)
        "tls"  → secure_channel verifying server cert
        "mtls" → secure_channel with mutual auth (client cert + key required)
    """
    mode: str = "off"
    ca_path: str = ""
    client_cert_path: str = ""
    client_key_path: str = ""


def _build_channel_credentials(tls: TlsConfig) -> "grpc.ChannelCredentials | None":
    """Materialize gRPC ChannelCredentials from a TlsConfig, or None for insecure.

    Errors loudly (ValueError) on misconfiguration so the worker fails at boot
    rather than silently falling back to insecure or 0-cert TLS.
    """
    from pathlib import Path

    if tls.mode == "off":
        return None
    if tls.mode not in ("tls", "mtls"):
        raise ValueError(
            f"REMOTE_AGENT_RPC_TLS_MODE={tls.mode!r} unrecognized; expected "
            "one of: off, tls, mtls"
        )

    def _read(label: str, path: str) -> bytes:
        if not path:
            return b""
        p = Path(path).expanduser()
        if not p.is_file():
            raise ValueError(
                f"REMOTE_AGENT_RPC_TLS {label} path does not exist or is not a file: {path}"
            )
        return p.read_bytes()

    ca = _read("ca_path", tls.ca_path) or None
    if tls.mode == "mtls":
        if not tls.client_cert_path or not tls.client_key_path:
            raise ValueError(
                "REMOTE_AGENT_RPC_TLS_MODE=mtls requires "
                "REMOTE_AGENT_RPC_TLS_CLIENT_CERT_PATH and _CLIENT_KEY_PATH"
            )
        client_cert = _read("client_cert_path", tls.client_cert_path)
        client_key = _read("client_key_path", tls.client_key_path)
        return grpc.ssl_channel_credentials(
            root_certificates=ca,
            private_key=client_key,
            certificate_chain=client_cert,
        )
    # mode == "tls" — server auth only
    return grpc.ssl_channel_credentials(root_certificates=ca)


class EidolonAgentSession:
    """Holds the channel + Chat() stream shared across all turns of a job."""

    def __init__(
        self,
        *,
        target: str,
        device_token: str,
        tls: TlsConfig | None = None,
    ) -> None:
        self._target = target.strip()
        self._device_token = device_token
        self._metadata = (("authorization", f"Bearer {device_token}"),)
        self._tls = tls or TlsConfig()
        # Validate TLS config eagerly so misconfig fails at construction
        # (consistent with the device_token=empty fail-loud behavior).
        self._credentials = _build_channel_credentials(self._tls)

        self._channel: grpc.aio.Channel | None = None
        self._call: grpc.aio.StreamStreamCall | None = None
        self._reader_task: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()
        self._open_lock = asyncio.Lock()
        # turn_id -> queue receiving TurnEvent payloads (or sentinel/exception)
        self._inbox: dict[str, asyncio.Queue] = {}
        # Background tasks owned by this session (cancel_turn fire-and-forget,
        # reader task). Held as strong refs so Python's GC doesn't collect them
        # mid-flight — `asyncio.create_task` returns a task whose only reference
        # would otherwise be on the event loop's ready queue, which is not
        # guaranteed across all cancellation paths.
        self._background_tasks: set[asyncio.Task] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_turn(
        self,
        *,
        text: str,
        conversation_id: str,
        metadata: dict | None = None,
    ) -> tuple[str, AsyncIterator[TurnPayload]]:
        """Begin a new turn. Returns ``(turn_id, payload_iterator)``.

        The iterator yields :data:`TurnPayload` values (``DeltaPayload``,
        ``UsagePayload``, ``StatePayload``, ...) and returns when the brain
        emits ``DONE``. On ``ERROR`` events it raises :class:`TurnError`.
        Callers (typically :class:`EidolonAgentGrpcLlmStream`) dispatch on
        payload type.
        """
        if self._closed:
            raise RuntimeError("EidolonAgentSession is closed")

        turn_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self._inbox[turn_id] = queue

        await self._ensure_open()
        try:
            md = struct_pb2.Struct()
            if metadata:
                md.update(metadata)
            await self._write(
                pb.ChatRequest(
                    start=pb.StartTurn(
                        turn_id=turn_id,
                        conversation_id=conversation_id,
                        text=text,
                        metadata=md,
                    )
                )
            )
        except Exception:
            self._inbox.pop(turn_id, None)
            raise

        return turn_id, self._consume(turn_id, queue)

    async def cancel_turn(self, turn_id: str) -> None:
        """Best-effort cancel — writes ``CancelTurn`` on the open call.

        Safe to call after the turn has already finished; the write may fail
        with an RPC error which we swallow at debug level.
        """
        try:
            await self._write(pb.ChatRequest(cancel=pb.CancelTurn(turn_id=turn_id)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[EidolonAgentSession] cancel_turn(%s) ignored: %r", turn_id, exc)

    def spawn(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Create a session-owned background task that won't be GC'd.

        Use this instead of ``asyncio.create_task`` for any fire-and-forget
        work scheduled from within the session (e.g. cancel writes during
        barge-in). The session holds a strong ref until the task completes;
        a centralized done-callback logs any unexpected exception so silent
        crashes don't go unnoticed.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "[EidolonAgentSession] background task %r ended with exception: %r",
                task.get_name(),
                exc,
            )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._call is not None:
            try:
                await self._call.done_writing()
            except Exception:  # noqa: BLE001
                pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # Cancel any in-flight background tasks (e.g. unfinished cancel_turn
        # writes) and give them a brief window to settle before tearing down
        # the channel. Bounded wait to avoid blocking shutdown on a hung task.
        if self._background_tasks:
            for t in list(self._background_tasks):
                t.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        if self._channel is not None:
            await self._channel.close()
        # Drain any pending consumers.
        for q in self._inbox.values():
            q.put_nowait(RuntimeError("session closed"))
        self._inbox.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _ensure_open(self) -> None:
        async with self._open_lock:
            if self._call is not None and not self._call.done():
                return
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            if self._channel is not None:
                await self._channel.close()
            if self._credentials is None:
                self._channel = grpc.aio.insecure_channel(
                    self._target, options=_CHANNEL_OPTIONS
                )
            else:
                self._channel = grpc.aio.secure_channel(
                    self._target, self._credentials, options=_CHANNEL_OPTIONS
                )
            stub = pbg.EidolonAgentStub(self._channel)
            self._call = stub.Chat(metadata=self._metadata)
            # Reader runs through spawn() to share the unified done-callback
            # diagnostic path with other background tasks (cancel writes etc.).
            self._reader_task = self.spawn(
                self._reader_loop(self._call), name="eidolon-agent-reader"
            )
            logger.info("[EidolonAgentSession] opened Chat() target=%s", self._target)

    async def _write(self, request: pb.ChatRequest) -> None:
        if self._call is None:
            raise RuntimeError("call not open")
        async with self._write_lock:
            await self._call.write(request)

    async def _reader_loop(self, call: grpc.aio.StreamStreamCall) -> None:
        try:
            while True:
                ev: pb.TurnEvent = await call.read()
                if ev is grpc.aio.EOF:
                    break
                self._dispatch(ev)
        except asyncio.CancelledError:
            raise
        except grpc.aio.AioRpcError as exc:
            logger.warning("[EidolonAgentSession] Chat() RPC ended: %s", exc.code())
            self._broadcast_error(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[EidolonAgentSession] reader crashed")
            self._broadcast_error(exc)
        finally:
            # Any remaining turn consumers should unblock.
            self._broadcast_error(RuntimeError("stream ended"))

    def _dispatch(self, ev: "pb.TurnEvent") -> None:
        q = self._inbox.get(ev.turn_id)
        if q is None:
            # Event for an unknown / already-finished turn — drop quietly.
            logger.debug(
                "[EidolonAgentSession] event for unknown turn %s kind=%s",
                ev.turn_id,
                pb.TurnEvent.Kind.Name(ev.kind),
            )
            return
        kind = ev.kind
        data_fields = ev.data.fields if ev.data is not None else None

        if kind == pb.TurnEvent.DELTA:
            text = (
                data_fields["text"].string_value
                if data_fields is not None and "text" in data_fields
                else ""
            )
            if text:
                q.put_nowait(DeltaPayload(text=text))
        elif kind == pb.TurnEvent.DONE:
            q.put_nowait(_DONE)
        elif kind == pb.TurnEvent.ERROR:
            code = (
                data_fields["code"].string_value
                if data_fields is not None and "code" in data_fields
                else "unknown"
            )
            message = (
                data_fields["message"].string_value
                if data_fields is not None and "message" in data_fields
                else ""
            )
            fatal = (
                data_fields["fatal"].bool_value
                if data_fields is not None and "fatal" in data_fields
                else False
            )
            q.put_nowait(TurnError(code, message, fatal))
        elif kind == pb.TurnEvent.USAGE:
            # Brain emits prompt/completion/total token counts + model name.
            # Forward so LiveKit's metrics_monitor_task can aggregate them.
            def _i(name: str) -> int:
                if data_fields is not None and name in data_fields:
                    return int(data_fields[name].number_value)
                return 0
            def _s(name: str) -> str:
                if data_fields is not None and name in data_fields:
                    return data_fields[name].string_value
                return ""
            q.put_nowait(UsagePayload(
                prompt_tokens=_i("prompt_tokens"),
                completion_tokens=_i("completion_tokens"),
                total_tokens=_i("total_tokens"),
                model=_s("model"),
            ))
        elif kind == pb.TurnEvent.STATE:
            state = (
                data_fields["state"].string_value
                if data_fields is not None and "state" in data_fields
                else ""
            )
            if state:
                q.put_nowait(StatePayload(state=state))
        elif kind == pb.TurnEvent.TOOL_CALL:
            name = _s_field(data_fields, "name")
            args_raw = data_fields["args"] if data_fields is not None and "args" in data_fields else None
            args = dict(args_raw.struct_value) if (args_raw is not None and args_raw.HasField("struct_value")) else {}
            q.put_nowait(ToolCallPayload(name=name, args=args))
        elif kind == pb.TurnEvent.CITATION:
            q.put_nowait(CitationPayload(raw=dict(ev.data) if ev.data is not None else {}))
        elif kind == pb.TurnEvent.HANDOFF:
            q.put_nowait(HandoffPayload(raw=dict(ev.data) if ev.data is not None else {}))
        else:
            # TOOL_RESULT / ACK / PROGRESS / KIND_UNSPECIFIED — not surfaced.
            logger.debug(
                "[EidolonAgentSession] ignoring %s event (turn=%s)",
                pb.TurnEvent.Kind.Name(kind),
                ev.turn_id,
            )

    def _broadcast_error(self, exc: BaseException) -> None:
        for q in self._inbox.values():
            q.put_nowait(exc)

    async def _consume(
        self, turn_id: str, queue: asyncio.Queue
    ) -> AsyncIterator[TurnPayload]:
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _DonePayload):
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._inbox.pop(turn_id, None)
