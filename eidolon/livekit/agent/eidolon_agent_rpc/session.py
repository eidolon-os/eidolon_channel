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
from typing import AsyncIterator

import grpc
import grpc.aio

from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)

logger = logging.getLogger("eidolon_agent_rpc.session")


# Channel-level gRPC options.
#
# Keepalive (A3, plan Phase A):
#   - Loopback never times out, but cross-host / via-LB deployments silently
#     drop idle long-conns. Probe every 10s, fail after 5s no-ACK.
#   - permit_without_calls=1 + max_pings_without_data=0 let us ping even on a
#     stream with no active turn (e.g. between voice turns).
#
# Latency hints (F3, plan Phase F):
#   - optimization_target=latency: tell gRPC C-core to prefer p99 latency over
#     throughput. Voice TTFD is the metric we care about, not bulk bytes/sec.
#   - bdp_probe=1: auto-adjust HTTP/2 BDP for bursty streaming (LLM token
#     emission is bursty), keeping flow control windows out of the critical path.
_CHANNEL_OPTIONS: list[tuple[str, int | str]] = [
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.optimization_target", "latency"),
    ("grpc.http2.bdp_probe", 1),
]


_QUEUE_SENTINEL_DONE = object()


class TurnError(RuntimeError):
    """Raised inside ``start_turn`` when the server emits an ``ERROR`` event."""

    def __init__(self, code: str, message: str, fatal: bool) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.fatal = fatal


class EidolonAgentSession:
    """Holds the channel + Chat() stream shared across all turns of a job."""

    def __init__(
        self,
        *,
        target: str,
        device_token: str,
    ) -> None:
        self._target = target.strip()
        self._device_token = device_token
        self._metadata = (("authorization", f"Bearer {device_token}"),)

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
        self, *, text: str, conversation_id: str
    ) -> tuple[str, AsyncIterator[str]]:
        """Begin a new turn. Returns ``(turn_id, delta_iterator)``.

        The iterator yields text fragments from ``DELTA`` events until a
        ``DONE`` event closes the turn. On ``ERROR`` events it raises
        :class:`TurnError`.
        """
        if self._closed:
            raise RuntimeError("EidolonAgentSession is closed")

        turn_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self._inbox[turn_id] = queue

        await self._ensure_open()
        try:
            await self._write(
                pb.ChatRequest(
                    start=pb.StartTurn(
                        turn_id=turn_id,
                        conversation_id=conversation_id,
                        text=text,
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
            self._channel = grpc.aio.insecure_channel(
                self._target, options=_CHANNEL_OPTIONS
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
        if kind == pb.TurnEvent.DELTA:
            text = ""
            if ev.data is not None:
                text = ev.data.fields["text"].string_value if "text" in ev.data.fields else ""
            if text:
                q.put_nowait(text)
        elif kind == pb.TurnEvent.DONE:
            q.put_nowait(_QUEUE_SENTINEL_DONE)
        elif kind == pb.TurnEvent.ERROR:
            code = (
                ev.data.fields["code"].string_value
                if ev.data and "code" in ev.data.fields
                else "unknown"
            )
            message = (
                ev.data.fields["message"].string_value
                if ev.data and "message" in ev.data.fields
                else ""
            )
            fatal = (
                ev.data.fields["fatal"].bool_value
                if ev.data and "fatal" in ev.data.fields
                else False
            )
            q.put_nowait(TurnError(code, message, fatal))
        else:
            # STATE / TOOL_CALL / TOOL_RESULT / CITATION / USAGE / ACK /
            # PROGRESS / HANDOFF — not surfaced to LiveKit yet.
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
    ) -> AsyncIterator[str]:
        try:
            while True:
                item = await queue.get()
                if item is _QUEUE_SENTINEL_DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            self._inbox.pop(turn_id, None)
