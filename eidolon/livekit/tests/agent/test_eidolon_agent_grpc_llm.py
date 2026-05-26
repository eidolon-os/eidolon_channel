"""Contract tests for :class:`EidolonAgentGrpcLlm`.

These spin up an in-process gRPC server with a hand-rolled
``EidolonAgentServicer`` that emits a tiny scripted ``TurnEvent`` sequence,
then drive the channel's adapter and assert it produces the expected
:class:`livekit.agents.llm.ChatChunk` stream — and that user-side
cancellation results in a ``CancelTurn`` being written on the same call.
"""

from __future__ import annotations

import asyncio
import socket

import grpc
import grpc.aio
import pytest
from google.protobuf import struct_pb2
from livekit.agents.llm import ChatContext, ChatMessage

from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _delta(turn_id: str, seq: int, text: str) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["text"] = text
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.DELTA, data=data)


def _done(turn_id: str, seq: int) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["status"] = "ok"
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.DONE, data=data)


class _ScriptedServicer(pbg.EidolonAgentServicer):
    """Replays a canned DELTA+DONE per StartTurn. Records cancels for assertions.

    A background task drains the request iterator so cancels are observed even
    while the response generator is mid-flight.
    """

    def __init__(self, *, deltas: list[str], delay_between: float = 0.0) -> None:
        self._deltas = deltas
        self._delay = delay_between
        self.cancels: list[str] = []
        self.starts: list[pb.StartTurn] = []
        self._start_q: asyncio.Queue[pb.StartTurn] = asyncio.Queue()
        self._cancel_evt = asyncio.Event()

    async def _drain_requests(self, request_iterator) -> None:
        async for req in request_iterator:
            kind = req.WhichOneof("payload")
            if kind == "start":
                self.starts.append(req.start)
                await self._start_q.put(req.start)
            elif kind == "cancel":
                self.cancels.append(req.cancel.turn_id)
                self._cancel_evt.set()

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        drain_task = asyncio.create_task(self._drain_requests(request_iterator))
        try:
            start = await self._start_q.get()
            turn_id = start.turn_id
            seq = 0
            for chunk in self._deltas:
                seq += 1
                yield _delta(turn_id, seq, chunk)
                if self._delay > 0:
                    try:
                        await asyncio.wait_for(
                            self._cancel_evt.wait(), timeout=self._delay
                        )
                        return  # cancelled mid-turn — don't emit DONE
                    except asyncio.TimeoutError:
                        pass
            seq += 1
            yield _done(turn_id, seq)
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


async def _serve(servicer: _ScriptedServicer) -> tuple[grpc.aio.Server, str]:
    port = _free_port()
    server = grpc.aio.server()
    pbg.add_EidolonAgentServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    return server, f"127.0.0.1:{port}"


def _ctx(text: str) -> ChatContext:
    ctx = ChatContext()
    ctx.items.append(ChatMessage(role="user", content=[text]))
    return ctx


@pytest.mark.asyncio
async def test_forwards_deltas_then_finishes() -> None:
    servicer = _ScriptedServicer(deltas=["你", "好啊"])
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token="test-token",
            conversation_id="livekit:room-abc",
        )
        try:
            stream = adapter.chat(chat_ctx=_ctx("打个招呼"))
            collected: list[str] = []
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    collected.append(chunk.delta.content)
            assert collected == ["你", "好啊"]
            assert len(servicer.starts) == 1
            assert servicer.starts[0].text == "打个招呼"
            assert servicer.starts[0].conversation_id == "livekit:room-abc"
            assert servicer.starts[0].turn_id  # non-empty uuid hex
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_cancel_writes_cancel_turn() -> None:
    # Server emits one delta, then waits up to 5s — long enough that the
    # adapter's _run task gets cancelled (via stream.aclose) before DONE.
    servicer = _ScriptedServicer(deltas=["第一段"], delay_between=5.0)
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token="test-token",
            conversation_id="livekit:room-xyz",
        )
        try:
            stream = adapter.chat(chat_ctx=_ctx("讲个故事"))
            # Pull the first delta to confirm StartTurn reached the server.
            first = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
            assert first.delta and first.delta.content == "第一段"

            # Barge-in: same path the LiveKit pipeline takes on user interrupt.
            await stream.aclose()

            await asyncio.wait_for(
                _wait_until(lambda: len(servicer.cancels) == 1), timeout=2.0
            )
            assert servicer.cancels[0] == servicer.starts[0].turn_id

            # A2 regression guard: the cancel write is spawned through
            # session.spawn (not raw create_task) so we keep a strong ref
            # while it runs, and the centralized done-callback discards it
            # afterwards. The set must therefore be empty by now (cancel
            # has completed, reader is still running for the reader task —
            # filter to only background tasks named like the cancel).
            cancel_tasks_left = [
                t for t in adapter._session._background_tasks  # type: ignore[union-attr]
                if t.get_name().startswith("eidolon-cancel-")
            ]
            assert cancel_tasks_left == [], f"leaked cancel tasks: {cancel_tasks_left}"
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


async def _wait_until(predicate, *, interval: float = 0.05) -> None:
    while not predicate():
        await asyncio.sleep(interval)
