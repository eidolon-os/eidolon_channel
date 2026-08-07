"""Contract test for :class:`ProactiveSubscriber`.

Spins up an in-process gRPC server whose ``SubscribeProactive`` streams a couple
of scripted :class:`ProactiveEvent` frames, then drives the channel-side
subscriber and asserts every event reaches the ``on_event`` handler verbatim.
"""

from __future__ import annotations

import asyncio
import socket

import grpc
import grpc.aio
import pytest

from eidolon.livekit.agent.eidolon_agent_rpc.proactive import (
    ProactiveReport,
    ProactiveSubscriber,
)
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


class _ProactiveServicer(pbg.EidolonAgentServicer):
    """Streams a fixed list of ProactiveEvents then ends the stream."""

    def __init__(self, *, events: list[pb.ProactiveEvent]) -> None:
        self._events = events
        self.requests: list[pb.SubscribeRequest] = []

    async def SubscribeProactive(self, request, context):  # type: ignore[override]
        self.requests.append(request)
        for event in self._events:
            yield event


async def _serve(servicer) -> tuple[grpc.aio.Server, str]:
    port = _free_port()
    server = grpc.aio.server()
    pbg.add_EidolonAgentServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    return server, f"127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_subscriber_forwards_events_to_handler() -> None:
    events = [
        pb.ProactiveEvent(
            instance_id="inst_abc",
            intent="long_task_done",
            text="会议纪要整理好了，包含摘要和行动项。",
            style_hint="report",
        ),
        pb.ProactiveEvent(
            instance_id="inst_abc",
            intent="long_task_done",
            text="第二条提醒。",
            style_hint="report",
        ),
    ]
    servicer = _ProactiveServicer(events=events)
    server, target = await _serve(servicer)
    received: list[ProactiveReport] = []

    async def _on_event(report: ProactiveReport) -> None:
        received.append(report)

    subscriber = ProactiveSubscriber(
        target=target,
        device_token="test-token",
        on_event=_on_event,
    )
    try:
        # _consume_once returns when the server ends the stream (both events
        # delivered), so we don't need the reconnect loop for the assertion.
        await asyncio.wait_for(subscriber._consume_once(), timeout=5.0)
    finally:
        await subscriber.aclose()
        await server.stop(grace=0.5)

    assert [r.text for r in received] == [
        "会议纪要整理好了，包含摘要和行动项。",
        "第二条提醒。",
    ]
    assert received[0].instance_id == "inst_abc"
    assert received[0].intent == "long_task_done"
    assert received[0].style_hint == "report"
    assert len(servicer.requests) == 1
    assert servicer.requests[0] == pb.SubscribeRequest()


@pytest.mark.asyncio
async def test_subscriber_handler_error_does_not_break_stream() -> None:
    events = [
        pb.ProactiveEvent(instance_id="i", intent="x", text="boom", style_hint=""),
        pb.ProactiveEvent(instance_id="i", intent="x", text="ok", style_hint=""),
    ]
    servicer = _ProactiveServicer(events=events)
    server, target = await _serve(servicer)
    received: list[str] = []

    async def _on_event(report: ProactiveReport) -> None:
        if report.text == "boom":
            raise RuntimeError("handler blew up")
        received.append(report.text)

    subscriber = ProactiveSubscriber(
        target=target,
        device_token="test-token",
        on_event=_on_event,
    )
    try:
        await asyncio.wait_for(subscriber._consume_once(), timeout=5.0)
    finally:
        await subscriber.aclose()
        await server.stop(grace=0.5)

    # First handler raised; the stream survived and delivered the second event.
    assert received == ["ok"]
