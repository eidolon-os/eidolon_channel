"""The worker applies its device's tool restriction where only the Channel can.

The Provider has already narrowed the session plan. What is left for the
worker (工具总览与开关设计 §6.3): read the restriction from the Provider's dispatch
and nowhere else, keep the Channel's own tools (music, vision.who) that it
withholds out of what is offered and routed, and tell the brain on every turn,
a warm-up included, so the Agent withholds its own tools as well.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm
from eidolon.livekit.agent.server import _resolve_device_tool_deny

from .test_eidolon_agent_grpc_llm import _ctx, _ScriptedServicer, _serve


def _job(metadata: dict) -> SimpleNamespace:
    return SimpleNamespace(job=SimpleNamespace(metadata=json.dumps(metadata)))


def test_the_restriction_is_read_from_the_providers_dispatch():
    restricted = _job({"conversation_id": "c", "tool_restriction": {
        "revision": 4, "denied": ["get_weather", "music.play"]}})
    assert _resolve_device_tool_deny(restricted) == frozenset({"get_weather", "music.play"})
    # Never restricted: the Provider writes nothing.
    assert _resolve_device_tool_deny(_job({"conversation_id": "c"})) == frozenset()
    assert _resolve_device_tool_deny(SimpleNamespace(job=SimpleNamespace(metadata=""))) == frozenset()


@pytest.mark.parametrize("restriction", [
    {"revision": 0, "denied": []},
    {"revision": 1, "denied": "get_weather"},
    {"revision": 1, "denied": ["Not A Key"]},
])
def test_a_malformed_restriction_fails_the_session_rather_than_lifting_it(restriction):
    with pytest.raises(ValidationError):
        _resolve_device_tool_deny(_job({"conversation_id": "c", "tool_restriction": restriction}))


async def test_the_channels_own_withheld_tools_are_neither_offered_nor_routed():
    async def handler(call):
        raise AssertionError(f"withheld {call.name} was routed")

    llm = EidolonAgentGrpcLlm(
        target="127.0.0.1:1", device_token=lambda: "x", conversation_id="c",
        device_tool_deny=frozenset({"music.play", "vision.who"}),
    )
    llm.add_channel_tools(handler, None, frozenset({"vision.who"}))
    llm.configure_channel_tools(handler, None, frozenset({"music.play", "music.stop"}))
    assert llm._channel_tool_capabilities == frozenset({"music.stop"})
    for name in ("music.play", "vision.who"):
        result = await llm._channel_tool_handler(
            SimpleNamespace(turn_id="t", call_id="c", name=name, arguments={}))
        assert (result.ok, result.error_code) == (False, "UNAVAILABLE")
    # The music runtime going away and coming back keeps the restriction.
    llm.configure_channel_tools(None, None, frozenset())
    llm.configure_channel_tools(handler, None, frozenset({"music.play", "music.stop"}))
    assert llm._channel_tool_capabilities == frozenset({"music.stop"})


async def _until(condition, timeout: float = 2.0) -> None:
    async def wait():
        while not condition():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


@pytest.mark.parametrize("denied, told", [
    (frozenset({"music.play", "get_weather"}), ["get_weather", "music.play"]),
    (frozenset(), []),
])
async def test_every_turn_and_its_warm_up_tell_the_brain_what_the_device_withholds(denied, told):
    """Always present on the wire, empty when there is no restriction."""

    servicer = _ScriptedServicer(deltas=["好"])
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target, device_token=lambda: "test-token",
            conversation_id="livekit:room-abc", device_tool_deny=denied,
        )
        try:
            await adapter.warm("帮我查一下明天的天气")
            await _until(lambda: len(servicer.starts) == 1)
            async for _ in adapter.chat(chat_ctx=_ctx("帮我查一下明天的天气")):
                pass
            warm_up, turn = servicer.starts
            assert (warm_up.speculative, turn.speculative) == (True, False)
            for start in (warm_up, turn):
                assert MessageToDict(start.metadata)["device_tool_deny"] == told
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)
