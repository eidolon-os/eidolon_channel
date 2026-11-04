import pytest
from eidolon_sdk.biz.capabilities import CapabilityDefinition, CapabilityRegistry

from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolCall, ChannelToolResult
from eidolon.livekit.agent.session.capability_router import CapabilityRouter


@pytest.mark.asyncio
async def test_dynamic_binding_routes_once_and_validates_arguments():
    registry = CapabilityRegistry()
    registry.register(
        CapabilityDefinition(
            name="example.blink",
            llm_name="example_blink",
            description="Blink.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            execution_location="channel",
            required_output="speech",
        )
    )
    router = CapabilityRouter(registry)
    calls = []

    async def execute(call):
        calls.append(call)
        return ChannelToolResult(
            call.turn_id, call.call_id, call.name, True, {}, outcome_state="completed"
        )

    router.register(["example.blink"], execute, None, location="channel")
    assert (await router.execute(ChannelToolCall("t", "c", "example.blink", {}))).ok
    assert not (await router.execute(ChannelToolCall("t", "d", "example.blink", {"bad": 1}))).ok
    assert not (await router.execute(ChannelToolCall("t", "e", "unknown.blink", {}))).ok
    assert len(calls) == 1
    with pytest.raises(ValueError):
        router.register(["example.blink"], execute, None, location="channel")
