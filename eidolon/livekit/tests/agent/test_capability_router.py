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


@pytest.mark.asyncio
async def test_a_device_receipt_is_checked_by_what_the_device_produced():
    """The device answers with an execution snapshot; the contract describes its result.

    Driven through the real session transport with the installed volume
    contract, whose result schema is exact (``volume`` required, nothing
    else). Checking the whole snapshot against it refused a volume the device
    had already applied — found on a StackChan, where the companion told the
    person it had failed.
    """
    from types import SimpleNamespace

    from eidolon_sdk.biz.capabilities import installed_registry
    from eidolon_sdk.biz.presentation import OutputSelection, SessionOutputPlan
    from eidolon_sdk.biz.presentation.capability_rpc import (
        INVOKE, CapabilityInvocation, ExecutionSnapshot,
    )

    from eidolon.livekit.agent.session import capability_rpc as rpc_module
    from eidolon.livekit.agent.session.capability_rpc import SessionCapabilityRPC

    applied = {"volume": 50}

    class Participant:
        async def perform_rpc(self, *, destination_identity, method, payload, response_timeout):
            request = CapabilityInvocation.model_validate_json(payload)
            assert method == INVOKE and request.arguments == {"volume": 50}
            return ExecutionSnapshot(
                execution_id=request.execution_id, state="completed", sequence=1,
                completion_basis="configuration_applied", result=dict(applied),
            ).model_dump_json()

    async def device_identity(room):
        return "device-1"

    room = SimpleNamespace(local_participant=Participant())
    plan = SessionOutputPlan(session_id="s-1", policy_revision=1,
                             outputs=OutputSelection(speech=True),
                             device_capabilities=("audio.volume.set",))
    original = rpc_module.wait_for_runtime_participant_identity
    rpc_module.wait_for_runtime_participant_identity = device_identity
    try:
        transport = SessionCapabilityRPC(room, plan)
        router = CapabilityRouter(installed_registry())
        router.register(transport.names, transport.execute, transport.cancel, location="device")
        result = await router.execute(
            ChannelToolCall("t", "c", "audio.volume.set", {"volume": 50})
        )
        assert result.ok and result.content["result"] == {"volume": 50}

        # A device that claims a result its contract does not admit is still refused.
        applied["volume"] = 101
        with pytest.raises(ValueError):
            await router.execute(ChannelToolCall("t", "d", "audio.volume.set", {"volume": 50}))
    finally:
        rpc_module.wait_for_runtime_participant_identity = original
