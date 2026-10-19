import asyncio
from unittest.mock import AsyncMock

import pytest
from eidolon.livekit.agent.eidolon_agent_rpc.session import EidolonAgentSession, pb


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["wire_cancel", "local_cancel", "close"])
async def test_channel_tool_cleanup_is_awaited(end):
    started = asyncio.Event()
    finished = asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    cleanup = AsyncMock()
    session = EidolonAgentSession(
        target="unused",
        device_token="unused",
        channel_tool_handler=handler,
        channel_tool_cancel_handler=cleanup,
        channel_tool_capabilities=frozenset({"music.play"}),
    )
    session._write = AsyncMock()
    session._inbox["t"] = asyncio.Queue()
    call = pb.TurnEvent(
        turn_id="t",
        kind=pb.TurnEvent.CHANNEL_TOOL_CALL,
        channel_tool_call=pb.ChannelToolCall(call_id="c", name="music.play"),
    )
    session._dispatch(call)
    await started.wait()
    original = session._channel_tool_tasks[("t", "c")]
    session._dispatch(call)  # Retransmission does not overwrite task ownership.
    assert session._channel_tool_tasks[("t", "c")] is original
    if end == "wire_cancel":
        event = pb.TurnEvent(
            turn_id="t",
            kind=pb.TurnEvent.CHANNEL_TOOL_CANCEL,
            channel_tool_cancel=pb.ChannelToolCancel(turn_id="t", call_id="c"),
        )
        session._dispatch(pb.TurnEvent.FromString(event.SerializeToString()))
        await asyncio.gather(original, return_exceptions=True)
    elif end == "local_cancel":
        await session.cancel_turn("t")
    else:
        await session.close_channel_tools()
    assert finished.is_set()
    cleanup.assert_awaited_once()
    assert not session._channel_tool_tasks
    await session.aclose()


@pytest.mark.asyncio
async def test_explicit_outcome_survives_wire_serialization():
    from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolResult
    session = EidolonAgentSession(target="unused", device_token="unused")
    session._write = AsyncMock()
    await session._send_channel_tool_result(ChannelToolResult(
        "t", "c", "example.start", True, {"state": "preparing"}, outcome_state="accepted",
        reference_context={"resource_id": "r"},
    ))
    frame = session._write.call_args.args[0]
    decoded = pb.ChatRequest.FromString(frame.SerializeToString())
    assert decoded.channel_tool_result.outcome_state == "accepted"
    assert decoded.channel_tool_result.reference_context["resource_id"] == "r"
    await session.aclose()
