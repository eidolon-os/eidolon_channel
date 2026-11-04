import asyncio
from types import SimpleNamespace

import pytest

from eidolon_sdk.biz.presentation import SessionOutputPlan, OutputSelection
from eidolon_sdk.biz.presentation.capability_rpc import ExecutionSnapshot
from eidolon.livekit.agent.session import capability_rpc as module
from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolCall


@pytest.mark.asyncio
async def test_rpc_waits_for_terminal_and_keeps_execution_identity(monkeypatch):
    calls = []
    async def peer(room):
        return "bound-device"
    async def perform_rpc(**kw):
        import json
        body = json.loads(kw["payload"])
        calls.append((kw["method"], body["execution_id"]))
        assert kw["destination_identity"] == "bound-device"
        return ExecutionSnapshot(execution_id=body["execution_id"],
            state="accepted" if len(calls) == 1 else "completed", sequence=len(calls),
            completion_basis="software_sequence" if len(calls) > 1 else "unconfirmed").model_dump_json()
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    room = SimpleNamespace(local_participant=SimpleNamespace(perform_rpc=perform_rpc))
    plan = SessionOutputPlan(session_id="session", policy_revision=1,
        outputs=OutputSelection(motion=True), motion_profile="eidolon.motion.v1",
        motion_capabilities=("motion.nod",))
    runtime = module.SessionCapabilityRPC(room, plan)
    call = ChannelToolCall("turn", "call", "motion.nod", {"repeats": 1, "wait_for_completion": True})
    result = await runtime.execute(call)
    assert result.ok and result.outcome_state == "completed"
    assert calls[0][1] == calls[1][1]
    assert not runtime.pending


@pytest.mark.asyncio
async def test_undeclared_capability_never_calls_transport():
    runtime = module.SessionCapabilityRPC(None, SessionOutputPlan(session_id="s", policy_revision=1,
        outputs=OutputSelection(motion=False)))
    result = await runtime.execute(ChannelToolCall("t", "c", "motion.nod", {}))
    assert not result.ok


@pytest.mark.asyncio
async def test_background_acceptance_does_not_wait_and_close_cancels_before_worker_starts(monkeypatch):
    import json
    methods = []
    async def peer(room):
        return "device"
    async def rpc(**kwargs):
        body = json.loads(kwargs["payload"])
        methods.append(kwargs["method"])
        assert "wait_for_completion" not in body.get("arguments", {})
        state = "cancelled" if kwargs["method"] == module.CANCEL else "accepted"
        return ExecutionSnapshot(execution_id=body["execution_id"], state=state, sequence=1).model_dump_json()
    monkeypatch.setattr(module, "wait_for_runtime_participant_identity", peer)
    plan = SessionOutputPlan(session_id="s", policy_revision=1, outputs=OutputSelection(motion=True),
        motion_profile="eidolon.motion.v2", motion_capabilities=("motion.play",))
    runtime = module.SessionCapabilityRPC(SimpleNamespace(local_participant=SimpleNamespace(perform_rpc=rpc)), plan)
    result = await runtime.execute(ChannelToolCall("t", "c", "motion.play",
        {"clip_id": "emotion:laughing1", "wait_for_completion": False}))
    assert result.ok and result.outcome_state == "accepted"
    assert len(methods) == 1
    await runtime.close()
    assert module.CANCEL in methods
    assert not runtime.executions and not runtime.pending
