import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon_sdk.biz.presentation import (
    FACE_PROFILE,
    OutputSelection,
    ResponseIntent,
    SessionOutputPlan,
)
from eidolon.livekit.agent.session.presentation import PresentationTransport, compile_expression
from eidolon.livekit.agent.session.agent_output_coordinator import AgentOutputCoordinator


def output():
    return SessionOutputPlan(
        session_id="session-1",
        policy_revision=1,
        outputs=OutputSelection(expression=True),
        expression_profile=FACE_PROFILE,
    )


def intent(kind="acknowledge"):
    return ResponseIntent(
        intent=kind, turn_id="turn-1", response_id="response:turn-1", session_id="session-1"
    )


@pytest.mark.parametrize(
    "kind,gesture",
    [
        ("acknowledge", "attend"),
        ("confirm", "affirm"),
        ("consider", "ponder"),
        ("clarify", "question"),
        ("comfort", "soften"),
        ("celebrate", "delight"),
        ("decline", "hesitate"),
        ("notify", "attention"),
    ],
)
def test_semantic_compiler_uses_device_gesture_catalog(kind, gesture):
    plan = compile_expression(intent(kind), output())
    assert plan.steps[0].gesture == gesture
    assert len(plan.model_dump_json().encode()) <= 2048
    assert plan.max_duration_ms == 1200


def test_compiler_requires_selected_session():
    with pytest.raises(ValueError):
        compile_expression(intent().model_copy(update={"session_id": "old"}), output())
    assert compile_expression(intent("none"), output()) is None


@pytest.mark.asyncio
async def test_delivery_requires_device_receipt_not_publish_success(monkeypatch):
    monkeypatch.setattr(
        "eidolon.livekit.agent.session.presentation.wait_for_runtime_participant_identity",
        AsyncMock(return_value="device"),
    )
    room = SimpleNamespace(
        on=Mock(), off=Mock(), local_participant=SimpleNamespace(publish_data=AsyncMock())
    )
    emit = Mock()
    transport = PresentationTransport(room, output(), emit)
    task = asyncio.create_task(transport.present(intent()))
    await asyncio.sleep(0)
    assert not task.done()
    envelope = json.loads(room.local_participant.publish_data.call_args.args[0])
    pid = envelope["id"]

    def receipt(status, seq, peer="device", response="response:turn-1"):
        transport.receive(
            SimpleNamespace(
                topic=CONTROL_TOPIC,
                participant=SimpleNamespace(identity=peer),
                data=json.dumps(
                    {
                        "op": "expression.play",
                        "ref": pid,
                        "result": {
                            "presentation_id": pid,
                            "response_id": response,
                            "status": status,
                            "sequence": seq,
                        },
                    }
                ).encode(),
            )
        )

    receipt("completed", 3, peer="other")
    receipt("completed", 3, response="old")
    assert not task.done()
    receipt("accepted", 1)
    receipt("started", 2)
    receipt("accepted", 1)
    assert not task.done()
    receipt("completed", 3)
    await task
    assert [c.args[0] for c in emit.call_args_list] == [
        "brain_presentation_sent",
        "brain_presentation_accepted",
        "brain_presentation_started",
        "brain_presentation_completed",
    ]
    assert not transport.pending
    await transport.close()


@pytest.mark.asyncio
async def test_cancellation_uses_existing_control_topic(monkeypatch):
    monkeypatch.setattr(
        "eidolon.livekit.agent.session.presentation.wait_for_runtime_participant_identity",
        AsyncMock(return_value="device"),
    )
    room = SimpleNamespace(
        on=Mock(), off=Mock(), local_participant=SimpleNamespace(publish_data=AsyncMock())
    )
    transport = PresentationTransport(room, output(), Mock())
    task = asyncio.create_task(transport.present(intent()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    envelope = json.loads(room.local_participant.publish_data.call_args.args[0])
    assert envelope["op"] == "expression.cancel"
    assert not transport.pending
    await transport.close()


def test_output_coordinator_does_not_mark_completed_expression_as_silent_failure():
    from eidolon.livekit.agent.observability import TurnTimeline

    timeline = TurnTimeline(turn_id="turn-1")
    coordinator = AgentOutputCoordinator()
    coordinator.record_brain_event(
        timeline, {"event": "brain_presentation_completed", "response_id": "r"}
    )
    state = coordinator.record_brain_event(timeline, {"event": "brain_done"})
    assert state["outcome"] == "presentation_completed" and not state["silent_failure"]


@pytest.mark.asyncio
async def test_real_grpc_feedback_finishes_silent_turn_without_text(monkeypatch):
    from .test_eidolon_agent_grpc_llm import _serve, _ctx, pb, pbg
    from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm

    class Servicer(pbg.EidolonAgentServicer):
        feedback = None

        async def Chat(self, requests, context):
            async for frame in requests:
                if frame.WhichOneof("payload") == "start":
                    tid = frame.start.turn_id
                    assert (
                        frame.start.metadata.fields["presentation_profile"].string_value
                        == FACE_PROFILE
                    )
                    yield pb.TurnEvent(
                        turn_id=tid,
                        seq=1,
                        kind=pb.TurnEvent.PRESENTATION,
                        presentation=pb.ResponseIntent(
                            schema_version=1,
                            response_id=f"response:{tid}",
                            turn_id=tid,
                            session_id="session-1",
                            intent="acknowledge",
                            stance="neutral",
                            intensity=0.3,
                            pace="normal",
                        ),
                    )
                elif frame.WhichOneof("payload") == "presentation_feedback":
                    self.feedback = frame.presentation_feedback
                    assert self.feedback.receipt.status == "completed"
                    yield pb.TurnEvent(turn_id=self.feedback.turn_id, seq=2, kind=pb.TurnEvent.DONE)

    monkeypatch.setattr(
        "eidolon.livekit.agent.session.presentation.wait_for_runtime_participant_identity",
        AsyncMock(return_value="device"),
    )
    room = SimpleNamespace(on=Mock(), off=Mock())

    async def publish(data, **kwargs):
        envelope = json.loads(data)
        plan = envelope["payload"]["plan"]
        adapter.presentation_transport.receive(
            SimpleNamespace(
                topic=CONTROL_TOPIC,
                participant=SimpleNamespace(identity="device"),
                data=json.dumps(
                    {
                        "op": "expression.play",
                        "ref": envelope["id"],
                        "result": {
                            "presentation_id": plan["presentation_id"],
                            "response_id": plan["response_id"],
                            "status": "completed",
                            "sequence": 3,
                        },
                    }
                ).encode(),
            )
        )

    room.local_participant = SimpleNamespace(publish_data=publish)
    servicer = Servicer()
    server, target = await _serve(servicer)
    adapter = EidolonAgentGrpcLlm(
        target=target,
        device_token=lambda: "test-token",
        conversation_id="session-1",
        output_plan=output(),
        presentation_room=room,
    )
    try:
        chunks = [chunk async for chunk in adapter.chat(chat_ctx=_ctx("你好"))]
        assert not [chunk for chunk in chunks if chunk.delta and chunk.delta.content]
        assert servicer.feedback.receipt.sequence == 3
    finally:
        await adapter.aclose()
        await server.stop(grace=0.5)
