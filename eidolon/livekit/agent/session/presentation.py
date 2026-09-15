"""Compile semantic responses and deliver them through existing room control.

This adapter owns transport futures only. AgentOutputCoordinator owns the turn's
observable output lifecycle; the device owns visual recipes and frame completion.
"""

from __future__ import annotations
import asyncio
import json
from typing import Any, Callable
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon_sdk.biz.presentation import (
    CancelExpression,
    EXPRESSION_CANCEL_OP,
    EXPRESSION_PLAY_OP,
    ExpressionPlan,
    ExpressionStep,
    PlayExpression,
    PresentationReceipt,
    ResponseIntent,
    SessionOutputPlan,
)
from ..runtime.resolver import wait_for_runtime_participant_identity
from .client_control import build_session_client_control_envelope

_GESTURES = {
    "acknowledge": "attend",
    "confirm": "affirm",
    "consider": "ponder",
    "clarify": "question",
    "comfort": "soften",
    "celebrate": "delight",
    "decline": "hesitate",
    "notify": "attention",
}


def compile_expression(intent: ResponseIntent, output: SessionOutputPlan) -> ExpressionPlan | None:
    if intent.session_id != output.session_id or not output.outputs.expression:
        raise ValueError("PRESENTATION_OUTSIDE_SELECTED_SESSION")
    if intent.intent == "none":
        return None
    duration = {"gentle": 1800, "normal": 1200, "brisk": 800}[intent.pace]
    return ExpressionPlan(
        presentation_id=f"face:{intent.turn_id}",
        response_id=intent.response_id,
        max_duration_ms=duration,
        steps=(
            ExpressionStep(
                gesture=_GESTURES[intent.intent],
                variant="subtle" if intent.stance in {"calm", "warm"} else "default",
                intensity=intent.intensity,
                duration_ms=duration,
            ),
        ),
    )


class PresentationTransport:
    def __init__(self, room: Any, output: SessionOutputPlan, emit: Callable[..., None]):
        self.room, self.output, self.emit = room, output, emit
        self.pending: dict[str, tuple[ExpressionPlan, asyncio.Future, int]] = {}
        self.peer = ""
        room.on("data_received", self.receive)

    async def present(self, intent: ResponseIntent) -> PresentationReceipt | None:
        plan = compile_expression(intent, self.output)
        if plan is None:
            self.emit(
                "brain_presentation_none", turn_id=intent.turn_id, response_id=intent.response_id
            )
            return
        if self.pending:
            raise ValueError("PRESENTATION_BUSY")
        self.peer = await wait_for_runtime_participant_identity(self.room)
        payload = PlayExpression(
            session_id=self.output.session_id,
            policy_revision=self.output.policy_revision,
            plan=plan,
        )
        envelope = build_session_client_control_envelope(
            op=EXPRESSION_PLAY_OP,
            reason="response",
            turn_id=intent.turn_id,
            payload=payload.model_dump(mode="json"),
        )
        envelope.update(id=plan.presentation_id, capability_version=1)
        future = asyncio.get_running_loop().create_future()
        self.pending[plan.presentation_id] = (plan, future, 0)
        try:
            await self.room.local_participant.publish_data(
                json.dumps(envelope).encode(),
                topic=CONTROL_TOPIC,
                reliable=True,
                destination_identities=[self.peer],
            )
            self.emit(
                "brain_presentation_sent", turn_id=intent.turn_id, response_id=intent.response_id
            )
            receipt = await asyncio.wait_for(future, timeout=(plan.max_duration_ms + 2500) / 1000)
            return receipt
        except (asyncio.CancelledError, TimeoutError):
            await self.cancel(plan)
            raise
        finally:
            self.pending.pop(plan.presentation_id, None)

    async def cancel(self, plan: ExpressionPlan) -> None:
        payload = CancelExpression(
            session_id=self.output.session_id,
            policy_revision=self.output.policy_revision,
            presentation_id=plan.presentation_id,
        )
        envelope = build_session_client_control_envelope(
            op=EXPRESSION_CANCEL_OP, reason="cancelled", payload=payload.model_dump(mode="json")
        )
        envelope["capability_version"] = 1
        await asyncio.wait_for(
            self.room.local_participant.publish_data(
                json.dumps(envelope).encode(),
                topic=CONTROL_TOPIC,
                reliable=True,
                destination_identities=[self.peer],
            ),
            timeout=1,
        )

    def receive(self, packet: Any) -> None:
        if (
            packet.topic != CONTROL_TOPIC
            or getattr(getattr(packet, "participant", None), "identity", None) != self.peer
        ):
            return
        try:
            if len(packet.data) > 4096:
                return
            envelope = json.loads(packet.data)
            entry = self.pending.get(envelope.get("ref"))
            if entry is None or envelope.get("op") != EXPRESSION_PLAY_OP:
                return
            plan, future, sequence = entry
            if future.done():
                return
            if envelope.get("status") in {"rejected", "expired", "unsupported"}:
                future.set_exception(ValueError("PRESENTATION_REJECTED"))
                return
            receipt = PresentationReceipt.model_validate(envelope["result"])
            if (
                receipt.presentation_id != plan.presentation_id
                or receipt.response_id != plan.response_id
                or receipt.sequence <= sequence
            ):
                return
            self.pending[plan.presentation_id] = (plan, future, receipt.sequence)
            self.emit(
                f"brain_presentation_{receipt.status}",
                response_id=plan.response_id,
                turn_id=plan.presentation_id.removeprefix("face:"),
                receipt=receipt.model_dump(mode="json"),
            )
            if receipt.status in {"completed", "cancelled", "rejected", "failed"}:
                future.set_result(receipt)
        except (ValueError, KeyError, TypeError):
            return

    async def close(self) -> None:
        self.room.off("data_received", self.receive)
        for _, future, _ in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()
