"""Compile semantic responses and deliver them through existing room control.

This adapter owns transport futures only. AgentOutputCoordinator owns the turn's
observable output lifecycle; the device owns visual recipes and frame completion.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable
from contextlib import suppress
from typing import Any, Callable

from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon_sdk.biz.presentation import (
    EXPRESSION_CANCEL_OP,
    EXPRESSION_PLAY_OP,
    CancelExpression,
    ExpressionPlan,
    ExpressionStep,
    PlayExpression,
    PresentationReceipt,
    ResponseIntent,
    SessionOutputPlan,
)
from eidolon_sdk.biz.presentation.motion import (
    STACKCHAN_HEAD_PROFILE,
    MotionReceipt,
    MotionRequest,
)

from ..runtime.resolver import wait_for_runtime_participant_identity
from .client_control import build_session_client_control_envelope

logger = logging.getLogger(__name__)

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
        self.motion_pending: dict[str, asyncio.Future] = {}
        self._motion_ops: dict[str, str] = {}
        self._motion_turns: dict[str, str] = {}
        self._executions: dict[str, asyncio.Task] = {}
        self._explicit_turn = ""
        self._last_motion_id = ""
        self.peer = ""
        self._tasks: dict[str, asyncio.Task] = {}
        self._closed = False
        room.on("data_received", self.receive)

    def start(
        self, intent: ResponseIntent, report: Callable[[str, PresentationReceipt], Awaitable[None]]
    ) -> None:
        """Own one bounded delivery per response, independently of LLM/TTS.

        Generation DONE must not cancel delivery. Only a new response, explicit
        interruption or session teardown does. Draining the previous task keeps
        cancellation ordered before the next expression.play on the same peer.
        """
        if self._closed or intent.turn_id in self._tasks:
            return
        previous = (*self._tasks.values(), *self._executions.values())
        self.interrupt()

        async def deliver() -> None:
            try:
                if previous:
                    await asyncio.gather(*previous, return_exceptions=True)
                # Independent lanes share one response owner and cancellation.
                # A failed head never suppresses an otherwise usable face.
                if (
                    self.output.outputs.motion
                    and not self.output.outputs.expression
                    and intent.turn_id != self._explicit_turn
                ):
                    receipt = await self.present_motion(intent)
                elif (
                    self.output.outputs.motion
                    and intent.intent != "none"
                    and intent.turn_id != self._explicit_turn
                ):
                    face, _ = await asyncio.gather(
                        self.present(intent), self.present_motion(intent), return_exceptions=True)
                    if isinstance(face, BaseException):
                        raise face
                    receipt = face
                elif self.output.outputs.expression:
                    receipt = await self.present(intent)
                else:
                    receipt = None
            except asyncio.CancelledError:
                receipt = self._receipt(intent, "cancelled", "RESPONSE_CANCELLED")
                self._emit_receipt(intent, receipt)
            except Exception:
                logger.exception("expression delivery failed turn=%s", intent.turn_id)
                receipt = self._receipt(intent, "failed", "PRESENTATION_TRANSPORT_FAILED")
                self._emit_receipt(intent, receipt)
            if receipt is not None:
                # Reporting a delivery failure must not turn into a model error.
                try:
                    await asyncio.wait_for(report(intent.turn_id, receipt), timeout=1)
                except Exception:
                    logger.warning(
                        "presentation feedback unavailable turn=%s", intent.turn_id, exc_info=True
                    )

        task = asyncio.create_task(deliver(), name=f"expression-{intent.turn_id}")
        self._tasks[intent.turn_id] = task
        task.add_done_callback(lambda done: self._tasks.pop(intent.turn_id, None))

    def interrupt(self, turn_id: str | None = None) -> None:
        for tid, task in (*tuple(self._tasks.items()), *tuple(self._executions.items())):
            if turn_id is None or tid == turn_id:
                if not task.done() and not task.cancelling():
                    task.cancel()

    def _receipt(self, intent: ResponseIntent, status: str, reason: str) -> PresentationReceipt:
        lane = "face" if self.output.outputs.expression else "head"
        return PresentationReceipt(
            presentation_id=f"{lane}:{intent.turn_id}",
            response_id=intent.response_id,
            status=status,
            sequence=1,
            reason=reason,
        )

    def _emit_receipt(self, intent: ResponseIntent, receipt: PresentationReceipt) -> None:
        self.emit(
            f"brain_presentation_{receipt.status}",
            turn_id=intent.turn_id,
            response_id=intent.response_id,
            receipt=receipt.model_dump(mode="json"),
        )

    async def present(self, intent: ResponseIntent) -> PresentationReceipt | None:
        plan = compile_expression(intent, self.output)
        if plan is None:
            self.emit(
                "brain_presentation_none", turn_id=intent.turn_id, response_id=intent.response_id
            )
            return
        if self.pending:
            raise ValueError("PRESENTATION_BUSY")
        self.peer = await asyncio.wait_for(
            wait_for_runtime_participant_identity(self.room), timeout=2.5
        )
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
            await asyncio.wait_for(
                self.room.local_participant.publish_data(
                    json.dumps(envelope).encode(),
                    topic=CONTROL_TOPIC,
                    reliable=True,
                    destination_identities=[self.peer],
                ),
                timeout=1,
            )
            self.emit(
                "brain_presentation_sent", turn_id=intent.turn_id, response_id=intent.response_id
            )
            receipt = await asyncio.wait_for(future, timeout=(plan.max_duration_ms + 2500) / 1000)
            return receipt
        except asyncio.CancelledError:
            with suppress(Exception):
                await self.cancel(plan)
            raise
        except TimeoutError:
            with suppress(Exception):
                await self.cancel(plan)
            receipt = self._receipt(intent, "failed", "PRESENTATION_TIMEOUT")
            self._emit_receipt(intent, receipt)
            return receipt
        finally:
            self.pending.pop(plan.presentation_id, None)

    async def present_motion(self, intent: ResponseIntent) -> PresentationReceipt | None:
        if intent.session_id != self.output.session_id or not self.output.outputs.motion:
            raise ValueError("MOTION_OUTSIDE_SELECTED_SESSION")
        # Hardware gesture vocabulary already exposed by the head.gesture action.
        # No arbitrary servo coordinates or per-frame commands from the model.
        gestures = {"acknowledge": "perk_up", "confirm": "nod", "consider": "glance",
                    "clarify": "glance", "comfort": "droop", "celebrate": "wake_wobble",
                    "decline": "shake", "notify": "perk_up"}
        name = gestures.get(intent.intent)
        if name is None:
            if not self.output.outputs.expression:
                self.emit("brain_presentation_none", turn_id=intent.turn_id, response_id=intent.response_id)
            return
        if intent.turn_id == self._explicit_turn:
            return None
        receipt = await self._deliver_motion(
            command_id=f"head:{intent.turn_id}",
            turn_id=intent.turn_id,
            payload={"name": name, "times": 1, "hold_ms": 350, "return_ms": 350},
        )
        result = self._receipt(intent, receipt.status, receipt.reason)
        if not self.output.outputs.expression:
            self._emit_receipt(intent, result)
        return result

    async def execute_motion(self, request: MotionRequest) -> MotionReceipt:
        """Validated explicit tool lane; share the existing device executor/ACKs."""
        self._explicit_turn = request.turn_id
        if (
            self._closed
            or not self.output.outputs.motion
            or self.output.motion_profile != STACKCHAN_HEAD_PROFILE
        ):
            return MotionReceipt(
                command_id=request.command_id, status="rejected", reason="MOTION_NOT_AVAILABLE"
            )
        if self._executions and request.action.action != "stop":
            return MotionReceipt(command_id=request.command_id, status="rejected", reason="BUSY")
        # Explicit actions supersede optional presentation, with ordered cancellation.
        previous = (*self._tasks.values(), *self._executions.values())
        self.interrupt()
        if previous:
            await asyncio.gather(*previous, return_exceptions=True)
        if request.action.action == "stop" and not self._last_motion_id:
            return MotionReceipt(
                command_id=request.command_id, status="rejected", reason="NO_KNOWN_MOTION"
            )
        stop = request.action.action == "stop"
        payload = {"motion_id": self._last_motion_id} if stop else request.action.gesture_payload()
        task = asyncio.create_task(
            self._deliver_motion(
                command_id=request.command_id,
                turn_id=request.turn_id,
                payload=payload,
                op="safety.stop" if stop else "head.gesture",
            )
        )
        self._executions[request.turn_id] = task
        try:
            return await task
        finally:
            self._executions.pop(request.turn_id, None)

    async def _deliver_motion(
        self, *, command_id: str, turn_id: str, payload: dict, op: str = "head.gesture"
    ) -> MotionReceipt:
        future = asyncio.get_running_loop().create_future()
        self.motion_pending[command_id] = future
        self._motion_ops[command_id] = op
        self._motion_turns[command_id] = turn_id
        payload = {
            **payload,
            "session_id": self.output.session_id,
            "policy_revision": self.output.policy_revision,
        }
        envelope = build_session_client_control_envelope(
            op=op, reason="response", turn_id=turn_id, payload=payload
        )
        envelope.update(id=command_id, capability_version=1)
        sent = False

        async def stop() -> None:
            if op != "head.gesture" or not sent:
                return
            cancel = build_session_client_control_envelope(op="safety.stop", reason="response_cancelled",
                payload={"motion_id": command_id, "session_id": self.output.session_id,
                         "policy_revision": self.output.policy_revision})
            cancel["capability_version"] = 1
            await asyncio.wait_for(self.room.local_participant.publish_data(
                json.dumps(cancel).encode(), topic=CONTROL_TOPIC, reliable=True,
                destination_identities=[self.peer]), timeout=1)

        try:
            self.peer = await asyncio.wait_for(wait_for_runtime_participant_identity(self.room), timeout=2.5)
            # Delivery may have occurred even if the publish await times out.
            sent = True
            await asyncio.wait_for(
                self.room.local_participant.publish_data(
                    json.dumps(envelope).encode(),
                    topic=CONTROL_TOPIC,
                    reliable=True,
                    destination_identities=[self.peer],
                ),
                timeout=1,
            )
            self.emit("brain_motion_sent", turn_id=turn_id, gesture=payload.get("name", "stop"))
            status, reason = await asyncio.wait_for(future, timeout=6)
            status = {"expired": "rejected", "unsupported": "rejected"}.get(status, status)
            basis = "drive_stopped" if op == "safety.stop" else "software_sequence"
            return MotionReceipt(
                command_id=command_id,
                status=status,
                reason=reason,
                completion_basis=basis if status == "completed" else "unconfirmed",
            )
        except asyncio.CancelledError:
            with suppress(Exception):
                await stop()
            self.emit("brain_motion_cancelled", turn_id=turn_id, reason="RESPONSE_CANCELLED")
            raise
        except Exception as error:
            with suppress(Exception):
                await stop()
            reason = (
                "MOTION_TIMEOUT" if isinstance(error, TimeoutError) else "MOTION_TRANSPORT_FAILED"
            )
            self.emit("brain_motion_failed", turn_id=turn_id, reason=reason)
            return MotionReceipt(command_id=command_id, status="failed", reason=reason)
        finally:
            self.motion_pending.pop(command_id, None)
            self._motion_ops.pop(command_id, None)
            self._motion_turns.pop(command_id, None)

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
            motion = self.motion_pending.get(envelope.get("ref"))
            if motion is not None and envelope.get("op") == self._motion_ops.get(
                envelope.get("ref")
            ):
                status = envelope.get("status")
                if not motion.done() and status in {
                    "accepted",
                    "started",
                    "completed",
                    "cancelled",
                    "rejected",
                    "failed",
                    "expired",
                    "unsupported",
                }:
                    self.emit(
                        f"brain_motion_{status}",
                        turn_id=self._motion_turns[envelope["ref"]],
                        reason=str(envelope.get("code") or "")[:96],
                    )
                    if (
                        status in {"accepted", "started", "completed"}
                        and envelope["op"] == "head.gesture"
                    ):
                        self._last_motion_id = envelope["ref"]
                    if status not in {"accepted", "started"}:
                        motion.set_result((status, str(envelope.get("code") or "")[:96]))
                return
            entry = self.pending.get(envelope.get("ref"))
            if entry is None or envelope.get("op") != EXPRESSION_PLAY_OP:
                return
            plan, future, sequence = entry
            if future.done():
                return
            if envelope.get("status") in {"rejected", "expired", "unsupported"}:
                # An ACK refusal is output evidence, not an LLM exception.
                # Preserve the device's reason instead of collapsing it.
                receipt = PresentationReceipt(
                    presentation_id=plan.presentation_id,
                    response_id=plan.response_id,
                    status="rejected",
                    sequence=sequence + 1,
                    reason=str(envelope.get("code") or envelope["status"])[:256],
                )
                self.emit(
                    "brain_presentation_rejected",
                    turn_id=plan.presentation_id.removeprefix("face:"),
                    response_id=plan.response_id,
                    receipt=receipt.model_dump(mode="json"),
                )
                future.set_result(receipt)
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
        self._closed = True
        self.interrupt()
        tasks = (*tuple(self._tasks.values()), *tuple(self._executions.values()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.room.off("data_received", self.receive)
        for _, future, _ in self.pending.values():
            if not future.done():
                future.cancel()
        self.pending.clear()
