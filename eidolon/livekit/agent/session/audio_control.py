"""Prepared audio endpoint's existing room command/receipt channel.

This contains no scenario arbitration or microphone policy. Room, device,
conversation and output revision come from the authorized session preparation.
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from uuid import uuid4
from eidolon_sdk.biz.contracts import CONTROL_TOPIC, CONTROL_OP_PLAYBACK_STOP
from eidolon_sdk.biz.control.audio_presentation import (
    AudioPresentation,
    AudioPresentationResult,
    CONTROL_OP_PLAYBACK_PRESENT,
)
from eidolon_sdk.biz.control.protocol import command_status_from_ack
from eidolon.livekit.control_receipts import read_control_receipt
from eidolon.livekit.agent.session.client_control import build_session_client_control_envelope


@dataclass
class Exchange:
    op: str
    accepted: asyncio.Future
    completed: asyncio.Future
    speaking: object = None
    did_accept: bool = False
    did_start: bool = False


class DevicePlaybackControl:
    def __init__(self, room, *, device_id: str, conversation_id: str, policy_revision: int):
        if (
            not device_id
            or not conversation_id
            or type(policy_revision) is not int
            or policy_revision < 1
        ):
            raise ValueError("prepared output scope is required")
        self.room = room
        self.device_id = device_id
        self.conversation_id = conversation_id
        self.policy_revision = policy_revision
        self.pending: dict[str, Exchange] = {}
        self.closed = False
        room.on("data_received", self.receive)
        room.on("disconnected", self.disconnect)
        room.on("participant_disconnected", self.peer_left)

    async def _send(self, op, payload, speaking=None):
        if self.closed or len(self.pending) >= 32:
            raise ConnectionError("device control unavailable")
        if op != CONTROL_OP_PLAYBACK_STOP and any(
            item.op == op and not item.completed.done() for item in self.pending.values()
        ):
            raise RuntimeError("audio presentation already active")
        loop = asyncio.get_running_loop()
        command_id = uuid4().hex
        exchange = Exchange(op, loop.create_future(), loop.create_future(), speaking)
        self.pending[command_id] = exchange

        # One bounded exchange survives the accepted ACK until final hardware
        # completion; do not unregister when the preparation await returns.
        def finished(future):
            self.pending.pop(command_id, None)
            if not exchange.accepted.done():
                exchange.accepted.cancel()
            if not future.cancelled():
                future.exception()

        exchange.completed.add_done_callback(finished)
        envelope = build_session_client_control_envelope(
            op=op, reason="audio_presentation", payload=payload
        )
        envelope.update(id=command_id, capability_version=1)
        try:
            async with asyncio.timeout(2):
                await self.room.local_participant.publish_data(
                    json.dumps(envelope).encode(),
                    topic=CONTROL_TOPIC,
                    reliable=True,
                    destination_identities=[self.device_id],
                )
                await exchange.accepted
            return exchange
        except BaseException:
            exchange.completed.cancel()
            if exchange.accepted.done() and not exchange.accepted.cancelled():
                exchange.accepted.exception()
            raise

    async def prepare(self, request: AudioPresentation, speaking):
        # Device conversation id scopes the temporary output policy. The group
        # scene id remains inside presentation; the two scopes are not conflated.
        exchange = await self._send(
            CONTROL_OP_PLAYBACK_PRESENT,
            {
                "session_id": self.conversation_id,
                "policy_revision": self.policy_revision,
                "presentation": request.model_dump(mode="json"),
            },
            speaking,
        )

        async def completed():
            try:
                async with asyncio.timeout(125):
                    body = await exchange.completed
                result = AudioPresentationResult.model_validate(body.get("result"))
                if not result.confirms(request, result.rendered_bytes):
                    raise ValueError("device completed another presentation")
                return result
            finally:
                exchange.completed.cancel()

        # Started now, so caller cancellation cannot leak a pending exchange by
        # dropping an un-awaited coroutine after the accepted ACK.
        task = asyncio.create_task(completed())
        task.add_done_callback(lambda _: exchange.completed.cancel())
        return task

    async def stop(self) -> bool:
        exchange = await self._send(
            CONTROL_OP_PLAYBACK_STOP,
            {
                "session_id": self.conversation_id,
                "policy_revision": self.policy_revision,
            },
        )
        try:
            async with asyncio.timeout(2):
                await exchange.completed
            return True
        finally:
            exchange.completed.cancel()

    def receive(self, packet):
        body = read_control_receipt(packet, device=self.device_id, max_bytes=4096)
        if body is None:
            return
        exchange = self.pending.get(body["ref"])
        if exchange is None or body.get("op") != exchange.op or exchange.completed.done():
            return
        status = command_status_from_ack(body["status"])
        if status == "accepted":
            exchange.did_accept = True
            if not exchange.accepted.done():
                exchange.accepted.set_result(None)
            return
        if status == "running":
            if exchange.did_accept and exchange.speaking and not exchange.did_start:
                exchange.did_start = True
                exchange.speaking()
            return
        if status == "succeeded" and (
            exchange.did_accept or exchange.op == CONTROL_OP_PLAYBACK_STOP
        ):
            if not exchange.accepted.done():
                exchange.accepted.set_result(None)
            exchange.completed.set_result(body)
            return
        error = RuntimeError("device rejected or failed playback control")
        if not exchange.accepted.done():
            exchange.accepted.set_exception(error)
        exchange.completed.set_exception(error)

    def peer_left(self, participant):
        if getattr(participant, "identity", None) == self.device_id:
            self.disconnect()

    def disconnect(self, *_):
        self.closed = True
        for exchange in tuple(self.pending.values()):
            error = ConnectionError("device playback channel disconnected")
            if not exchange.accepted.done():
                exchange.accepted.set_exception(error)
            if not exchange.completed.done():
                exchange.completed.set_exception(error)

    def close(self):
        self.disconnect()
        self.room.off("data_received", self.receive)
        self.room.off("disconnected", self.disconnect)
        self.room.off("participant_disconnected", self.peer_left)
