"""Route declared session capabilities over native LiveKit RPC."""

import asyncio
import hashlib
import time

from eidolon_sdk.biz.presentation.capability_rpc import (
    CANCEL,
    INVOKE, INVOKE_V2, RegisteredInvocation, REGISTRY_PROFILE,
    STATUS,
    CapabilityInvocation,
    ExecutionScope,
    ExecutionSnapshot,
)
from ..eidolon_agent_rpc.channel_tools import ChannelToolResult
from ..runtime.resolver import wait_for_runtime_participant_identity


class SessionCapabilityRPC:
    def __init__(self, room, plan):
        self.room, self.plan = room, plan
        self.pending = {}
        from eidolon_sdk.biz.capabilities import installed_registry
        self.registry = installed_registry()

    def identifier(self, call):
        return (
            "exec:"
            + hashlib.sha256(
                f"{self.plan.session_id}:{call.turn_id}:{call.call_id}".encode()
            ).hexdigest()[:32]
        )

    def scope(self, call):
        return ExecutionScope(
            execution_id=self.identifier(call),
            session_id=self.plan.session_id,
            policy_revision=self.plan.policy_revision,
        )

    async def rpc(self, peer, method, request):
        payload = await self.room.local_participant.perform_rpc(
            destination_identity=peer,
            method=method,
            payload=request.model_dump_json(),
            response_timeout=5,
        )
        result = ExecutionSnapshot.model_validate_json(payload)
        if result.execution_id != request.execution_id:
            raise ValueError("mismatched execution receipt")
        return result

    async def execute(self, call):
        if not self.plan.outputs.motion or call.name not in self.plan.motion_capabilities:
            return ChannelToolResult(
                call.turn_id, call.call_id, call.name, False, {}, error_code="UNAVAILABLE"
            )
        peer = await asyncio.wait_for(wait_for_runtime_participant_identity(self.room), 3)
        self.pending[self.identifier(call)] = peer
        try:
            request = CapabilityInvocation(
                **self.scope(call).model_dump(),
                capability=call.name,
                arguments=dict(call.arguments),
                expires_at_ms=int(time.time() * 1000) + 10000,
                max_duration_ms=30000,
            )
            method = INVOKE
            if self.plan.motion_profile == REGISTRY_PROFILE:
                definition = self.registry.get(call.name)
                request = RegisteredInvocation(
                    **request.model_dump(exclude={"protocol_version", "capability_version"}),
                    capability_version=definition.version,
                    contract_digest=definition.contract_digest,
                    binding_ref="device:" + self.plan.session_id,
                    capability_set_revision=self.registry.snapshot(
                        self.plan.motion_capabilities, conversation_id=self.plan.session_id,
                        policy_revision=self.plan.policy_revision)["revision"],
                )
                method = INVOKE_V2
            result = await self.rpc(peer, method, request)
            async with asyncio.timeout(35):
                while result.state in {"accepted", "running"}:
                    await asyncio.sleep(0.15)
                    result = await self.rpc(peer, STATUS, self.scope(call))
            return ChannelToolResult(
                call.turn_id,
                call.call_id,
                call.name,
                result.state == "completed",
                result.model_dump(mode="json"),
                error_code=result.error,
                outcome_state=result.state,
            )
        except BaseException:
            try:
                await self.cancel(call)
            except Exception:
                pass  # Preserve the original failure; cancellation was not confirmed.
            raise
        finally:
            self.pending.pop(self.identifier(call), None)

    async def cancel(self, call):
        peer = self.pending.get(self.identifier(call))
        if peer is not None:
            await self.rpc(peer, CANCEL, self.scope(call))
