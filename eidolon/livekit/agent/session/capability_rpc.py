"""Route declared session capabilities over native LiveKit RPC."""

import asyncio
import hashlib
import logging
import time

from eidolon_sdk.biz.presentation.capability_rpc import (
    CANCEL,
    INVOKE,
    INVOKE_V2,
    REGISTRY_PROFILE,
    STATUS,
    CapabilityInvocation,
    ExecutionScope,
    ExecutionSnapshot,
    RegisteredInvocation,
)

from ..eidolon_agent_rpc.channel_tools import ChannelToolResult
from ..runtime.resolver import wait_for_runtime_participant_identity


class SessionCapabilityRPC:
    def __init__(self, room, plan):
        self.room, self.plan = room, plan
        self.pending = {}
        self.executions = {}
        self.closed = False
        self.foreground = set()
        self._resource_hints = None
        self.receipts = []
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
        task = asyncio.current_task()
        self.foreground.add(task)
        try:
            retiring = [work for _, work, stop in tuple(self.executions.values()) if stop.is_set()]
            if retiring:
                await asyncio.gather(*retiring, return_exceptions=True)
            return await self._execute(call)
        finally:
            self.foreground.discard(task)

    async def _execute(self, call):
        if self.closed or not self.plan.outputs.motion or call.name not in self.plan.motion_capabilities:
            return ChannelToolResult(
                call.turn_id, call.call_id, call.name, False, {}, error_code="UNAVAILABLE"
            )
        peer = await asyncio.wait_for(wait_for_runtime_participant_identity(self.room), 3)
        identifier = self.identifier(call)
        self.pending[identifier] = peer
        background = False
        arguments = dict(call.arguments)
        definition = self.registry.get(call.name)
        wait_for_completion = arguments.pop("wait_for_completion", False) if definition.supports_background else True
        try:
            request = CapabilityInvocation(
                **self.scope(call).model_dump(),
                capability=call.name,
                arguments=arguments,
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
            if not wait_for_completion and result.state in {"accepted", "running"}:
                stop = asyncio.Event()
                task = asyncio.create_task(self._follow(call, peer, result, stop))
                self.executions[identifier] = (call, task, stop)
                background = True
            else:
                result = await self._wait(call, peer, result)
            return ChannelToolResult(
                call.turn_id,
                call.call_id,
                call.name,
                result.state == "completed" or (background and result.state in {"accepted", "running"}),
                result.model_dump(mode="json"),
                error_code=result.error,
                outcome_state="accepted" if background else result.state,
            )
        except BaseException:
            try:
                await self.cancel(call)
            except Exception:
                pass  # Preserve the original failure; cancellation was not confirmed.
            raise
        finally:
            if not background:
                self.pending.pop(self.identifier(call), None)

    async def cancel(self, call):
        peer = self.pending.get(self.identifier(call))
        if peer is not None:
            await self.rpc(peer, CANCEL, self.scope(call))

    async def _wait(self, call, peer, result):
        async with asyncio.timeout(35):
            while result.state in {"accepted", "running"}:
                await asyncio.sleep(0.15)
                result = await self.rpc(peer, STATUS, self.scope(call))
        return result

    async def _follow(self, call, peer, result, stop):
        waiting = asyncio.create_task(self._wait(call, peer, result))
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((waiting, stopping), return_when=asyncio.FIRST_COMPLETED)
            if stopping in done:
                result = await self.rpc(peer, CANCEL, self.scope(call))
            else:
                result = await waiting
            self.receipts.append({"name": call.name, "turn_id": call.turn_id,
                "call_id": call.call_id, **result.model_dump(mode="json")})
            self.receipts = self.receipts[-16:]
            logging.getLogger(__name__).info(
                "background capability receipt turn=%s execution=%s state=%s error=%s",
                call.turn_id, result.execution_id, result.state, result.error)
        except Exception:
            logging.getLogger(__name__).exception("background capability failed execution=%s", self.identifier(call))
            try:
                await self.cancel(call)
            except Exception:
                logging.getLogger(__name__).exception("background cancellation unconfirmed execution=%s", self.identifier(call))
        finally:
            waiting.cancel()
            stopping.cancel()
            await asyncio.gather(waiting, stopping, return_exceptions=True)
            self.pending.pop(self.identifier(call), None)
            self.executions.pop(self.identifier(call), None)

    def interrupt(self, turn_id=None):
        for call, _, stop in tuple(self.executions.values()):
            if turn_id is None or call.turn_id == turn_id:
                stop.set()

    async def close(self):
        self.closed = True
        foreground = list(self.foreground)
        for task in foreground:
            if not task.cancelling():
                task.cancel()
        self.interrupt()
        await asyncio.gather(*foreground, return_exceptions=True)
        # No foreground invocation can hand off a new accepted execution now.
        tasks = [task for _, task, _ in self.executions.values()]
        self.interrupt()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def resource_hints(self):
        """Session-local, read-only discovery; no model round trip or new protocol."""
        if self._resource_hints is not None:
            return self._resource_hints
        from ..eidolon_agent_rpc.channel_tools import ChannelToolCall
        hints = {}
        try:
            async with asyncio.timeout(2):
                for name in self.plan.motion_capabilities:
                    definition = self.registry.get(name)
                    if definition.resource_preview_arguments is None or definition.side_effect:
                        continue
                    result = await self.execute(ChannelToolCall("resource-preview", name, name,
                        dict(definition.resource_preview_arguments)))
                    if result.ok:
                        value = result.content
                        for key in definition.reference_result_path:
                            value = value.get(key) if isinstance(value, dict) else None
                        if value:
                            hints[name] = value
        except Exception:
            logging.getLogger(__name__).warning("resource preview unavailable; ordinary discovery remains available")
        # A transient failure must not poison later turns.
        if hints:
            self._resource_hints = hints
        return hints
