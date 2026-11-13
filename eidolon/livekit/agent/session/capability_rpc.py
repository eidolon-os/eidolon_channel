"""Route declared session capabilities over native LiveKit RPC."""

import asyncio
import hashlib
import json
import logging
import time

from eidolon_sdk.biz.capabilities.camera import (
    ATTRIBUTE_EXECUTION,
    ATTRIBUTE_POLICY_REVISION,
    ATTRIBUTE_SESSION,
    FRAME_MEDIA_TYPE,
    FRAME_TOPIC,
    MAX_FRAME_BYTES,
)
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

# A device registers its capability handlers once its hardware is ready, which
# can be seconds after it joins (Reachy wakes first); discovery asks again until
# it answers, within a bound for a device that never will.
RESOURCE_RETRY_S = 0.5
RESOURCE_DISCOVERY_S = 30.0
# The device sends a still before it reports completion; this covers the rest
# of the transfer when the receipt overtakes the last chunks.
FRAME_WAIT_S = 3.0


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
        # One set on both sides: the device binds every invocation to its revision.
        self.names = tuple(dict.fromkeys((*plan.motion_capabilities, *plan.device_capabilities)))
        self.frames = {}
        self._frame_readers = set()
        self._receives_frames = any(
            self.registry.get(name).required_input == "camera" for name in plan.device_capabilities)
        if self._receives_frames:
            room.unregister_byte_stream_handler(FRAME_TOPIC)  # a previous session of this room
            room.register_byte_stream_handler(FRAME_TOPIC, self._on_frame)

    def _allowed(self, name):
        return (name in self.plan.device_capabilities
                or (self.plan.outputs.motion and name in self.plan.motion_capabilities))

    def _on_frame(self, reader, participant_identity):
        task = asyncio.create_task(self._read_frame(reader, participant_identity))
        self._frame_readers.add(task)
        task.add_done_callback(self._frame_readers.discard)

    async def _read_frame(self, reader, participant_identity):
        """Keep a still only for an execution this session is waiting on."""
        attributes = dict(reader.info.attributes or {})
        identifier = attributes.get(ATTRIBUTE_EXECUTION, "")
        try:
            if (
                self.closed
                or identifier not in self.frames
                or self.pending.get(identifier) != participant_identity
                or attributes.get(ATTRIBUTE_SESSION) != self.plan.session_id
                or attributes.get(ATTRIBUTE_POLICY_REVISION) != str(self.plan.policy_revision)
                or (reader.info.size or 0) > MAX_FRAME_BYTES
            ):
                logging.getLogger(__name__).warning(
                    "camera frame dropped: no matching execution sender=%s execution=%s",
                    participant_identity, identifier)
                return
            chunks, size = [], 0
            async for chunk in reader:
                size += len(chunk)
                if size > MAX_FRAME_BYTES:
                    logging.getLogger(__name__).warning(
                        "camera frame dropped: oversized execution=%s", identifier)
                    return
                chunks.append(chunk)
            future = self.frames.get(identifier)  # gone once the execution stopped waiting
            if future is not None and not future.done():
                future.set_result(b"".join(chunks))
        except Exception:
            logging.getLogger(__name__).exception("camera frame read failed execution=%s", identifier)
        finally:
            await reader.aclose()

    async def _attach_frame(self, call, result):
        """The still the device sent for this execution, checked against its receipt."""
        identifier = self.identifier(call)
        facts = result.result if isinstance(result.result, dict) else {}
        try:
            frame = await asyncio.wait_for(self.frames[identifier], FRAME_WAIT_S)
        except TimeoutError:
            frame = b""
        if (
            not frame
            or len(frame) != facts.get("bytes")
            or hashlib.sha256(frame).hexdigest() != facts.get("sha256")
        ):
            logging.getLogger(__name__).warning(
                "camera frame missing or altered execution=%s received=%d", identifier, len(frame))
            return ChannelToolResult(call.turn_id, call.call_id, call.name, False,
                result.model_dump(mode="json"), error_code="transfer_failed", outcome_state="failed")
        logging.getLogger(__name__).info(
            "camera frame received execution=%s bytes=%d size=%sx%s", identifier, len(frame),
            facts.get("width"), facts.get("height"))
        return ChannelToolResult(call.turn_id, call.call_id, call.name, True,
            result.model_dump(mode="json"), outcome_state=result.state,
            attachment=frame, attachment_media_type=facts.get("media_type") or FRAME_MEDIA_TYPE)

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
        if self.closed or not self._allowed(call.name):
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
        if definition.required_input == "camera":
            self.frames[identifier] = asyncio.get_running_loop().create_future()
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
                        self.names, conversation_id=self.plan.session_id,
                        policy_revision=self.plan.policy_revision)["revision"],
                )
                method = INVOKE_V2
            logging.getLogger(__name__).info(
                "capability invoke session=%s turn=%s call=%s execution=%s method=%s arguments=%s wait=%s",
                self.plan.session_id, call.turn_id, call.call_id, identifier, method,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True)[:4096], wait_for_completion)
            result = await self.rpc(peer, method, request)
            if not wait_for_completion and result.state in {"accepted", "running"}:
                stop = asyncio.Event()
                task = asyncio.create_task(self._follow(call, peer, result, stop))
                self.executions[identifier] = (call, task, stop)
                background = True
            else:
                result = await self._wait(call, peer, result)
            if definition.required_input == "camera" and result.state == "completed":
                return await self._attach_frame(call, result)
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
            self.frames.pop(identifier, None)
            if not background:
                self.pending.pop(self.identifier(call), None)

    async def cancel(self, call):
        peer = self.pending.get(self.identifier(call))
        if peer is not None:
            logging.getLogger(__name__).info("capability cancel turn=%s call=%s execution=%s",
                call.turn_id, call.call_id, self.identifier(call))
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
                "background capability receipt turn=%s execution=%s state=%s error=%s result=%s",
                call.turn_id, result.execution_id, result.state, result.error,
                json.dumps(result.result, ensure_ascii=True, sort_keys=True)[:4096])
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
        if self._receives_frames:
            self.room.unregister_byte_stream_handler(FRAME_TOPIC)
        readers = list(self._frame_readers)
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        discovery, self._resource_hints = self._resource_hints, None
        if discovery is not None:
            discovery.cancel()
            await asyncio.gather(discovery, return_exceptions=True)
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

    def prefetch_resource_hints(self):
        """Start discovery once per session so the first turn finds it done."""
        if self._resource_hints is None and not self.closed:
            self._resource_hints = asyncio.create_task(
                self._discover_resources(), name="device-resource-discovery")

    async def resource_hints(self):
        """Hints for a turn; waits on the session's one discovery for at most 2 s."""
        self.prefetch_resource_hints()
        discovery = self._resource_hints
        if discovery is None:
            return {}
        await asyncio.wait({discovery}, timeout=2)
        if not discovery.done():
            return {}  # Still waiting for the device; a later turn gets the result.
        hints = {} if discovery.cancelled() else discovery.result()
        if not hints and self._resource_hints is discovery:
            # A transient failure must not poison later turns.
            self._resource_hints = None
        return hints

    async def _discover_resources(self):
        """Session-local, read-only discovery; no model round trip or new protocol."""
        from ..eidolon_agent_rpc.channel_tools import ChannelToolCall
        loop = asyncio.get_running_loop()
        deadline = loop.time() + RESOURCE_DISCOVERY_S
        while True:
            hints = {}
            try:
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
                return hints
            except Exception:
                if self.closed or loop.time() >= deadline:
                    logging.getLogger(__name__).warning(
                        "resource preview unavailable; ordinary discovery remains available")
                    return {}
            await asyncio.sleep(RESOURCE_RETRY_S)
