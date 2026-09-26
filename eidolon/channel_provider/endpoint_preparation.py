"""Prepare an explicit set of standing channels using existing room controls.

Owns transport preparation only; callers own authorization, reservations and
activation of their scene's worker. No inference, media or conversation mode.
"""
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from eidolon_sdk.biz.control.protocol import build_command_envelope
from .contracts import BackendUnavailable, InvalidTransition
from .ports import ServingAction, ServingRequest


@dataclass
class EndpointPreparation:
    adapter: object
    handles: tuple[dict, ...]
    activate: Callable[[dict[str, str]], Awaitable[None]]
    state: str = "preparing"
    error: str = ""
    task: asyncio.Task | None = None
    command_ids: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    prepared: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    remove_observers: list = field(default_factory=list)

    def __post_init__(self):
        self._device_ids = {handle["device"] for handle in self.handles}
        if not self.handles or len(self._device_ids) != len(self.handles):
            raise InvalidTransition("preparation requires distinct endpoints")

    def request(self, device_id: str, request: ServingRequest) -> None:
        if device_id not in self._device_ids:
            raise InvalidTransition("device is outside this preparation")
        if self.state not in ("preparing", "ready"):
            raise InvalidTransition("preparation is no longer active")
        previous = self.sessions.get(device_id)
        if request.control_request_id is not None:
            if request.control_request_id != self.command_ids.get(device_id):
                raise InvalidTransition("request does not belong to this preparation")
        elif previous != request.conversation_id:
            raise InvalidTransition("request does not belong to this preparation")
        if request.action is ServingAction.STOP:
            if previous == request.conversation_id:
                self.stopped.set()
            return
        if self.state != "preparing" and previous != request.conversation_id:
            raise InvalidTransition("device belongs to an active directed conversation")
        if previous is not None and previous != request.conversation_id:
            raise InvalidTransition("device changed its pending conversation")
        self.sessions[device_id] = request.conversation_id
        if len(self.sessions) == len(self.handles):
            self.prepared.set()

    async def run(self) -> None:
        wakes = []
        try:
            # Checking precedes any wake. A private conversation is never
            # silently replaced by preparing a cross-device presentation.
            for handle in self.handles:
                await self.adapter.require_idle(handle)
            for handle in self.handles:
                command_id = f"join:{uuid4().hex}"
                self.command_ids[handle["device"]] = command_id
                command = build_command_envelope(command_id=command_id,
                    device_id=handle["device"], payload={"prepare_only": True}, op="room.join",
                    src_type="channel", src_id="channel-provider", ttl_ms=20_000)
                wakes.append(asyncio.create_task(self.adapter.deliver_control(
                    handle, command, wait_for_terminal=True)))
            async with asyncio.timeout(20):
                prepared = asyncio.create_task(self.prepared.wait())
                stopped = asyncio.create_task(self.stopped.wait())
                try:
                    done, _ = await asyncio.wait([prepared, stopped, *wakes],
                                                return_when=asyncio.FIRST_COMPLETED)
                    if stopped in done:
                        raise InvalidTransition("preparation was stopped")
                    for wake in wakes:
                        if wake in done:
                            # No device may complete room.join before the
                            # authorized worker starts the selected endpoint plans.
                            result = await wake
                            raise BackendUnavailable(f"device preparation ended early: {result}")
                    for handle in self.handles:
                        self.remove_observers.append(self.adapter.observe_session_end(
                            handle, self.sessions[handle["device"]], self.stopped.set))
                    if self.stopped.is_set():
                        raise InvalidTransition("preparation was stopped")
                    await self.activate(dict(self.sessions))
                    results = await asyncio.gather(*wakes)
                    if any(result != "succeeded" for result in results):
                        raise BackendUnavailable("device did not confirm its session plan")
                finally:
                    for waiter in (prepared, stopped):
                        waiter.cancel()
                    await asyncio.gather(prepared, stopped, return_exceptions=True)
            self.state = "ready"
            await self.stopped.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
        finally:
            for wake in wakes:
                wake.cancel()
            await asyncio.gather(*wakes, return_exceptions=True)
            if self.state != "failed":
                self.state = "closing"

    async def cleanup(self) -> None:
        # Keep failure visible; cleanup failure must retain the reservation so
        # a later close/revoke can retry rather than authorizing overlapping IO.
        results = await asyncio.gather(*(
            self.adapter.end_prepared_session(handle, self.sessions[handle["device"]])
            for handle in self.handles if handle["device"] in self.sessions
        ), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("endpoint cleanup failed", failures)
        for remove in self.remove_observers:
            remove()
        self.remove_observers.clear()
        if self.state != "failed":
            self.state = "closed"
