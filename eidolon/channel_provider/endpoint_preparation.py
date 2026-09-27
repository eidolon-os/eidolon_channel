"""Prepare an explicit set of standing channels using existing room controls.

Owns transport preparation only; callers own authorization, reservations and
activation of their scene's worker. No inference, media or conversation mode.
"""
import asyncio
import logging
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
    _cleanup_failed: bool = field(default=False, init=False)
    _cleanup_complete: bool = field(default=False, init=False)
    close_task: asyncio.Task | None = field(default=None, init=False)
    checkpoint: Callable[[], None] = field(default_factory=lambda: (lambda: None), init=False)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _producers_revoked: bool = field(default=False, init=False)
    _ended_devices: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self._device_ids = {handle["device"] for handle in self.handles}
        if not self.handles or len(self._device_ids) != len(self.handles):
            raise InvalidTransition("preparation requires distinct endpoints")

    @property
    def closure_complete(self) -> bool:
        return self._cleanup_complete

    def lifecycle_checkpoint(self) -> dict:
        return dict(state=self.state, error=self.error, sessions=dict(self.sessions),
            command_ids=dict(self.command_ids), producers_revoked=self._producers_revoked,
            ended_devices=sorted(self._ended_devices), cleanup_failed=self._cleanup_failed,
            cleanup_complete=self._cleanup_complete)

    def restore_lifecycle(self, data: dict) -> None:
        sessions, commands = dict(data["sessions"]), dict(data["command_ids"])
        ended = set(data["ended_devices"])
        attempted = set(sessions) | set(commands)
        if not attempted <= self._device_ids or not ended <= attempted:
            raise InvalidTransition("closure checkpoint contains an unowned endpoint")
        if any(not isinstance(v, str) or not v for v in (*sessions.values(), *commands.values())):
            raise InvalidTransition("closure checkpoint lacks endpoint correlation")
        if data["cleanup_complete"] and (not data["producers_revoked"] or ended != attempted):
            raise InvalidTransition("closure checkpoint has no terminal proof")
        self.state, self.error = data["state"], data["error"]
        self.sessions, self.command_ids = sessions, commands
        self._producers_revoked, self._ended_devices = data["producers_revoked"], ended
        self._cleanup_failed, self._cleanup_complete = data["cleanup_failed"], data["cleanup_complete"]
        if not self._cleanup_complete:
            self.state = "closing"

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
        self.checkpoint()
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
                self.checkpoint()
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
            self.checkpoint()
            await self.stopped.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
            logging.getLogger(__name__).warning("endpoint preparation failed: %s", self.error)
        finally:
            for wake in wakes:
                wake.cancel()
            await asyncio.gather(*wakes, return_exceptions=True)
            if self.state != "failed":
                self.state = "closing"

    async def cleanup(self) -> None:
        """One monotonic close transaction, shared by stop, failure and retry.

        Producer revocation precedes Body revocation. Terminal device receipts
        are checkpoints: a retry never reopens a completed stage or asks an
        already-confirmed endpoint to be reachable again. Unknown stays unknown.
        The service releases the group reservation only when this returns.
        """
        async with self._close_lock:
            handles = tuple(handle for handle in self.handles
                if handle["device"] in self.sessions or handle["device"] in self.command_ids)
            try:
                if not self._producers_revoked:
                    await self.adapter.quiesce_prepared_sessions(handles, dict(self.sessions))
                    self._producers_revoked = True
                    self.checkpoint()
                pending = tuple(h for h in handles if h["device"] not in self._ended_devices)

                async def end(handle):
                    device = handle["device"]
                    await self.adapter.end_prepared_session(handle, self.sessions.get(device),
                        control_request_id=self.command_ids.get(device))
                    self._ended_devices.add(device)
                    self.checkpoint()

                results = await asyncio.gather(*(end(h) for h in pending), return_exceptions=True)
                failures = [(h["device"], r) for h, r in zip(pending, results)
                            if isinstance(r, BaseException)]
                if failures:
                    raise BackendUnavailable("endpoint cleanup unconfirmed: " +
                        ", ".join(device for device, _ in failures)) from BaseExceptionGroup(
                            "endpoint cleanup failed", [error for _, error in failures])
            except BaseException as exc:
                self._cleanup_failed = True
                self.state = "failed"
                self.error = str(exc) or "close interrupted; confirmation remains pending"
                self.checkpoint()
                logging.getLogger(__name__).warning(
                    "scene close incomplete producers_revoked=%s confirmed=%d pending=%d error=%s",
                    self._producers_revoked, len(self._ended_devices),
                    len(handles) - len(self._ended_devices), self.error)
                raise
            for remove in self.remove_observers:
                remove()
            self.remove_observers.clear()
            if self.state != "failed" or self._cleanup_failed:
                self.state = "closed"
                self.error = ""
            self._cleanup_failed = False
            self._cleanup_complete = True
            try:
                self.checkpoint()
            except Exception as exc:
                self._cleanup_complete = False
                self._cleanup_failed = True
                self.state = "failed"
                self.error = "close checkpoint could not be committed"
                raise BackendUnavailable(self.error) from exc
