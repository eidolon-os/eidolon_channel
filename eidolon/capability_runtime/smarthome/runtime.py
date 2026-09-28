"""Smart home execution and panel sync for every Owner on this Host.

This is where a command meets the Provider that owns the device, and where
every panel of that Owner learns what happened. It keeps no master data: the
registry is System Data's and is re-read when System Data says it changed,
and device state is whatever the Providers report. What it does own is order.
Per Owner, one execution or snapshot runs at a time, so a panel always sees a
snapshot's seq before the delta that follows it, and seq advances by exactly
one per delta.

It is also the one place a scene becomes commands: a request names a scene
and the runtime expands the scene's actions as the registry holds them now.

``owner_id`` and a panel's ``device_ref`` are always the caller's trusted
context (a runtime token, a channel binding), never anything a payload says.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from eidolon_sdk.biz.smarthome import (
    ERROR_DEADLINE_EXCEEDED,
    ERROR_DEVICE_OFFLINE,
    ERROR_UNKNOWN_DEVICE,
    ERROR_UNKNOWN_SCENE,
    OP_DELTA,
    OP_SNAPSHOT,
    ChangeSource,
    Command,
    CommandResult,
    ExecuteRequest,
    ExecuteResult,
    Origin,
    PanelArea,
    PanelChange,
    PanelDelta,
    PanelDevice,
    PanelExecute,
    PanelRequest,
    PanelScene,
    PanelSnapshot,
    PanelSync,
    Registry,
    SmartHomeError,
    validate_command,
    validate_state,
)

from pydantic import ValidationError

from .ports import PanelSink, RegistrySource, SmartHomeProvider

logger = logging.getLogger("eidolon.capability_runtime.smarthome")

IDEMPOTENCY_TTL_MS = 10 * 60_000
IDEMPOTENCY_CAPACITY = 1024
TOUCH_DEADLINE_MS = 3_000
PANEL_SEND_TIMEOUT_S = 5.0


class IdempotencyConflict(ValueError):
    """A request_id was reused for different commands or a different scene.

    Answering with the first result would report commands that never ran.
    """


def local_utc_offset_minutes(now_ms: int) -> int:
    """This Host's UTC offset at ``now_ms``, which the panel shows local time with."""
    offset = datetime.fromtimestamp(now_ms / 1000).astimezone().utcoffset()
    return int(offset.total_seconds() // 60) if offset is not None else 0


@dataclass(frozen=True, slots=True)
class _Recorded:
    expires_at_ms: int
    fingerprint: str
    result: ExecuteResult


@dataclass
class _Owner:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    registry: Registry | None = None
    # Last state each Provider reported, for devices a registered Provider serves.
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    seq: int = 0
    panels: dict[str, None] = field(default_factory=dict)  # insertion-ordered set
    recorded: OrderedDict[tuple[str, str], _Recorded] = field(default_factory=OrderedDict)


class SmartHomeRuntime:
    def __init__(
        self,
        *,
        registry: RegistrySource,
        panels: PanelSink,
        providers: Mapping[str, SmartHomeProvider],
        now_ms: Callable[[], int] | None = None,
        idempotency_ttl_ms: int = IDEMPOTENCY_TTL_MS,
        idempotency_capacity: int = IDEMPOTENCY_CAPACITY,
        touch_deadline_ms: int = TOUCH_DEADLINE_MS,
        utc_offset_minutes: Callable[[int], int] = local_utc_offset_minutes,
    ) -> None:
        self._registry_source = registry
        self._panels = panels
        self._providers = dict(providers)
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._ttl_ms = idempotency_ttl_ms
        self._capacity = idempotency_capacity
        self._touch_deadline_ms = touch_deadline_ms
        self._utc_offset_minutes = utc_offset_minutes
        self._owners: dict[str, _Owner] = {}

    # -- execution ---------------------------------------------------------

    async def execute(self, owner_id: str, request: ExecuteRequest) -> ExecuteResult:
        """Run each command independently and report each one honestly, in order.

        A repeat of ``request_id`` within the idempotency window returns the
        recorded result and runs nothing. ``deadline_ms`` is Unix epoch ms. A
        request already past it, or naming a scene the registry does not have,
        is refused whole before anything runs. A command reached after it is
        ``failed``/``DEADLINE_EXCEEDED``, having not been attempted; a Provider
        still busy at it is ``unknown``, since it may yet act.
        """
        return await self._execute(owner_id, request, scope="")

    async def handle_panel_execute(
        self, owner_id: str, device_ref: str, message: PanelExecute
    ) -> ExecuteResult:
        request = ExecuteRequest(
            request_id=message.request_id,
            commands=message.commands,
            scene_id=message.scene_id,
            origin=Origin(kind="touch", device_ref=device_ref),
            deadline_ms=self._now_ms() + self._touch_deadline_ms,
        )
        # A panel's request ids are only unique to that panel.
        return await self._execute(owner_id, request, scope=device_ref)

    async def _execute(
        self, owner_id: str, request: ExecuteRequest, *, scope: str
    ) -> ExecuteResult:
        key = (scope, request.request_id)
        fingerprint = _fingerprint(request)
        owner = self._owner(owner_id)
        async with owner.lock:
            recorded = self._lookup(owner, key, fingerprint)
            if recorded is not None:
                return recorded
            if request.deadline_ms <= self._now_ms():
                result = _refused(request, ERROR_DEADLINE_EXCEEDED)
                self._record(owner, key, fingerprint, result)
                return result
            registry = await self._registry(owner_id, owner)
            commands = request.commands
            if request.scene_id is not None:
                scene = registry.scene(request.scene_id)
                if scene is None:
                    result = _refused(request, ERROR_UNKNOWN_SCENE)
                    self._record(owner, key, fingerprint, result)
                    return result
                commands = scene.actions
            before: dict[str, dict[str, Any] | None] = {}
            results = []
            for command in commands:
                results.append(
                    await self._run(owner_id, owner, registry, command, request.deadline_ms, before)
                )
            result = ExecuteResult(request_id=request.request_id, results=tuple(results))
            # Recorded before anyone is told, so a failed send can never cause a rerun.
            self._record(owner, key, fingerprint, result)
            changes = tuple(
                PanelChange(device_id=device_id, state=owner.states[device_id])
                for device_id, previous in before.items()
                if owner.states[device_id] != previous
            )
            if changes:
                owner.seq += 1
                delta = PanelDelta(
                    revision=registry.revision,
                    seq=owner.seq,
                    source=ChangeSource(kind=request.origin.kind, label=request.origin.label),
                    changes=changes,
                )
                payload = delta.model_dump(mode="json")
                await self._broadcast(owner_id, owner, OP_DELTA, lambda _: payload)
            return result

    async def _run(
        self,
        owner_id: str,
        owner: _Owner,
        registry: Registry,
        command: Command,
        deadline_ms: int,
        before: dict[str, dict[str, Any] | None],
    ) -> CommandResult:
        remaining_ms = deadline_ms - self._now_ms()
        if remaining_ms <= 0:
            # Whoever asked has stopped waiting, and this one was never started.
            return _failed(command, ERROR_DEADLINE_EXCEEDED)
        # Another Owner's device is simply not in this registry.
        device = registry.device(command.device_id)
        if device is None:
            return _failed(command, ERROR_UNKNOWN_DEVICE)
        try:
            validate_command(device.type, command)
        except SmartHomeError as exc:
            return _failed(command, exc.code)
        provider = self._providers.get(device.provider)
        if provider is None:
            return _failed(command, ERROR_DEVICE_OFFLINE)
        try:
            async with asyncio.timeout(remaining_ms / 1000):
                state = await provider.execute(owner_id, device, command)
            validate_state(device.type, state)
        except SmartHomeError as exc:
            return _failed(command, exc.code)
        except TimeoutError:
            # The Provider may still act on it. Reconcile, never resend.
            return _unknown(command, ERROR_DEADLINE_EXCEEDED)
        except Exception:
            logger.exception(
                "smart home provider=%s owner=%s device=%s left the command unresolved",
                device.provider,
                owner_id,
                device.device_id,
            )
            return _unknown(command, None)
        before.setdefault(device.device_id, owner.states.get(device.device_id))
        owner.states[device.device_id] = state
        return CommandResult(device_id=command.device_id, status="succeeded", state=state)

    # -- panels ------------------------------------------------------------

    async def attach_panel(self, owner_id: str, device_ref: str) -> None:
        """Start sending this Owner's changes to the panel, beginning with a snapshot."""
        owner = self._owner(owner_id)
        async with owner.lock:
            registry = await self._registry(owner_id, owner)
            owner.panels[device_ref] = None
            await self._panels.send(
                owner_id, device_ref, OP_SNAPSHOT, self._snapshot(owner, registry, device_ref)
            )

    def detach_panel(self, owner_id: str, device_ref: str) -> None:
        owner = self._owners.get(owner_id)
        if owner is not None:
            owner.panels.pop(device_ref, None)

    async def handle_panel_request(self, owner_id: str, device_ref: str, body: object) -> None:
        """One decoded ``PANEL_REQUEST_TOPIC`` body from the panel bound to ``device_ref``.

        The transport's entry point, so it never raises into the transport: a
        body that is not a ``PanelRequest`` is dropped, and a request that
        could not be carried out is logged. The panel sees the outcome as a
        delta, or asks to sync when it sees none.
        """
        try:
            message = PanelRequest.model_validate(body).message()
        except ValidationError:
            logger.warning(
                "smart home panel request rejected owner=%s panel=%s: malformed body",
                owner_id,
                device_ref,
            )
            return
        try:
            if isinstance(message, PanelExecute):
                await self.handle_panel_execute(owner_id, device_ref, message)
            else:
                await self.handle_panel_sync(owner_id, device_ref, message)
        except IdempotencyConflict:
            logger.warning(
                "smart home panel request refused owner=%s panel=%s: request_id %s reused",
                owner_id,
                device_ref,
                message.request_id if isinstance(message, PanelExecute) else "",
            )
        except Exception:
            logger.exception(
                "smart home panel request failed owner=%s panel=%s", owner_id, device_ref
            )

    async def handle_panel_sync(self, owner_id: str, device_ref: str, message: PanelSync) -> None:
        """A panel lost track (a seq gap, a reconnect); a whole snapshot is the only answer."""
        owner = self._owner(owner_id)
        async with owner.lock:
            registry = await self._registry(owner_id, owner)
            logger.debug(
                "smart home panel sync owner=%s panel=%s known=(%s,%s) current=(%s,%s)",
                owner_id,
                device_ref,
                message.known_revision,
                message.known_seq,
                registry.revision,
                owner.seq,
            )
            await self._panels.send(
                owner_id, device_ref, OP_SNAPSHOT, self._snapshot(owner, registry, device_ref)
            )

    async def on_registry_changed(self, owner_id: str) -> None:
        """Re-read the registry, converge Providers on it, and re-snapshot every panel.

        A registry change is never a delta: panels replace everything they hold.
        """
        owner = self._owner(owner_id)
        async with owner.lock:
            owner.registry = None
            registry = await self._registry(owner_id, owner)
            await self._broadcast(
                owner_id,
                owner,
                OP_SNAPSHOT,
                lambda device_ref: self._snapshot(owner, registry, device_ref),
            )

    # -- internals ---------------------------------------------------------

    def _owner(self, owner_id: str) -> _Owner:
        if not owner_id:
            raise ValueError("owner_id is required")
        return self._owners.setdefault(owner_id, _Owner())

    async def _registry(self, owner_id: str, owner: _Owner) -> Registry:
        if owner.registry is None:
            registry = await self._registry_source.get(owner_id)
            states: dict[str, dict[str, Any]] = {}
            # Every Provider converges, including one left with no devices, so
            # removing a device from the registry forgets its state.
            for name, provider in self._providers.items():
                devices = [d for d in registry.devices if d.provider == name]
                await provider.reconcile(owner_id, devices)
                states.update(await provider.states(owner_id, devices))
            owner.registry, owner.states = registry, states
        return owner.registry

    def _snapshot(self, owner: _Owner, registry: Registry, device_ref: str) -> dict[str, Any]:
        devices = []
        for device in registry.devices:
            state = owner.states.get(device.device_id)
            devices.append(
                PanelDevice(
                    device_id=device.device_id,
                    name=device.name,
                    type=device.type,
                    area_id=device.area_id,
                    # No Provider here reports it; the panel greys it out.
                    online=state is not None,
                    state=state,
                )
            )
        snapshot = PanelSnapshot(
            revision=registry.revision,
            seq=owner.seq,
            panel_area_id=registry.area_of(device_ref),
            utc_offset_minutes=self._utc_offset_minutes(self._now_ms()),
            areas=tuple(
                PanelArea(area_id=area.area_id, name=area.name)
                for area in sorted(registry.areas, key=lambda area: area.order)
            ),
            devices=tuple(devices),
            scenes=tuple(PanelScene(scene_id=s.scene_id, name=s.name) for s in registry.scenes),
        )
        return snapshot.model_dump(mode="json")

    async def _broadcast(
        self, owner_id: str, owner: _Owner, op: str, payload_for: Callable[[str], dict[str, Any]]
    ) -> None:
        """Best effort per panel: one that misses this finds a gap and asks to sync."""

        async def send(device_ref: str) -> None:
            try:
                async with asyncio.timeout(PANEL_SEND_TIMEOUT_S):
                    await self._panels.send(owner_id, device_ref, op, payload_for(device_ref))
            except Exception:
                logger.exception(
                    "smart home %s not delivered owner=%s panel=%s", op, owner_id, device_ref
                )

        await asyncio.gather(*(send(device_ref) for device_ref in tuple(owner.panels)))

    def _lookup(
        self, owner: _Owner, key: tuple[str, str], fingerprint: str
    ) -> ExecuteResult | None:
        now_ms = self._now_ms()
        # Constant TTL: insertion order is expiry order.
        while owner.recorded and next(iter(owner.recorded.values())).expires_at_ms <= now_ms:
            owner.recorded.popitem(last=False)
        recorded = owner.recorded.get(key)
        if recorded is None:
            return None
        if recorded.fingerprint != fingerprint:
            raise IdempotencyConflict(f"request_id {key[1]!r} was used for a different request")
        return recorded.result

    def _record(
        self, owner: _Owner, key: tuple[str, str], fingerprint: str, result: ExecuteResult
    ) -> None:
        owner.recorded[key] = _Recorded(self._now_ms() + self._ttl_ms, fingerprint, result)
        while len(owner.recorded) > self._capacity:
            owner.recorded.popitem(last=False)


def _fingerprint(request: ExecuteRequest) -> str:
    return json.dumps(
        request.model_dump(mode="json", include={"commands", "scene_id"}),
        sort_keys=True,
        separators=(",", ":"),
    )


def _refused(request: ExecuteRequest, code: str) -> ExecuteResult:
    return ExecuteResult(request_id=request.request_id, error=code)


def _failed(command: Command, code: str) -> CommandResult:
    return CommandResult(device_id=command.device_id, status="failed", code=code)


def _unknown(command: Command, code: str | None) -> CommandResult:
    return CommandResult(device_id=command.device_id, status="unknown", code=code)
