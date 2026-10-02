"""Channel-owned panel projection of Hub device state."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from eidolon_sdk.biz.smarthome import (
    OP_RESULT,
    OP_SNAPSHOT,
    ExecuteRequest,
    ExecuteResult,
    Origin,
    PanelArea,
    PanelDevice,
    PanelExecute,
    PanelRequest,
    PanelScene,
    PanelSnapshot,
    PanelSync,
    Registry,
    VoiceResult,
)
from pydantic import ValidationError

from .ports import HomeRuntime, PanelSink

logger = logging.getLogger("eidolon.capability_runtime.smarthome")

TOUCH_DEADLINE_MS = 3_000
PANEL_SEND_TIMEOUT_S = 5.0


def local_utc_offset_minutes(now_ms: int) -> int:
    """This Host's UTC offset at ``now_ms``, which the panel shows local time with."""
    offset = datetime.fromtimestamp(now_ms / 1000).astimezone().utcoffset()
    return int(offset.total_seconds() // 60) if offset is not None else 0


@dataclass
class _Owner:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    registry: Registry | None = None
    # Last state each Provider reported, for devices a registered Provider serves.
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    seq: int = 0
    panels: dict[str, None] = field(default_factory=dict)  # insertion-ordered set


class SmartHomeRuntime:
    """Panel projection and touch translation; never owns a device Provider."""

    def __init__(
        self,
        *,
        backend: HomeRuntime,
        panels: PanelSink,
        now_ms: Callable[[], int] | None = None,
        touch_deadline_ms: int = TOUCH_DEADLINE_MS,
        utc_offset_minutes: Callable[[int], int] = local_utc_offset_minutes,
    ):
        self._backend = backend
        self._panels = panels
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._touch_deadline_ms = touch_deadline_ms
        self._utc_offset_minutes = utc_offset_minutes
        self._owners: dict[str, _Owner] = {}

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
        result = await self._backend.execute(owner_id, request)
        await self.on_registry_changed(owner_id)
        return result

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

    async def send_voice_result(self, owner_id: str, device_ref: str, result: VoiceResult) -> None:
        await self._panels.send(owner_id, device_ref, OP_RESULT, result.model_dump(mode="json"))
        await self.on_registry_changed(owner_id)

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
        except Exception:
            logger.exception(
                "smart home panel request failed owner=%s panel=%s", owner_id, device_ref
            )

    async def handle_panel_sync(self, owner_id: str, device_ref: str, message: PanelSync) -> None:
        """A panel lost track (a seq gap, a reconnect); a whole snapshot is the only answer."""
        owner = self._owner(owner_id)
        async with owner.lock:
            registry = await self._registry(owner_id, owner)
            owner.panels[device_ref] = None
            logger.info(
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
        """Re-read Hub's authoritative snapshot and update panel projections.

        A registry change is never a delta: panels replace everything they hold.
        """
        owner = self._owner(owner_id)
        async with owner.lock:
            await self._registry(owner_id, owner)

    async def refresh_active_panels(self) -> None:
        """Catch registry edits made through System Data while panels are open."""
        for owner_id, owner in tuple(self._owners.items()):
            if not owner.panels:
                continue
            try:
                async with owner.lock:
                    await self._registry(owner_id, owner)
            except Exception:
                logger.exception("smart home registry refresh failed owner=%s", owner_id)

    # -- internals ---------------------------------------------------------

    def _owner(self, owner_id: str) -> _Owner:
        if not owner_id:
            raise ValueError("owner_id is required")
        return self._owners.setdefault(owner_id, _Owner())

    async def _registry(self, owner_id: str, owner: _Owner) -> Registry:
        body = await self._backend.snapshot(owner_id)
        registry = Registry.model_validate(body["registry"])
        states = {key: value["state"] for key, value in body["status"].items() if value["online"]}
        previous_registry, previous_states = owner.registry, owner.states
        owner.registry, owner.states = registry, states
        if previous_registry is not None:
            if previous_registry != registry or previous_states.keys() != states.keys():
                await self._broadcast(
                    owner_id, owner, OP_SNAPSHOT, lambda ref: self._snapshot(owner, registry, ref)
                )
            elif previous_states != states:
                owner.seq += 1
                await self._broadcast(
                    owner_id, owner, OP_SNAPSHOT, lambda ref: self._snapshot(owner, registry, ref)
                )
        return registry

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
