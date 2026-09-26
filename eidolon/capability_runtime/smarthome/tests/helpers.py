from __future__ import annotations

from typing import Any

from eidolon_sdk.biz.smarthome import (
    OP_DELTA,
    OP_SNAPSHOT,
    Command,
    ExecuteRequest,
    Origin,
    PanelDelta,
    PanelSnapshot,
    Registry,
)
from eidolon_sdk.biz.smarthome.samples import apartment

OWNER = "owner-a"
OTHER_OWNER = "owner-b"
PANEL = "panel-living"
NOW_MS = 1_700_000_000_000


def home(
    *, revision: int = 1, placements: dict[str, str] | None = None, **changes: Any
) -> Registry:
    """The sample apartment, with the given panels placed and fields replaced."""
    value = apartment().model_dump(mode="json")
    value["revision"] = revision
    placed = {PANEL: "living"} if placements is None else placements
    value["placements"] = [{"device_ref": ref, "area_id": area} for ref, area in placed.items()]
    value.update(changes)
    return Registry.model_validate(value)


def cmd(device_id: str, trait: str, command: str, **params: Any) -> Command:
    return Command(device_id=device_id, trait=trait, command=command, params=params)


def request(
    request_id: str,
    *commands: Command,
    kind: str = "voice",
    label: str | None = None,
    deadline_ms: int = NOW_MS + 3_000,
) -> ExecuteRequest:
    return ExecuteRequest(
        request_id=request_id,
        commands=commands,
        origin=Origin(kind=kind, label=label),
        deadline_ms=deadline_ms,
    )


def scene(
    request_id: str,
    scene_id: str,
    *,
    kind: str = "voice",
    label: str | None = None,
    deadline_ms: int = NOW_MS + 3_000,
) -> ExecuteRequest:
    return ExecuteRequest(
        request_id=request_id,
        scene_id=scene_id,
        origin=Origin(kind=kind, label=label),
        deadline_ms=deadline_ms,
    )


class Clock:
    def __init__(self, now_ms: int = NOW_MS) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


class FakeRegistry:
    """What System Data would answer, per Owner."""

    def __init__(self, registries: dict[str, Registry]) -> None:
        self.registries = dict(registries)
        self.reads = 0

    async def get(self, owner_id: str) -> Registry:
        self.reads += 1
        return self.registries[owner_id]


class RecordingPanels:
    """Every payload handed to the transport, checked against the SDK on the way."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, dict[str, Any]]] = []
        self.unreachable: set[str] = set()

    async def send(self, owner_id: str, device_ref: str, op: str, payload: dict[str, Any]) -> None:
        if device_ref in self.unreachable:
            raise ConnectionError("panel is gone")
        {OP_SNAPSHOT: PanelSnapshot, OP_DELTA: PanelDelta}[op].model_validate(payload)
        self.sent.append((owner_id, device_ref, op, payload))

    def to(self, device_ref: str, op: str | None = None) -> list[dict[str, Any]]:
        return [p for _, ref, o, p in self.sent if ref == device_ref and op in (None, o)]

    def ops(self, device_ref: str) -> list[tuple[str, int]]:
        return [(o, p["seq"]) for _, ref, o, p in self.sent if ref == device_ref]

    def last(self, device_ref: str, op: str) -> dict[str, Any]:
        return self.to(device_ref, op)[-1]


def device_state(snapshot: dict[str, Any], device_id: str) -> dict[str, Any]:
    return next(d for d in snapshot["devices"] if d["device_id"] == device_id)["state"]
