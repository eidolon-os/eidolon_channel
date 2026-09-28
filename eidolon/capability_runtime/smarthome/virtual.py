"""The Host's virtual Provider: simulated devices whose state survives a restart.

Phase one has no real devices behind the registry, so "turn on the air
conditioner" means this Provider's copy of that air conditioner becomes on.
It is a Provider like any later one (Home Assistant, a Matter bridge): the
runtime, the panels and the Agent cannot tell it apart, which is the point.

State is kept per (owner_id, device_id) in its own SQLite file. The vocabulary
and bounds are the SDK's; what a valid command *does* is decided here.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from eidolon_sdk.biz.smarthome import (
    ERROR_UNSUPPORTED_COMMAND,
    Command,
    Device,
    SmartHomeError,
    device_type,
    initial_state,
    validate_command,
    validate_state,
)

logger = logging.getLogger("eidolon.capability_runtime.smarthome.virtual")

SCHEMA_VERSION = 1
PROVIDER_NAME = "virtual"

_RUN_STATES = {"start": "running", "pause": "paused", "stop": "idle", "dock": "docked"}


def apply_command(kind: str, state: Mapping[str, Any], command: Command) -> dict[str, Any]:
    """The full state after ``command``; raises ``SmartHomeError`` for one it refuses.

    Steps are clamped to the trait's range rather than refused: "a bit warmer"
    at the top of the range is a no-op, not an error. Adjusting a light's level
    or a thermostat implies wanting the device on, so both switch it on.
    """
    validate_command(kind, command)
    spec = device_type(kind)
    params = command.params
    new = dict(state)
    match command.trait, command.command:
        case "on_off", "on" | "off":
            new["on"] = command.command == "on"
        case "on_off", "toggle":
            new["on"] = not state["on"]
        case "level", "set":
            new["level"] = params["value"]
        case "level", "step":
            new["level"] = _clamp(state["level"] + params["delta"], 0, 100)
        case "thermostat", "set_mode":
            new["mode"] = params["mode"]
        case "thermostat", "set_target":
            new["target_c"] = _celsius(params["celsius"])
        case "thermostat", "step":
            low, high = spec.target_c or (None, None)
            new["target_c"] = _celsius(_clamp(state["target_c"] + params["delta"], low, high))
        case "fan_speed", "set":
            new["speed"] = params["value"]
        case "position", "open":
            new["position"] = 100
        case "position", "close":
            new["position"] = 0
        case "position", "stop":
            pass  # A virtual cover is never mid-travel.
        case "position", "set":
            new["position"] = params["value"]
        case "lock", "lock" | "unlock":
            new["locked"] = command.command == "lock"
        case "operational", ("start" | "pause" | "stop" | "dock") as verb:
            new["run_state"] = _RUN_STATES[verb]
        case "volume", "set":
            new["volume"] = params["value"]
        case "volume", "step":
            new["volume"] = _clamp(state["volume"] + params["delta"], 0, 100)
        case "volume", "mute":
            new["muted"] = params["muted"]
        case _:
            # The SDK accepted a command this Provider has no meaning for yet.
            raise SmartHomeError(ERROR_UNSUPPORTED_COMMAND, f"{command.trait}.{command.command}")
    if command.trait in ("level", "thermostat") and "on" in new:
        new["on"] = True
    validate_state(kind, new)
    return new


def _clamp(value: float, low: float | None, high: float | None) -> float:
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value


def _celsius(value: float) -> int | float:
    # 26.0 and 26 are the same setpoint; keep the wire showing 26.
    return int(value) if float(value).is_integer() else float(value)


class VirtualProvider:
    """``SmartHomeProvider`` for devices that exist only on this Host."""

    name = PROVIDER_NAME

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._transaction() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in {0, SCHEMA_VERSION}:
                raise RuntimeError(f"unsupported virtual smart home schema version: {version}")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS virtual_devices (
                    owner_id TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, device_id)
                )
                """
            )
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        # Whether the front door is locked is the Owner's business.
        os.chmod(self._path, 0o600)

    async def reconcile(self, owner_id: str, devices: Sequence[Device]) -> None:
        wanted = {device.device_id for device in devices}
        with self._transaction() as db:
            rows = self._rows(db, owner_id)
            db.executemany(
                "DELETE FROM virtual_devices WHERE owner_id=? AND device_id=?",
                [(owner_id, device_id) for device_id in rows if device_id not in wanted],
            )
            for device in devices:
                row = rows.get(device.device_id)
                if self._stored_state(device, row) is None:
                    if row is not None:
                        logger.warning(
                            "restarting virtual device owner=%s device=%s: stored state no "
                            "longer fits type %s",
                            owner_id,
                            device.device_id,
                            device.type,
                        )
                    self._write(db, owner_id, device, initial_state(device.type))

    async def states(self, owner_id: str, devices: Sequence[Device]) -> dict[str, dict[str, Any]]:
        with self._transaction() as db:
            rows = self._rows(db, owner_id)
        return {
            device.device_id: self._stored_state(device, rows.get(device.device_id))
            or initial_state(device.type)
            for device in devices
        }

    async def execute(self, owner_id: str, device: Device, command: Command) -> dict[str, Any]:
        if command.device_id != device.device_id:
            raise ValueError("command addresses a different device")
        with self._transaction() as db:
            row = db.execute(
                "SELECT device_type, state_json FROM virtual_devices "
                "WHERE owner_id=? AND device_id=?",
                (owner_id, device.device_id),
            ).fetchone()
            current = self._stored_state(device, row) or initial_state(device.type)
            state = apply_command(device.type, current, command)
            if row is None or state != current:
                self._write(db, owner_id, device, state)
        return state

    @staticmethod
    def _rows(db: sqlite3.Connection, owner_id: str) -> dict[str, tuple[str, str]]:
        return {
            device_id: (kind, state_json)
            for device_id, kind, state_json in db.execute(
                "SELECT device_id, device_type, state_json FROM virtual_devices WHERE owner_id=?",
                (owner_id,),
            )
        }

    @staticmethod
    def _stored_state(device: Device, row: tuple[str, str] | None) -> dict[str, Any] | None:
        """The stored state if it still describes this device, else None."""
        if row is None or row[0] != device.type:
            return None
        try:
            state = json.loads(row[1])
            validate_state(device.type, state)
        except (AttributeError, TypeError, ValueError):
            return None
        return state

    @staticmethod
    def _write(
        db: sqlite3.Connection, owner_id: str, device: Device, state: dict[str, Any]
    ) -> None:
        db.execute(
            """INSERT INTO virtual_devices VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (owner_id, device_id) DO UPDATE SET device_type=excluded.device_type,
            state_json=excluded.state_json, updated_at_ms=excluded.updated_at_ms""",
            (
                owner_id,
                device.device_id,
                device.type,
                json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                time.time_ns() // 1_000_000,
            ),
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=5.0, isolation_level=None)
        try:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            db.commit()
        finally:
            db.close()
