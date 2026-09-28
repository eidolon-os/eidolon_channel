"""What the smart home runtime needs from the world around it.

Three ports, each with one direction. The registry comes from System Data and is
only ever read here. Panels are only ever sent to; what they send back arrives
through the runtime's own methods, with identity supplied by the transport
binding. Device state is owned by a Provider, which the runtime picks by the
registry's ``Device.provider``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from eidolon_sdk.biz.smarthome import Command, Device, Registry


class RegistrySource(Protocol):
    async def get(self, owner_id: str) -> Registry:
        """The Owner's current registry. Raises if System Data cannot answer."""
        ...


class PanelSink(Protocol):
    async def send(self, owner_id: str, device_ref: str, op: str, payload: dict[str, Any]) -> None:
        """Deliver one validated panel payload to the panel bound to ``device_ref``.

        Wrapping it in an ``eidolon.control`` envelope (see ``wire.panel_command``)
        and choosing the channel are the binding's business. Raises when the
        payload could not be handed to the transport.
        """
        ...


class SmartHomeProvider(Protocol):
    """Owns the state of the devices it implements, per Owner."""

    async def reconcile(self, owner_id: str, devices: Sequence[Device]) -> None:
        """Converge on exactly these devices: start new ones, forget removed ones."""
        ...

    async def states(self, owner_id: str, devices: Sequence[Device]) -> dict[str, dict[str, Any]]:
        """Each reachable device's full current state, keyed by device_id."""
        ...

    async def execute(self, owner_id: str, device: Device, command: Command) -> dict[str, Any]:
        """Carry out one command and return the device's full state after it.

        Raises ``SmartHomeError`` for a command it refused without acting.
        """
        ...
