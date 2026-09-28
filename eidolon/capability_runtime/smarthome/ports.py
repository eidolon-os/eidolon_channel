"""Channel presentation ports. Device execution is owned by Hub."""

from __future__ import annotations

from typing import Any, Protocol

from eidolon_sdk.biz.smarthome import ExecuteRequest, ExecuteResult


class HomeRuntime(Protocol):
    async def snapshot(self, owner_id: str) -> dict[str, Any]: ...
    async def execute(self, owner_id: str, request: ExecuteRequest) -> ExecuteResult: ...


class PanelSink(Protocol):
    async def send(self, owner_id: str, device_ref: str, op: str, payload: dict[str, Any]) -> None:
        """Deliver one validated panel payload to the panel bound to ``device_ref``.

        Wrapping it in an ``eidolon.control`` envelope (see ``wire.panel_command``)
        and choosing the channel are the binding's business. Raises when the
        payload could not be handed to the transport.
        """
        ...
