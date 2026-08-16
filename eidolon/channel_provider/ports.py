"""The port every channel adapter implements.

The vocabulary here is deliberately the domain's, not any transport's: a
channel is opened and closed, and what comes back is an opaque grant plus an
opaque handle. LiveKit's rooms, an MQTT broker's topics and a WebSocket's
endpoints are all details that live behind this line, including how their
resources are named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .spec import ChannelSpec


@dataclass(frozen=True, slots=True)
class ChannelGrant:
    """What the device is handed so it can reach its channel.

    `binding_format` tags the wire shape of `payload`. The Hub relays both
    without inspecting them, and the device selects the assignment whose format
    it understands, so introducing a second adapter needs no Hub change and
    leaves older firmware able to ignore what it cannot speak.
    """

    binding_format: str
    payload: bytes = field(repr=False)
    expires_at_ms: int
    # Everything the adapter needs to close or re-open this channel later.
    # Persisted verbatim; no other layer may interpret it.
    handle: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - defensive, keeps tokens out of logs
        return (
            f"ChannelGrant(binding_format={self.binding_format!r}, "
            f"payload=<redacted>, expires_at_ms={self.expires_at_ms!r})"
        )


class ChannelAdapter(Protocol):
    """One way of realising a device channel."""

    @property
    def name(self) -> str:
        """Stable identifier persisted alongside the handle."""
        ...

    @property
    def carries_media(self) -> bool:
        """Whether this transport can carry audio or video, not just data."""
        ...

    async def healthcheck(self) -> None:
        """Raise `BackendUnavailable` if this adapter cannot serve right now."""
        ...

    async def open(self, spec: ChannelSpec, *, issued_at_ms: int) -> ChannelGrant:
        """Provision the channel and mint the device's grant.

        Called both for a first provision and for a refresh, so it must be
        idempotent for the same spec.
        """
        ...

    async def close(self, handle: dict[str, Any]) -> None:
        """Release whatever `open` provisioned. Must tolerate an absent channel."""
        ...

    async def shutdown(self) -> None:
        """Release adapter-wide resources at process exit."""
        ...
