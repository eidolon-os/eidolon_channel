"""The port every channel adapter implements.

The vocabulary here is deliberately the domain's, not any transport's: a
channel is opened and closed, and what comes back is an opaque grant plus an
opaque handle. LiveKit's rooms, an MQTT broker's topics and a WebSocket's
endpoints are all details that live behind this line, including how their
resources are named.

A channel and a session are two different lifetimes. The channel lasts as long
as the device is enrolled: it is the device's standing way to reach us, and the
device sits in it continuously. A session is one stretch of conversation inside
that channel, and it is the expensive one — it is what an agent, its models and
its upstream speech services are paid for. Opening a channel therefore must not
start a session, which is why serving is its own pair of operations here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .spec import ChannelSpec


class ServingAction(Enum):
    """The desired serving direction for a device channel."""

    START = "start"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ServingRequest:
    """What a device asked of its own channel.

    Statements of desired state, not events: the same one twice means the same
    thing once for one ``conversation_id``. The correlation key also fences a
    late close from a previous conversation.
    """

    action: ServingAction
    conversation_id: str


# Called by an adapter when the device on a channel asks. Awaited, so an adapter
# learns whether the request was actually carried out.
ServingRequestSink = Callable[[ServingRequest], Awaitable[None]]


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

    @property
    def serves_dataonly(self) -> bool:
        """Whether this adapter will carry a Body that declares no media.

        Not "can this transport carry data" — every one of them can, which is
        exactly why that question could not be the gate. A Body declaring no
        media gets no serving spec, so no agent is dispatched to it, and while
        a media-less spec trivially "needed nothing" every adapter qualified:
        the device was handed a channel, joined it, and was never answered, by
        anyone, with nothing raised anywhere. So this is the narrower question
        an adapter must answer for itself, and answering it `False` is how a
        deployment says it has nothing to offer such a device yet.
        """
        ...

    async def healthcheck(self) -> None:
        """Raise `BackendUnavailable` if this adapter cannot serve right now."""
        ...

    async def open(self, spec: ChannelSpec, *, issued_at_ms: int) -> ChannelGrant:
        """Converge the channel resource and mint a fresh device grant.

        Called both for a first provision and for a refresh, so it must be
        idempotent for the same spec.  A returned handle names the standing
        transport resource; it does not represent ownership of the credential
        that was just minted.
        """
        ...

    def resource_identity(self, handle: dict[str, Any]) -> str:
        """Return the stable transport-resource identity named by ``handle``.

        Credential operations have their own operation ids, but a refresh can
        legitimately mint a new grant for the same standing room/topic.  The
        service compares this identity before retiring a previous resource so
        rotating authorization can never destroy the channel it authorizes.
        """
        ...

    async def close(self, handle: dict[str, Any]) -> None:
        """Destroy the standing transport resource named by ``handle``.

        This is a resource-lifecycle operation, not credential revocation.  It
        is called only when no active provision retains the same resource
        identity (or when the device channel is explicitly revoked).
        """
        ...

    async def open_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        """Bring this channel's agent to it, so a conversation can happen.

        Converges rather than counts: asking twice leaves one session, because
        the caller is a device that may retry and must never end up served
        twice. Raises `ChannelNotServable` for a channel whose spec carried no
        `ServingSpec` — a device that cannot speak has no session to open.
        """
        ...

    async def close_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        """Send the agent away. The channel itself stays open.

        The device keeps its place and its credentials; only the served part of
        the channel ends. Must tolerate there being no session to close.
        """
        ...

    async def accept_requests(
        self, handle: dict[str, Any], *, sink: ServingRequestSink
    ) -> None:
        """Start carrying this channel's own requests to be served.

        A channel runs both ways, so the device can say "serve me now" over the
        one connection it already has and is already known on — it needs no
        second address and no second credential to be heard. How that reaches
        us is this adapter's business: a message on a topic, a frame, a callback
        from the transport. What comes back out is only ever a `ServingRequest`.

        Idempotent per handle, because the service re-states what it wants
        watched on every provision and on every restart.
        """
        ...

    async def stop_accepting(self, handle: dict[str, Any]) -> None:
        """Stop listening to this channel. Must tolerate one never watched."""
        ...

    async def device_is_on_channel(self, handle: dict[str, Any]) -> bool | None:
        """Whether the device itself is on this channel right now.

        The device, not the channel: a channel that exists and an agent sitting
        in it are both true of a body that is switched off, and neither is what
        anyone means by "is it there". So an adapter answers this only about the
        one participant that is the device.

        Three answers, and the third is the important one. ``True`` and
        ``False`` are observations. ``None`` means this adapter cannot see —
        the transport did not answer, or this kind of channel has no notion of
        being on it — and it must never be collapsed into ``False``, which
        would tell someone their speaker is off when the truth is that nobody
        with standing looked.

        Must not raise for an unreachable backend: an observation that failed
        is ``None``, and presence is never worth failing a read for.
        """
        ...

    async def shutdown(self) -> None:
        """Release adapter-wide resources at process exit."""
        ...
