"""Realising a device channel as a LiveKit room.

Everything LiveKit-shaped lives behind this file: room naming, grants, agent
dispatch and the binding's wire shape. The service above it never learns that
rooms exist.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse

from eidolon_sdk.biz.contracts import (
    SESSION_CLOSE_TYPE,
    SESSION_CONVERSATION_ID_FIELD,
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
    SESSION_END_TYPE,
    SESSION_OPEN_TYPE,
    WIRE_SCHEMA_VERSION,
    normalize_conversation_id,
)
from livekit import api, rtc
from livekit.api.twirp_client import TwirpError, TwirpErrorCode
from livekit.protocol.agent import JobStatus
from livekit.protocol.models import ParticipantInfo

from ...contracts import BackendUnavailable, ChannelNotServable, canonical_json
from ...ports import ChannelGrant, ServingAction, ServingRequest, ServingRequestSink
from ...spec import ChannelSpec
from .config import LiveKitConfig

logger = logging.getLogger("eidolon.channel_provider.livekit")

ADAPTER_NAME = "livekit"
BINDING_FORMAT = "application/vnd.eidolon.livekit-session+json;v=2"

def _routable_address() -> str:
    """This machine's address on the interface it would leave by.

    A UDP socket connected to a documentation address (RFC 5737 TEST-NET-1)
    sends nothing and reaches nothing; it only makes the kernel choose a route,
    and the local end of that choice is the address. Asked rather than
    configured because a Host grows and loses addresses on its own — a
    maintenance cable, a new access point — and no file written earlier knows
    which one is current.
    """

    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("192.0.2.1", 9))
        return str(connection.getsockname()[0])
    finally:
        connection.close()


_HEALTHCHECK_ROOM = "__eidolon_channel_provider_healthcheck__"
_ENDED_JOB_STATUSES = frozenset({JobStatus.JS_SUCCESS, JobStatus.JS_FAILED})
_SESSION_REQUESTS = {
    SESSION_OPEN_TYPE: ServingAction.START,
    SESSION_CLOSE_TYPE: ServingAction.STOP,
}
_REJOIN_BASE_DELAY = 1.0
_REJOIN_MAX_DELAY = 30.0
# How long a dispatch has to become a serving agent before we say it did
# not. Generous on purpose: a cold worker measured 3.6s of prewarm and
# then joined in under a second, so this is far outside the normal spread
# and only fires when nothing arrived at all.
_SERVING_CONFIRM_DELAY = 10.0


@dataclass
class _Listening:
    """One channel this adapter has undertaken to carry requests for.

    Outlives any particular connection to it, which is the point: the promise is
    to the channel, and a connection is only how it is currently being kept.
    """

    device: str
    sink: ServingRequestSink
    connection: Any = None
    retry: asyncio.Task[None] | None = None
    attempt: int = 0


def _is_spent(dispatch: Any) -> bool:
    """Whether this dispatch's work is over rather than under way.

    An empty job list is deliberately *not* spent. LiveKit publishes the job a
    beat after it starts running, so a dispatch created moments ago reads as
    having no jobs at all — treating that as finished would tear down the very
    session that is starting up.
    """
    jobs = list(dispatch.state.jobs)
    return bool(jobs) and all(job.state.status in _ENDED_JOB_STATUSES for job in jobs)


def _failure_of(dispatch: Any) -> str:
    """What this dispatch's failed jobs say for themselves, or "" if none did.

    The one place `JS_FAILED` must not be interchangeable with `JS_SUCCESS`.
    `_is_spent` is right to fold them together — a finished job cannot serve a
    new conversation whichever way it finished — but that answer is about the
    dispatch's usefulness, and this one is about a device that was not served,
    which is the only question the two statuses answer differently. `error` is
    the job's own account of why, and on this side of the edge it is the only
    description of the failure that exists.
    """
    return "; ".join(
        f"job={getattr(job, 'id', '') or '?'} failed: "
        f"{getattr(job.state, 'error', '') or 'no reason reported'}"
        for job in dispatch.state.jobs
        if job.state.status == JobStatus.JS_FAILED
    )


def _why_unserved(dispatch: Any) -> str:
    """Say what the dispatch knows about a conversation nobody is serving.

    Three answers worth telling apart, because each sends the reader somewhere
    else: a job that died, and its own error; no job at all, meaning nothing
    ever took the standing order — what an absent or misnamed worker looks like
    from here; or a job LiveKit still holds as live, meaning the agent left the
    room without its job being marked.
    """
    failure = _failure_of(dispatch)
    if failure:
        return failure
    jobs = list(dispatch.state.jobs)
    if not jobs:
        return "no worker took the dispatch"
    return "dispatch still holds " + ", ".join(
        JobStatus.Name(job.state.status) for job in jobs
    )


class LiveKitChannelAdapter:
    """Opens one LiveKit room per device and mints that device's token."""

    def __init__(self, config: LiveKitConfig) -> None:
        self._config = config
        self._api: api.LiveKitAPI | None = None
        # One connection per channel we are listening to, keyed by room so that
        # re-stating a channel converges instead of stacking up connections.
        self._listeners: dict[str, _Listening] = {}
        self._requests: set[asyncio.Task[None]] = set()

    @property
    def name(self) -> str:
        return ADAPTER_NAME

    @property
    def carries_media(self) -> bool:
        return True

    @property
    def serves_dataonly(self) -> bool:
        """No: a room for a Body with nothing to say is a room nobody serves.

        LiveKit could carry such a channel technically. It is refused because a
        Manifest that declares no media produces no serving spec, so no agent
        is dispatched — and that combination is what silently handed a device a
        room where nothing would ever answer it. When a sensor-only Body is a
        real product, this becomes a real answer, and everything it needs
        beyond a room can be built behind it.
        """

        return False

    def resource_identity(self, handle: dict[str, Any]) -> str:
        room = str(handle.get("room") or "")
        return f"livekit:room:{room}" if room else ""

    def _client(self) -> api.LiveKitAPI:
        # LiveKitAPI owns an aiohttp session and must be constructed in the
        # running service loop, not while the synchronous composition root loads.
        if self._api is None:
            self._api = api.LiveKitAPI(
                url=self._config.api_url,
                api_key=self._config.api_key,
                api_secret=self._config.api_secret,
            )
        return self._api

    async def healthcheck(self) -> None:
        try:
            await self._client().room.list_rooms(api.ListRoomsRequest(names=[_HEALTHCHECK_ROOM]))
        except Exception as exc:
            raise BackendUnavailable("LiveKit health check failed") from exc

    # -- naming -----------------------------------------------------------
    # Resource names are this adapter's business. Another adapter names topics
    # or endpoints instead, and nothing above this line has to care.

    def _client_url(self) -> str:
        """Where a device should reach this Host's LiveKit, decided now.

        A configured ``ws://:7880`` — scheme and port, no host — says the host
        is not knowable at configuration time, which is the truth: Ops observed
        one address at deploy and froze it into an environment variable, and a
        Host that then moved networks or renewed a lease went on handing out the
        address it used to have. The Local API survives that by offering every
        candidate and being re-located; a session binding names one server and
        has neither.

        So it is answered per binding, from the address the kernel says it would
        use to leave this machine. This is the signalling URL only: LiveKit
        gathers its own ICE candidates, and eidolond refreshes that transport
        when its captured network inputs change. Resolving this URL does not
        by itself prove that the media transport is ready.

        Not the same as knowing where the *device* is — a Host with two networks
        still has to pick one, and picking the routable one is a rule, not
        knowledge. What this removes is staleness, not the assumption.
        """

        parsed = urlparse(self._config.client_url)
        if parsed.hostname is not None:
            return self._config.client_url
        port = parsed.port
        return urlunparse(
            (parsed.scheme, f"{_routable_address()}:{port}", "", "", "", "")
        )

    def _room_name(self, spec: ChannelSpec) -> str:
        digest = hashlib.sha256(
            f"{self._config.room_prefix}\0{spec.device_id}".encode()
        ).hexdigest()[:24]
        return f"{self._config.room_prefix}-{digest}"

    # -- lifecycle --------------------------------------------------------

    async def open(self, spec: ChannelSpec, *, issued_at_ms: int) -> ChannelGrant:
        room = self._room_name(spec)
        await self._declare_room(room)
        ttl = self._config.grant_ttl_seconds
        payload = canonical_json(
            {
                "schema_version": 2,
                "session": {
                    "server_url": self._client_url(),
                    "token": self._token(room, spec, ttl_seconds=ttl),
                    "identity": spec.device_id,
                    "room_name": room,
                },
                "audio": {
                    "sample_rate": self._config.sample_rate,
                    "channels": self._config.channels,
                },
            }
        ).encode()
        # The handle carries the agent name because opening a session later must
        # not depend on re-deriving the spec: by then the manifest that produced
        # it is the Hub's, not ours, and may already have moved on.
        # `device` is the identity the token above was minted for, so a request
        # arriving here can be checked against the only participant entitled to
        # make one — the room also holds the agent, which speaks on the same
        # topic in the other direction.
        handle: dict[str, Any] = {"room": room, "device": spec.device_id}
        if spec.serving is not None:
            handle["agent"] = spec.serving.agent_name
        return ChannelGrant(
            binding_format=BINDING_FORMAT,
            payload=payload,
            expires_at_ms=issued_at_ms + ttl * 1000,
            handle=handle,
        )

    async def close(self, handle: dict[str, Any]) -> None:
        room = handle.get("room")
        if not room:
            return
        try:
            await self._client().room.delete_room(api.DeleteRoomRequest(room=room))
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.NOT_FOUND:
                return
            raise BackendUnavailable("LiveKit room revocation failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit room revocation failed") from exc

    # -- hearing the device -----------------------------------------------

    async def accept_requests(
        self, handle: dict[str, Any], *, sink: ServingRequestSink
    ) -> None:
        """Sit in the room so the device can say when it wants to be served.

        The device is already connected and already authenticated here, by the
        very credential this adapter minted for it, so its own channel is the
        one place it can ask for a conversation without being issued a second
        identity anywhere else.

        Joins hidden: this participant is infrastructure. The agent selects the
        runtime actor from who is visibly in the room, and a listener that
        showed up there would be something for it to reason about.
        """
        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        if not room or not device:
            # The one refusal: nothing about this channel says who may speak for
            # it, so no arrival could ever be attributed. Trying harder cannot
            # change that, and the caller needs to hear so.
            raise ChannelNotServable("channel handle cannot identify its device")
        if room in self._listeners:
            return
        watch = _Listening(device=device, sink=sink)
        self._listeners[room] = watch
        try:
            await self._join(room, watch)
        except Exception as exc:
            # Undertaking to carry a channel is not the same as being connected
            # to it this instant. A device provisioned in the seconds after the
            # transport restarts found exactly this gap on a real Host: the
            # first join timed out, and because the failure was final the
            # channel stayed deaf until something else provisioned it again —
            # while a connection lost one second later would have been rebuilt.
            # Same promise, so the same persistence at both ends.
            self._rejoin_later(room, watch, exc)
            return
        logger.info("listening to room=%s for device=%s", room, device)

    async def _join(self, room: str, watch: _Listening) -> None:
        connection = rtc.Room()

        @connection.on("data_received")
        def _received(packet: Any) -> None:
            self._note_serving_failure(packet, device=watch.device, room=room)
            request = self._requested(packet, device=watch.device, room=room)
            if request is not None:
                self._dispatch_request(watch.sink, request, room=room)

        @connection.on("disconnected")
        def _dropped(reason: Any = None) -> None:
            # Losing this connection is silent in the worst way: the device goes
            # on asking to be heard and nothing anywhere reports that nobody is
            # listening. Measured against a real server — a listener dropped from
            # its room does not come back on its own — so getting back in is this
            # adapter's job for as long as it has said it is carrying the channel.
            if self._listeners.get(room) is watch:
                self._rejoin_later(room, watch, reason)

        try:
            await connection.connect(self._rtc_url(), self._listener_token(room))
        except Exception as exc:
            raise BackendUnavailable("LiveKit channel could not be listened to") from exc
        watch.connection = connection
        watch.attempt = 0

    def _rejoin_later(self, room: str, watch: _Listening, reason: Any) -> None:
        if watch.retry is not None and not watch.retry.done():
            return
        watch.connection = None
        logger.warning("lost room=%s (%s); rejoining", room, reason)

        async def _rejoin() -> None:
            while self._listeners.get(room) is watch:
                # Recomputed every time round, because a transport that is down
                # rather than blinking must be asked less often, not forever at
                # the same rate. Observed on a real Host holding at the base
                # delay for minutes while the sandbox denied it a route out.
                delay = min(_REJOIN_MAX_DELAY, _REJOIN_BASE_DELAY * (2**watch.attempt))
                watch.attempt += 1
                await asyncio.sleep(delay)
                if self._listeners.get(room) is not watch:
                    return
                try:
                    await self._join(room, watch)
                except Exception:
                    logger.warning(
                        "room=%s still unreachable after %.0fs; will keep trying",
                        room, delay,
                    )
                    continue
                logger.info("listening to room=%s again", room)
                return

        watch.retry = asyncio.create_task(_rejoin())

    async def _dispatch_for(
        self, room: str, *, agent: str, conversation_id: str
    ) -> Any | None:
        """The dispatch standing for one conversation, or None if none is.

        Matched the way `close_session` matches it — by agent name and the
        conversation in its own metadata — rather than by an id we remembered:
        LiveKit keeps dispatch records of its own in the same room, and a
        session an earlier process opened must still be recognisable here.
        """
        for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
            if (
                dispatch.agent_name == agent
                and self._dispatch_metadata(dispatch).get(SESSION_CONVERSATION_ID_FIELD)
                == conversation_id
            ):
                return dispatch
        return None

    def _confirm_serving_later(
        self, room: str, *, agent: str, conversation_id: str
    ) -> None:
        """Check that the dispatch we just placed actually became a serving agent.

        This port promises to "bring this channel's agent to it" and what it
        does is place a standing order. Those are not the same thing, and the
        gap between them is invisible: a job that dies on startup leaves the
        dispatch accepted, this log saying `opened session`, the channel record
        answering 200, and the device holding an open microphone against a room
        with nothing in it. Measured on 2026-09-07: the agent was a participant
        for 1.5s and no surface anywhere reported that the conversation had no
        one in it.

        So we go and look, and the order of the three questions is the method.

        Is this conversation still wanted? A dispatch that is no longer there
        was withdrawn by `close_session`, and the agent leaves promptly after
        that — so every exchange shorter than this window leaves the device
        alone in its room, which is indistinguishable from a dispatch that
        never served. Asking this first is the difference between a report and
        an alarm that fires on healthy traffic, which is the same disease as
        the silence it was added to cure.

        Is anyone serving it? `Kind.AGENT` is the exact question — not a head
        count, which includes the device and any infrastructure participant,
        and not an identity guess, since LiveKit names the agent itself. It also
        outranks the dispatch record, which publishes a job a beat late: an
        agent demonstrably in the room is serving whatever the bookkeeping says.

        And if nobody is — why, from the dispatch already in hand. See
        `_why_unserved`: "nothing came" and "the job died with this error" are
        different reports to be woken by.

        This never fails a request. The device has already been answered by the
        time this runs; what it changes is that the Provider stops being the
        last component to know.
        """

        async def _confirm() -> None:
            await asyncio.sleep(_SERVING_CONFIRM_DELAY)
            try:
                dispatch = await self._dispatch_for(
                    room, agent=agent, conversation_id=conversation_id
                )
            except Exception:
                logger.debug("could not read dispatch of room=%s", room, exc_info=True)
                return
            if dispatch is None:
                # Withdrawn while we waited: this conversation ended, which is
                # what a short exchange looks like from here.
                return
            try:
                participants = await self._client().room.list_participants(
                    api.ListParticipantsRequest(room=room)
                )
            except Exception:
                # Nobody looked, which is not the same as nothing being there.
                logger.debug(
                    "could not confirm serving on room=%s", room, exc_info=True
                )
                return
            listed = getattr(participants, "participants", []) or []
            if any(
                getattr(p, "kind", None) == ParticipantInfo.Kind.AGENT
                for p in listed
            ):
                return
            logger.error(
                "dispatch=%s on room=%s agent=%s conversation_id=%s produced no serving "
                "agent within %.0fs — the device asked for a conversation and nothing "
                "is in the room to hold it (%s)",
                dispatch.id,
                room,
                agent,
                conversation_id,
                _SERVING_CONFIRM_DELAY,
                _why_unserved(dispatch),
            )

        task = asyncio.create_task(_confirm())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    async def device_is_on_channel(self, handle: dict[str, Any]) -> bool | None:
        """Is the device's own participant in its room?

        Exact rather than inferred, and that distinction is the whole method.
        A room's participant count includes the agent, which sits in the room
        whether or not anything is listening — so counting would report every
        provisioned body as present forever. The device's identity is its
        ``device_id`` (see ``_token``), so this asks for that one identity and
        answers about that one thing.

        Never raises. LiveKit being unreachable is not an offline speaker; it is
        nobody having looked, which is ``None``.
        """

        room = str(handle.get("room") or "")
        device = str(handle.get("device") or "")
        if not room or not device:
            return None
        try:
            participants = await self._client().room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
        except Exception:
            # Includes the ordinary case of a room LiveKit has already reaped,
            # which is indistinguishable here from a transport failure — and
            # guessing between them is exactly what `None` exists to avoid.
            logger.debug("could not read participants of room=%s", room, exc_info=True)
            return None
        return any(
            getattr(participant, "identity", "") == device
            for participant in getattr(participants, "participants", []) or []
        )

    async def stop_accepting(self, handle: dict[str, Any]) -> None:
        # Dropped from the registry first: the disconnect below fires the same
        # event a failure does, and this is what tells them apart.
        watch = self._listeners.pop(str(handle.get("room") or ""), None)
        if watch is None:
            return
        if watch.retry is not None:
            watch.retry.cancel()
        if watch.connection is None:
            return
        try:
            await watch.connection.disconnect()
        except Exception:  # pragma: no cover - defensive; the channel is going anyway
            logger.debug("failed to leave room=%s cleanly", handle.get("room"), exc_info=True)

    def _requested(self, packet: Any, *, device: str, room: str) -> ServingRequest | None:
        """Read one packet as a request, or decide it is not one.

        Everything here is untrusted input, and the topic carries traffic in
        both directions, so anything that is not one of exactly two things said
        by someone entitled to say it is silently not a request.

        Rejects senders known to be someone else, rather than admitting only the
        sender known to be the device — because for roughly the first three
        seconds after a device connects, packets arrive attributed to nobody
        (measured against a real server: delivered immediately, sender resolved
        at +3s). Admitting only the known device would throw away exactly the
        requests of a device that connects and immediately wants to talk.

        That leniency is bounded by who can be in this room at all: every
        credential for it is minted by this adapter, for one device and one
        agent, and the agent only ever speaks the other direction of this topic
        — which is not a request type and is rejected below on that ground.
        """
        if getattr(packet, "topic", None) != SESSION_CONTROL_TOPIC:
            return None
        identity = getattr(getattr(packet, "participant", None), "identity", None)
        if identity is not None and identity != device:
            return None
        try:
            body = json.loads(bytes(packet.data))
        except (TypeError, ValueError, UnicodeDecodeError):
            logger.warning("room=%s sent an unreadable session request", room)
            return None
        if not isinstance(body, dict):
            return None
        if body.get("schema_v") != WIRE_SCHEMA_VERSION:
            return None
        action = _SESSION_REQUESTS.get(body.get("type"))
        conversation_id = normalize_conversation_id(body.get(SESSION_CONVERSATION_ID_FIELD))
        if action is None or conversation_id is None:
            return None
        return ServingRequest(action=action, conversation_id=conversation_id)

    def _note_serving_failure(self, packet: Any, *, device: str, room: str) -> None:
        """Hear the agent say it could not serve, since we are already in the room.

        The other half of the same edge. `session_end{error}` is published from
        the job's failure path while the room is still live, so it arrives here,
        on this connection, seconds before any confirmation window closes — and
        until now `_requested` dropped it as "not a request", which was true and
        was the end of it. The component that placed the dispatch was the last
        one to learn it had died.

        Read as news, never as an instruction. The device has already been told
        by the agent itself, and ending the session on the strength of a packet
        would take the channel away from the only party that can ask for another
        conversation. So this says so, and does nothing.

        Only `error`. `session_end` also carries the reasons an ordinary
        conversation ends with, and warning on those would teach the reader to
        skip the warning — which is how it would fail to be there for this.
        """
        if getattr(packet, "topic", None) != SESSION_CONTROL_TOPIC:
            return
        identity = getattr(getattr(packet, "participant", None), "identity", None)
        if identity == device:
            # The mirror of `_requested`'s gate: this is the direction the
            # device does not speak, and an unresolved sender is still heard
            # for the reason given there.
            return
        try:
            body = json.loads(bytes(packet.data))
        except (TypeError, ValueError, UnicodeDecodeError):
            # `_requested` has already said so about this same packet.
            return
        if not isinstance(body, dict) or body.get("schema_v") != WIRE_SCHEMA_VERSION:
            return
        if body.get("type") != SESSION_END_TYPE or body.get("reason") != SESSION_END_ERROR:
            return
        logger.warning(
            "room=%s agent reported session_end reason=%s conversation_id=%s — the "
            "conversation this channel opened is not being served",
            room,
            SESSION_END_ERROR,
            normalize_conversation_id(body.get(SESSION_CONVERSATION_ID_FIELD)) or "",
        )

    def _dispatch_request(
        self, sink: ServingRequestSink, request: ServingRequest, *, room: str
    ) -> None:
        """Hand the request on without blocking the room's callback.

        Held in a set until done: a bare task is only weakly referenced, and one
        collected mid-flight would drop a conversation the device asked for.
        """

        async def _carry() -> None:
            try:
                await sink(request)
            except Exception:
                logger.exception("room=%s could not act on %s", room, request.value)

        task = asyncio.create_task(_carry())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    def _rtc_url(self) -> str:
        """Reach LiveKit the short way, not by the address devices are given.

        The client URL is a public name that has to resolve and terminate TLS
        from wherever a device happens to be; this process sits next to the
        server and has no reason to go out and come back.
        """
        api_url = self._config.api_url
        for scheme, replacement in (("https://", "wss://"), ("http://", "ws://")):
            if api_url.startswith(scheme):
                return replacement + api_url[len(scheme) :]
        return api_url

    def _listener_token(self, room: str) -> str:
        return (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(f"channel-provider-{room}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=False,
                    can_subscribe=False,
                    can_publish_data=False,
                    hidden=True,
                )
            )
            .to_jwt()
        )

    async def shutdown(self) -> None:
        for room in list(self._listeners):
            await self.stop_accepting({"room": room})
        for task in list(self._requests):
            task.cancel()
        if self._requests:
            await asyncio.gather(*self._requests, return_exceptions=True)
        if self._api is not None:
            await self._api.aclose()
            self._api = None

    # -- internals --------------------------------------------------------

    async def _declare_room(self, room: str) -> None:
        """Create the room the device will live in, and deliberately nothing more.

        The room is born with no agent dispatch. A dispatch is a standing order:
        LiveKit acts on it the moment anyone is in the room, so a room born with
        one would summon an agent — with its speech models and its metered
        upstream services — the instant the device connected, and keep one for
        as long as the device stayed. Since the device now stays permanently,
        that is a session that never ends. Serving is therefore not part of
        declaring the room; it is `open_session`, asked for when there is
        actually something to say.
        """
        try:
            await self._client().room.create_room(api.CreateRoomRequest(name=room))
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.ALREADY_EXISTS:
                return
            raise BackendUnavailable("LiveKit room creation failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit room creation failed") from exc

    # -- serving ----------------------------------------------------------

    async def open_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        room, agent = self._serving(handle)
        try:
            for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
                if dispatch.agent_name != agent:
                    continue
                metadata = self._dispatch_metadata(dispatch)
                if (
                    not _is_spent(dispatch)
                    and metadata.get(SESSION_CONVERSATION_ID_FIELD) == conversation_id
                ):
                    return
                # A spent dispatch or one belonging to a superseded conversation
                # cannot stand in for the conversation the device is asking for.
                # How the previous one ended is said here because this is
                # the last moment anything knows: deleting the dispatch deletes
                # the only account of a job that failed, so if the confirmation
                # window missed it — this process restarted, or LiveKit could
                # not be read then — nothing ever says so.
                failure = _failure_of(dispatch)
                logger.log(
                    logging.WARNING if failure else logging.INFO,
                    "clearing dispatch=%s on room=%s previous_conversation=%s%s",
                    dispatch.id,
                    room,
                    metadata.get(SESSION_CONVERSATION_ID_FIELD, ""),
                    f" — it had {failure}" if failure else "",
                )
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
            await self._client().agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=room,
                    agent_name=agent,
                    metadata=canonical_json(
                        {
                            "schema_v": WIRE_SCHEMA_VERSION,
                            SESSION_CONVERSATION_ID_FIELD: conversation_id,
                        }
                    ),
                )
            )
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch failed") from exc
        logger.info("opened session on room=%s agent=%s", room, agent)
        self._confirm_serving_later(room, agent=agent, conversation_id=conversation_id)

    async def close_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        """Withdraw the standing order, which is what ends the agent's job.

        Deleting the dispatch — rather than deleting the room — is what keeps
        the device's channel intact across the end of a conversation. LiveKit
        removes the agent from the room promptly, so the room is clean for the
        next session even while the old job is still draining.

        Matches on the agent name rather than a remembered dispatch id: LiveKit
        also keeps dispatch records of its own, and a session that a previous
        process opened must still be closable by this one.
        """
        room, agent = self._serving(handle)
        try:
            for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
                if (
                    dispatch.agent_name == agent
                    and self._dispatch_metadata(dispatch).get(SESSION_CONVERSATION_ID_FIELD)
                    == conversation_id
                ):
                    await self._client().agent_dispatch.delete_dispatch(
                        dispatch_id=dispatch.id, room_name=room
                    )
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.NOT_FOUND:
                return
            raise BackendUnavailable("LiveKit agent dispatch withdrawal failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch withdrawal failed") from exc
        logger.info("closed session on room=%s agent=%s", room, agent)

    @staticmethod
    def _dispatch_metadata(dispatch: Any) -> dict[str, Any]:
        try:
            value = json.loads(str(getattr(dispatch, "metadata", "") or "{}"))
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _serving(handle: dict[str, Any]) -> tuple[str, str]:
        room = handle.get("room")
        agent = handle.get("agent")
        if not room:
            raise ChannelNotServable("channel handle names no room")
        if not agent:
            raise ChannelNotServable("this channel was not provisioned to be served")
        return room, agent

    def _token(self, room: str, spec: ChannelSpec, *, ttl_seconds: int) -> str:
        """Mint a credential, and nothing more.

        The token says who the device is and what it may do. It carries no room
        configuration: server-side orchestration is declared where the room is
        declared, never routed through a credential handed to the device.
        """
        sources: list[str] = []
        if spec.audio.publishes:
            sources.append("microphone")
        if spec.video.publishes:
            sources.append("camera")
        metadata: dict[str, str] = {
            "kind": "device",
            "device_id": spec.device_id,
            "owner_id": spec.owner_id,
            "manifest_id": spec.manifest_id,
            "session_intent": "user_initiated",
        }
        if spec.serving is not None:
            metadata["interaction_mode"] = spec.serving.interaction_mode
        return (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(spec.device_id)
            .with_name(spec.device_id)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_create=False,
                    room_join=True,
                    room=room,
                    can_publish=bool(sources),
                    can_subscribe=spec.audio.subscribes or spec.video.subscribes,
                    can_publish_data=True,
                    can_publish_sources=sources or None,
                    can_update_own_metadata=False,
                )
            )
            .with_metadata(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
            .to_jwt()
        )
