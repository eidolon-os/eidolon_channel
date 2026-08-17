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
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from eidolon_sdk.biz.contracts import (
    SESSION_CLOSE_TYPE,
    SESSION_CONTROL_TOPIC,
    SESSION_OPEN_TYPE,
)
from livekit import api, rtc
from livekit.api.twirp_client import TwirpError, TwirpErrorCode
from livekit.protocol.agent import JobStatus

from ...contracts import BackendUnavailable, ChannelNotServable, canonical_json
from ...ports import ChannelGrant, ServingRequest, ServingRequestSink
from ...spec import ChannelSpec
from .config import LiveKitConfig

logger = logging.getLogger("eidolon.channel_provider.livekit")

ADAPTER_NAME = "livekit"
BINDING_FORMAT = "application/vnd.eidolon.livekit-session+json;v=2"

_HEALTHCHECK_ROOM = "__eidolon_channel_provider_healthcheck__"
_ENDED_JOB_STATUSES = frozenset({JobStatus.JS_SUCCESS, JobStatus.JS_FAILED})
_SESSION_REQUESTS = {
    SESSION_OPEN_TYPE: ServingRequest.START,
    SESSION_CLOSE_TYPE: ServingRequest.STOP,
}
_REJOIN_BASE_DELAY = 1.0
_REJOIN_MAX_DELAY = 30.0


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
                    "server_url": self._config.client_url,
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
        delay = min(_REJOIN_MAX_DELAY, _REJOIN_BASE_DELAY * (2**watch.attempt))
        watch.attempt += 1
        logger.warning(
            "lost room=%s (%s); rejoining in %.0fs (attempt %d)",
            room, reason, delay, watch.attempt,
        )

        async def _rejoin() -> None:
            while self._listeners.get(room) is watch:
                await asyncio.sleep(delay)
                if self._listeners.get(room) is not watch:
                    return
                try:
                    await self._join(room, watch)
                except Exception:
                    logger.warning("room=%s still unreachable; will keep trying", room)
                    watch.attempt += 1
                    continue
                logger.info("listening to room=%s again", room)
                return

        watch.retry = asyncio.create_task(_rejoin())

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
        return _SESSION_REQUESTS.get(body.get("type"))

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

    async def open_session(self, handle: dict[str, Any]) -> None:
        room, agent = self._serving(handle)
        try:
            for dispatch in await self._client().agent_dispatch.list_dispatch(room_name=room):
                if dispatch.agent_name != agent:
                    continue
                if not _is_spent(dispatch):
                    return
                # A record whose work is over is not a session, and leaving it
                # here would let it stand in for one forever: every later
                # request would read it as "already served" and the device would
                # never be heard again. Clearing it is what makes a teardown
                # that failed to withdraw its own dispatch recoverable.
                logger.info("clearing spent dispatch=%s on room=%s", dispatch.id, room)
                await self._client().agent_dispatch.delete_dispatch(
                    dispatch_id=dispatch.id, room_name=room
                )
            await self._client().agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(room=room, agent_name=agent)
            )
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch failed") from exc
        logger.info("opened session on room=%s agent=%s", room, agent)

    async def close_session(self, handle: dict[str, Any]) -> None:
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
                if dispatch.agent_name == agent:
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
            "device_kind": spec.device_kind,
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
