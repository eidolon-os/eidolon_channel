"""Realising a device channel as a LiveKit room.

Everything LiveKit-shaped lives behind this file: room naming, grants, agent
dispatch and the binding's wire shape. The service above it never learns that
rooms exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import timedelta
from typing import Any

from livekit import api
from livekit.api.twirp_client import TwirpError, TwirpErrorCode
from livekit.protocol.agent import JobStatus

from ...contracts import BackendUnavailable, ChannelNotServable, canonical_json
from ...ports import ChannelGrant
from ...spec import ChannelSpec
from .config import LiveKitConfig

logger = logging.getLogger("eidolon.channel_provider.livekit")

ADAPTER_NAME = "livekit"
BINDING_FORMAT = "application/vnd.eidolon.livekit-session+json;v=2"

_HEALTHCHECK_ROOM = "__eidolon_channel_provider_healthcheck__"
_ENDED_JOB_STATUSES = frozenset({JobStatus.JS_SUCCESS, JobStatus.JS_FAILED})


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
        handle: dict[str, Any] = {"room": room}
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

    async def shutdown(self) -> None:
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
