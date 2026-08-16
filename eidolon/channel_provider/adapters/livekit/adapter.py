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

from ...contracts import BackendUnavailable, canonical_json
from ...ports import ChannelGrant
from ...spec import ChannelSpec
from .config import LiveKitConfig

logger = logging.getLogger("eidolon.channel_provider.livekit")

ADAPTER_NAME = "livekit"
BINDING_FORMAT = "application/vnd.eidolon.livekit-session+json;v=2"

_HEALTHCHECK_ROOM = "__eidolon_channel_provider_healthcheck__"


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
        await self._declare_room(room, spec)
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
        return ChannelGrant(
            binding_format=BINDING_FORMAT,
            payload=payload,
            expires_at_ms=issued_at_ms + ttl * 1000,
            handle={"room": room},
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

    async def _declare_room(self, room: str, spec: ChannelSpec) -> None:
        """Create the room together with its full serving contract.

        LiveKit applies a room's agent dispatch when the room is created, and a
        join token's room configuration only takes effect if that join is what
        creates the room. Declaring the room here without its dispatch would
        therefore silently disable it: the token's copy would be ignored for a
        room that already exists, the automatic publisher dispatch would fall
        back to the anonymous agent pool, no worker would answer, and the device
        would sit in a room nobody ever joins. The dispatch is stated once, here,
        where the room itself is stated.
        """
        request = api.CreateRoomRequest(name=room)
        if spec.serving is not None:
            request.agents.append(api.RoomAgentDispatch(agent_name=spec.serving.agent_name))
        try:
            await self._client().room.create_room(request)
        except TwirpError as exc:
            if exc.code == TwirpErrorCode.ALREADY_EXISTS:
                await self._ensure_dispatch(room, spec)
                return
            raise BackendUnavailable("LiveKit room creation failed") from exc
        except Exception as exc:
            raise BackendUnavailable("LiveKit room creation failed") from exc

    async def _ensure_dispatch(self, room: str, spec: ChannelSpec) -> None:
        """Repair the serving contract of a room this adapter did not create.

        `create_room` returns the existing room untouched, so a room left over
        from an earlier provision keeps whatever dispatch it was born with.
        Declaring the dispatch explicitly is idempotent and converges the room
        onto the contract this spec asks for.
        """
        if spec.serving is None:
            return
        try:
            existing = await self._client().agent_dispatch.list_dispatch(room_name=room)
            if any(d.agent_name == spec.serving.agent_name for d in existing):
                return
            await self._client().agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(room=room, agent_name=spec.serving.agent_name)
            )
        except Exception as exc:
            raise BackendUnavailable("LiveKit agent dispatch declaration failed") from exc

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
