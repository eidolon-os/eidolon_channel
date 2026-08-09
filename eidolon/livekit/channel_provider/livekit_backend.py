"""LiveKit room and credential ownership for the Channel Provider."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from livekit import api
from livekit.api.twirp_client import TwirpError, TwirpErrorCode

from .config import LiveKitConfig
from .contracts import BackendUnavailable, canonical_json

logger = logging.getLogger("eidolon.channel_provider.livekit")


@dataclass(frozen=True, slots=True)
class LiveKitBinding:
    payload: bytes
    expires_at_ms: int

    def __repr__(self) -> str:
        return "LiveKitBinding(payload=<redacted>, expires_at_ms=%r)" % self.expires_at_ms


class ChannelBackend(Protocol):
    async def healthcheck(self) -> None: ...

    async def ensure_rooms(self, active_room: str, control_room: str) -> None: ...

    async def revoke_rooms(self, active_room: str, control_room: str) -> None: ...

    def build_binding(
        self,
        *,
        active_room: str,
        control_room: str,
        device_id: str,
        owner_id: str,
        issued_at_ms: int,
    ) -> LiveKitBinding: ...

    async def close(self) -> None: ...


class LiveKitChannelBackend:
    def __init__(self, config: LiveKitConfig) -> None:
        self._config = config
        self._api: api.LiveKitAPI | None = None

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
            await self._client().room.list_rooms(
                api.ListRoomsRequest(names=["__eidolon_channel_provider_healthcheck__"])
            )
        except Exception as exc:
            raise BackendUnavailable("LiveKit health check failed") from exc

    async def ensure_rooms(self, active_room: str, control_room: str) -> None:
        for room in (active_room, control_room):
            try:
                await self._client().room.create_room(api.CreateRoomRequest(name=room))
            except TwirpError as exc:
                if exc.code == TwirpErrorCode.ALREADY_EXISTS:
                    continue
                raise BackendUnavailable("LiveKit room creation failed") from exc
            except Exception as exc:
                raise BackendUnavailable("LiveKit room creation failed") from exc

    async def revoke_rooms(self, active_room: str, control_room: str) -> None:
        for room in (active_room, control_room):
            try:
                await self._client().room.delete_room(api.DeleteRoomRequest(room=room))
            except TwirpError as exc:
                if exc.code == TwirpErrorCode.NOT_FOUND:
                    continue
                raise BackendUnavailable("LiveKit room revocation failed") from exc
            except Exception as exc:
                raise BackendUnavailable("LiveKit room revocation failed") from exc

    def build_binding(
        self,
        *,
        active_room: str,
        control_room: str,
        device_id: str,
        owner_id: str,
        issued_at_ms: int,
    ) -> LiveKitBinding:
        ttl = self._config.grant_ttl_seconds
        metadata = {
            "kind": "device",
            "device_id": device_id,
            "owner_id": owner_id,
            "interaction_mode": self._config.interaction_mode,
            "session_intent": "user_initiated",
        }
        active = self._token(
            room=active_room,
            identity=device_id,
            metadata=metadata,
            ttl_seconds=ttl,
            publish=True,
            subscribe=True,
            dispatch_agent=True,
        )
        control = self._token(
            room=control_room,
            identity=device_id,
            metadata=metadata,
            ttl_seconds=ttl,
            publish=False,
            subscribe=False,
            dispatch_agent=False,
        )
        expires_at_ms = issued_at_ms + ttl * 1000
        payload = canonical_json(
            {
                "schema_version": 1,
                "active": {
                    "server_url": self._config.client_url,
                    "token": active,
                    "identity": device_id,
                    "room_name": active_room,
                },
                "control": {
                    "server_url": self._config.client_url,
                    "token": control,
                    "identity": device_id,
                    "room_name": control_room,
                },
                "audio": {
                    "sample_rate": self._config.sample_rate,
                    "channels": self._config.channels,
                },
            }
        ).encode()
        return LiveKitBinding(payload=payload, expires_at_ms=expires_at_ms)

    def _token(
        self,
        *,
        room: str,
        identity: str,
        metadata: dict[str, str],
        ttl_seconds: int,
        publish: bool,
        subscribe: bool,
        dispatch_agent: bool,
    ) -> str:
        builder = (
            api.AccessToken(self._config.api_key, self._config.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_ttl(timedelta(seconds=ttl_seconds))
            .with_grants(
                api.VideoGrants(
                    room_create=False,
                    room_join=True,
                    room=room,
                    can_publish=publish,
                    can_subscribe=subscribe,
                    can_publish_data=True,
                    can_publish_sources=["microphone"] if publish else None,
                    can_update_own_metadata=False,
                )
            )
            .with_metadata(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
        )
        if dispatch_agent:
            builder = builder.with_room_config(
                api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name=self._config.agent_name)]
                )
            )
        return builder.to_jwt()

    async def close(self) -> None:
        if self._api is not None:
            await self._api.aclose()
