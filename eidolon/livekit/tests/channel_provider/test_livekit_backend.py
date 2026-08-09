from __future__ import annotations

import json

import jwt
from livekit.api.twirp_client import TwirpError, TwirpErrorCode

from eidolon.livekit.channel_provider.livekit_backend import LiveKitChannelBackend

from .helpers import livekit_config


class FakeRoomService:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.health_names: list[list[str]] = []

    async def list_rooms(self, request) -> None:
        self.health_names.append(list(request.names))

    async def create_room(self, request) -> None:
        self.created.append(request.name)
        if request.name == "existing-control":
            raise TwirpError(TwirpErrorCode.ALREADY_EXISTS, "exists", status=409)

    async def delete_room(self, request) -> None:
        self.deleted.append(request.room)
        if request.room == "missing-control":
            raise TwirpError(TwirpErrorCode.NOT_FOUND, "missing", status=404)


class FakeLiveKitApi:
    def __init__(self) -> None:
        self.room = FakeRoomService()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _claims(token: str, secret: str) -> dict:
    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        options={"verify_aud": False, "verify_exp": False, "verify_nbf": False},
    )


def test_binding_has_esp_schema_and_least_privilege_livekit_grants() -> None:
    config = livekit_config(grant_ttl_seconds=900)
    backend = LiveKitChannelBackend(config)

    binding = backend.build_binding(
        active_room="room-voice",
        control_room="room-control",
        device_id="device-1",
        owner_id="owner-1",
        issued_at_ms=1_700_000_000_000,
    )

    value = json.loads(binding.payload)
    assert value == {
        "schema_version": 1,
        "active": {
            "server_url": "wss://livekit.example.test",
            "token": value["active"]["token"],
            "identity": "device-1",
            "room_name": "room-voice",
        },
        "control": {
            "server_url": "wss://livekit.example.test",
            "token": value["control"]["token"],
            "identity": "device-1",
            "room_name": "room-control",
        },
        "audio": {"sample_rate": 16000, "channels": 1},
    }
    assert binding.expires_at_ms == 1_700_000_900_000
    assert backend._api is None  # the administrative client remains event-loop lazy

    active = _claims(value["active"]["token"], config.api_secret)
    control = _claims(value["control"]["token"], config.api_secret)
    metadata = json.loads(active["metadata"])
    assert metadata == {
        "kind": "device",
        "device_id": "device-1",
        "owner_id": "owner-1",
        "interaction_mode": "full_duplex",
        "session_intent": "user_initiated",
    }
    assert json.loads(control["metadata"]) == metadata
    assert active["video"] == {
        "roomCreate": False,
        "roomJoin": True,
        "room": "room-voice",
        "canPublish": True,
        "canSubscribe": True,
        "canPublishData": True,
        "canPublishSources": ["microphone"],
        "canUpdateOwnMetadata": False,
    }
    assert active["roomConfig"] == {"agents": [{"agentName": "eidolon"}]}
    assert control["video"] == {
        "roomCreate": False,
        "roomJoin": True,
        "room": "room-control",
        "canPublish": False,
        "canSubscribe": False,
        "canPublishData": True,
        "canUpdateOwnMetadata": False,
    }
    assert "roomConfig" not in control
    assert active["exp"] - active["nbf"] == 900


async def test_room_lifecycle_is_explicit_and_not_found_is_idempotent() -> None:
    backend = LiveKitChannelBackend(livekit_config())
    client = FakeLiveKitApi()
    backend._api = client

    await backend.healthcheck()
    await backend.ensure_rooms("new-voice", "existing-control")
    await backend.revoke_rooms("existing-voice", "missing-control")
    await backend.close()

    assert client.room.created == ["new-voice", "existing-control"]
    assert client.room.deleted == ["existing-voice", "missing-control"]
    assert client.room.health_names == [["__eidolon_channel_provider_healthcheck__"]]
    assert client.closed is True
