from __future__ import annotations

import base64
import json

import pytest

from eidolon.channel_provider.adapters.livekit import LiveKitChannelAdapter
from eidolon.channel_provider.contracts import ProvisionRequest
from eidolon.channel_provider.spec import derive_spec

from .helpers import audio_manifest, encoded, livekit_config, provision_payload


class FakeRoomService:
    def __init__(self) -> None:
        self.created: list[object] = []
        self.deleted: list[str] = []
        self.health_names: list[list[str]] = []

    async def create_room(self, request):
        self.created.append(request)

    async def delete_room(self, request):
        self.deleted.append(request.room)

    async def list_rooms(self, request):
        self.health_names.append(list(request.names))
        return object()


class FakeDispatchService:
    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []

    async def list_dispatch(self, room_name: str):
        return []

    async def create_dispatch(self, request):
        self.created.append((request.room, request.agent_name))


class FakeClient:
    def __init__(self) -> None:
        self.room = FakeRoomService()
        self.agent_dispatch = FakeDispatchService()

    async def aclose(self) -> None:
        pass


def _adapter() -> tuple[LiveKitChannelAdapter, FakeClient]:
    adapter = LiveKitChannelAdapter(livekit_config())
    client = FakeClient()
    adapter._api = client  # noqa: SLF001 - the constructor defers session creation
    return adapter, client


def _spec(**manifest_kwargs):
    payload = provision_payload(manifest=audio_manifest(**manifest_kwargs))
    request = ProvisionRequest.parse(encoded(payload))
    return derive_spec(request.device, agent_name="eidolon")


async def test_room_is_created_with_its_agent_dispatch() -> None:
    """The serving contract is stated where the room is stated.

    A token's room configuration is ignored for a room that already exists, so
    a room declared without its dispatch can never be served.
    """
    adapter, client = _adapter()

    await adapter.open(_spec(), issued_at_ms=1_000)

    assert len(client.room.created) == 1
    request = client.room.created[0]
    assert [agent.agent_name for agent in request.agents] == ["eidolon"]


async def test_device_without_a_microphone_gets_no_agent_and_no_publish_grant() -> None:
    """A camera that never speaks is not assigned a voice agent."""
    adapter, client = _adapter()
    spec = _spec(direction="", video="publish")
    assert spec.serving is None

    grant = await adapter.open(spec, issued_at_ms=1_000)

    assert list(client.room.created[0].agents) == []
    binding = json.loads(grant.payload)
    claims = _claims(binding["session"]["token"])
    assert claims["video"]["canPublishSources"] == ["camera"]


async def test_interaction_mode_comes_from_the_device_not_the_deployment() -> None:
    adapter, _ = _adapter()

    grant = await adapter.open(_spec(interaction_mode="ptt"), issued_at_ms=1_000)

    binding = json.loads(grant.payload)
    metadata = json.loads(_claims(binding["session"]["token"])["metadata"])
    assert metadata["interaction_mode"] == "ptt"


async def test_token_carries_no_server_side_orchestration() -> None:
    """Routing policy never travels through a credential handed to the device."""
    adapter, _ = _adapter()

    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    claims = _claims(json.loads(grant.payload)["session"]["token"])
    assert "roomConfig" not in claims
    assert "roomPreset" not in claims


async def test_binding_describes_one_session() -> None:
    adapter, _ = _adapter()

    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    binding = json.loads(grant.payload)
    assert binding["schema_version"] == 2
    assert set(binding) == {"schema_version", "session", "audio"}
    assert binding["session"]["room_name"] == grant.handle["room"]


async def test_close_releases_the_room_named_in_the_handle() -> None:
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    await adapter.close(grant.handle)

    assert client.room.deleted == [grant.handle["room"]]


def _claims(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


@pytest.mark.parametrize("handle", [{}, {"room": ""}])
async def test_close_tolerates_a_channel_that_was_never_opened(handle: dict) -> None:
    adapter, client = _adapter()

    await adapter.close(handle)

    assert client.room.deleted == []
