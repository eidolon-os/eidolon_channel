from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest
from livekit.protocol.agent import JobStatus

from eidolon.channel_provider.adapters.livekit import LiveKitChannelAdapter
from eidolon.channel_provider.contracts import ChannelNotServable, ProvisionRequest
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


class FakeJob:
    def __init__(self, status: int) -> None:
        self.state = SimpleNamespace(status=status)


class FakeDispatch:
    def __init__(self, dispatch_id: str, agent_name: str, jobs: list[FakeJob] | None = None):
        self.id = dispatch_id
        self.agent_name = agent_name
        # LiveKit publishes a job a beat after it starts, so a just-created
        # dispatch legitimately reports none.
        self.state = SimpleNamespace(jobs=jobs or [])


class FakeDispatchService:
    """Stands in for LiveKit's own dispatch bookkeeping.

    Seeded with the anonymous record LiveKit keeps for every room, so the tests
    exercise the same "pick out ours" matching the adapter has to do for real.
    """

    def __init__(self) -> None:
        self.dispatches: dict[str, list[FakeDispatch]] = {}
        self.created: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self._next = 0

    def _room(self, room_name: str) -> list[FakeDispatch]:
        return self.dispatches.setdefault(room_name, [FakeDispatch("AD_anon", "")])

    async def list_dispatch(self, room_name: str):
        return list(self._room(room_name))

    async def create_dispatch(self, request):
        self._next += 1
        self._room(request.room).append(FakeDispatch(f"AD_{self._next}", request.agent_name))
        self.created.append((request.room, request.agent_name))

    async def delete_dispatch(self, dispatch_id: str, room_name: str):
        room = self._room(room_name)
        room[:] = [d for d in room if d.id != dispatch_id]
        self.deleted.append((room_name, dispatch_id))


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


async def test_opening_a_channel_summons_no_agent() -> None:
    """A device sitting in its channel is not yet a conversation.

    A dispatch declared at room creation would be acted on as soon as the device
    connected, and the device now never disconnects — so the room is born bare
    and the agent is asked for per session.
    """
    adapter, client = _adapter()

    await adapter.open(_spec(), issued_at_ms=1_000)

    assert len(client.room.created) == 1
    assert list(client.room.created[0].agents) == []
    assert client.agent_dispatch.created == []


async def test_device_without_a_microphone_gets_no_agent_and_no_publish_grant() -> None:
    """A camera that never speaks is not assigned a voice agent."""
    adapter, client = _adapter()
    spec = _spec(direction="", video="publish")
    assert spec.serving is None

    grant = await adapter.open(spec, issued_at_ms=1_000)

    assert list(client.room.created[0].agents) == []
    assert "agent" not in grant.handle
    binding = json.loads(grant.payload)
    claims = _claims(binding["session"]["token"])
    assert claims["video"]["canPublishSources"] == ["camera"]


async def test_a_session_brings_the_agent_and_ending_it_keeps_the_channel() -> None:
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    room = grant.handle["room"]

    await adapter.open_session(grant.handle)
    assert client.agent_dispatch.created == [(room, "eidolon")]

    await adapter.close_session(grant.handle)
    assert client.agent_dispatch.deleted == [(room, "AD_1")]
    # The device's way back in must survive the end of a conversation.
    assert client.room.deleted == []


async def test_opening_a_session_twice_leaves_one_session() -> None:
    """A device that retries must not end up served twice."""
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    await adapter.open_session(grant.handle)
    await adapter.open_session(grant.handle)

    assert len(client.agent_dispatch.created) == 1


async def test_closing_a_session_nobody_is_serving_is_success() -> None:
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)

    await adapter.close_session(grant.handle)

    assert client.agent_dispatch.deleted == []


async def test_livekits_own_dispatch_records_are_left_alone() -> None:
    """Only the dispatch this adapter placed is ours to withdraw."""
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    room = grant.handle["room"]

    await adapter.open_session(grant.handle)
    await adapter.close_session(grant.handle)

    assert [d.agent_name for d in client.agent_dispatch.dispatches[room]] == [""]


async def test_a_session_that_is_still_starting_is_not_restarted() -> None:
    """The job LiveKit has not published yet must not be mistaken for a dead one."""
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    await adapter.open_session(grant.handle)

    await adapter.open_session(grant.handle)

    assert len(client.agent_dispatch.created) == 1
    assert client.agent_dispatch.deleted == []


@pytest.mark.parametrize(
    "statuses",
    [
        [JobStatus.JS_RUNNING],
        [JobStatus.JS_PENDING],
        # One job still going is enough to keep the whole dispatch alive.
        [JobStatus.JS_SUCCESS, JobStatus.JS_RUNNING],
    ],
)
async def test_a_live_session_is_left_alone(statuses) -> None:
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    room = grant.handle["room"]
    client.agent_dispatch._room(room).append(
        FakeDispatch("AD_live", "eidolon", [FakeJob(s) for s in statuses])
    )

    await adapter.open_session(grant.handle)

    assert client.agent_dispatch.created == []


@pytest.mark.parametrize(
    "statuses", [[JobStatus.JS_SUCCESS], [JobStatus.JS_FAILED], [JobStatus.JS_SUCCESS] * 2]
)
async def test_a_spent_dispatch_cannot_block_the_device_forever(statuses) -> None:
    """A teardown that failed to withdraw its dispatch must still be recoverable.

    LiveKit never hands out a second job for one dispatch record (measured), so
    a record left behind by a finished job is inert — and would silently read as
    "already served" on every later request.
    """
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1_000)
    room = grant.handle["room"]
    client.agent_dispatch._room(room).append(
        FakeDispatch("AD_spent", "eidolon", [FakeJob(s) for s in statuses])
    )

    await adapter.open_session(grant.handle)

    assert client.agent_dispatch.deleted == [(room, "AD_spent")]
    assert client.agent_dispatch.created == [(room, "eidolon")]


async def test_a_channel_that_carries_no_conversation_cannot_hold_a_session() -> None:
    adapter, _ = _adapter()
    grant = await adapter.open(_spec(direction="", video="publish"), issued_at_ms=1_000)

    with pytest.raises(ChannelNotServable):
        await adapter.open_session(grant.handle)


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
