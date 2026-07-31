from __future__ import annotations

from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError
from eidolon.livekit.agent import server
from eidolon.livekit.agent.server import _resolve_session_metadata


def _participant(
    identity: str,
    metadata: str = "",
    *,
    can_publish: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        identity=identity,
        metadata=metadata,
        permissions=SimpleNamespace(can_publish=can_publish),
    )


class _FakeRoom:
    def __init__(self, participants_after_connect: dict[str, SimpleNamespace]):
        self._connected = False
        self._participants_after_connect = participants_after_connect
        self.remote_participants: dict[str, SimpleNamespace] = {}

    def isconnected(self) -> bool:
        return self._connected


class _FakeContext:
    def __init__(self, participants_after_connect: dict[str, SimpleNamespace]):
        self.room = _FakeRoom(participants_after_connect)
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        self.room._connected = True
        self.room.remote_participants = self.room._participants_after_connect


@pytest.mark.asyncio
async def test_session_metadata_ignores_hub_control_participant():
    hub = _participant(
        "eidolon-hub-control-test",
        can_publish=False,
    )
    box = _participant(
        "box-3",
        (
            '{"kind":"device","device_id":"box-3",'
            '"interaction_mode":"full_duplex",'
            '"session_intent":"presence_initiated",'
            '"avatar_requested":false}'
        ),
        can_publish=True,
    )
    ctx = _FakeContext(
        {
            hub.identity: hub,
            box.identity: box,
        }
    )

    mode, intent, avatar = await _resolve_session_metadata(ctx)

    assert ctx.connect_calls == 1
    assert mode == "full_duplex"
    assert intent == "presence_initiated"
    assert avatar is False


@pytest.mark.asyncio
async def test_session_metadata_propagates_runtime_actor_failure(monkeypatch):
    ctx = _FakeContext({})

    async def _runtime_actor_unavailable(_room):
        raise DeviceTokenResolverError("runtime actor unavailable")

    monkeypatch.setattr(
        server,
        "wait_for_runtime_participant_metadata",
        _runtime_actor_unavailable,
    )

    with pytest.raises(DeviceTokenResolverError, match="runtime actor unavailable"):
        await _resolve_session_metadata(ctx)

    assert ctx.connect_calls == 1
