"""Smart-home dispatch joins the room before resolving its runtime device."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent import server
from eidolon.livekit.agent import smarthome


class _Room:
    name = "eidolon-device-test"
    remote_participants = {}

    def __init__(self) -> None:
        self.connected = False
        self.published: list[dict] = []

    def isconnected(self) -> bool:
        return self.connected

    @property
    def local_participant(self):
        if not self.connected:
            raise RuntimeError("cannot access local participant before connecting")

        async def publish_data(payload, **_kwargs):
            self.published.append(json.loads(payload))

        return SimpleNamespace(identity="agent-test", publish_data=publish_data)


class _Context:
    def __init__(self) -> None:
        self.room = _Room()
        self.job = SimpleNamespace(
            metadata=json.dumps({
                "conversation_id": "esp32-test-00000001",
                "smarthome_panel": True,
                "smarthome_owner": "owner-test",
                "smarthome_device": "device-instance-test",
            }),
            agent_name="eidolon",
            dispatch_id="AD_test",
        )
        self.proc = SimpleNamespace(userdata={})
        self.shutdown_callbacks = []
        self.deleted: list[str] = []
        self.api = SimpleNamespace(agent_dispatch=SimpleNamespace(delete_dispatch=self._delete))

    async def connect(self) -> None:
        self.room.connected = True

    def add_shutdown_callback(self, callback) -> None:
        self.shutdown_callbacks.append(callback)

    async def _delete(self, *, dispatch_id, room_name) -> None:
        self.deleted.append(dispatch_id)


@pytest.mark.asyncio
async def test_smarthome_joins_before_resolving_participant(monkeypatch) -> None:
    ctx = _Context()

    async def run_smarthome_session(**kwargs):
        assert kwargs["room"].isconnected()

    monkeypatch.setattr(smarthome, "run_smarthome_session", run_smarthome_session)
    await server.run_agent(ctx, server.AgentConfig())
    assert ctx.room.connected
    assert ctx.deleted == []


@pytest.mark.asyncio
async def test_smarthome_startup_failure_withdraws_its_dispatch(monkeypatch) -> None:
    ctx = _Context()

    async def run_smarthome_session(**_kwargs):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(smarthome, "run_smarthome_session", run_smarthome_session)
    with pytest.raises(RuntimeError, match="startup failed"):
        await server.run_agent(ctx, server.AgentConfig())

    assert ctx.deleted == ["AD_test"]
    assert ctx.room.published == [{
        "schema_v": 1,
        "type": "session_end",
        "conversation_id": "esp32-test-00000001",
        "reason": "error",
    }]
