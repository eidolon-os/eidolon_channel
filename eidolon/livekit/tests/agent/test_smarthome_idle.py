"""A home command session ends on its own when nobody uses it.

2026-09-29: a device reset mid-session. Nothing else ends this profile when its
device does not: the Provider's listener keeps the room occupied, and the next
connection with the same identity is linked instead of closing the session. So
the session served the device's next run, and every conversation that run asked
for was refused as a conflict, until the host was restarted.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import livekit.agents.voice as voice
import pytest
from eidolon_sdk.biz.contracts import SESSION_END_IDLE_NORMAL, SESSION_INTENT_PRESENCE
from livekit.agents import StopResponse

from eidolon.livekit.agent import server, smarthome
from eidolon.livekit.agent import factory
from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.runtime import resolver
from eidolon.livekit.agent.runtime.services import ChannelRuntimeServices
from eidolon.interaction_context import DeviceConnectionContext, InteractionContextError


class _Stt:
    async def aclose(self) -> None:
        pass


class _Agent:
    def __init__(self, **kwargs) -> None:
        self.options = kwargs


class _Session:
    instances: list[_Session] = []

    def __init__(self, **_kwargs) -> None:
        self.handlers: dict[str, list] = {}
        self.user_state = "listening"
        self.agent_state = "listening"
        self.agent = None
        self.closed = False
        _Session.instances.append(self)

    def on(self, event: str, callback) -> None:
        self.handlers.setdefault(event, []).append(callback)

    async def start(self, *, agent, room, room_options) -> None:
        self.agent = agent

    async def aclose(self) -> None:
        self.closed = True


class _Room:
    name = "eidolon-device-test"

    def on(self, _event: str, _callback) -> None:
        pass


@pytest.fixture
def home_runtime(monkeypatch):
    _Session.instances.clear()
    services = AsyncMock()
    services.resolve_room.return_value = SimpleNamespace(
        owner_id="owner-test", companion_id="companion-test", device_id="device-instance-test",
    )
    monkeypatch.setattr(factory, "_build_runtime_services", lambda *_args, **_kwargs: services)
    monkeypatch.setattr(voice, "AgentSession", _Session)
    monkeypatch.setattr(voice, "Agent", _Agent)
    monkeypatch.setattr(SharedStageFactory, "_build_stt", staticmethod(lambda _cfg: SimpleNamespace(stt=_Stt())))

    async def participant(_room):
        return "device-instance-test"

    async def end_home_session(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr(resolver, "wait_for_runtime_participant_identity", participant)
    monkeypatch.setattr(smarthome, "end_home_session", end_home_session)


def _cfg(*, idle_ms: int):
    cfg = server.AgentConfig()
    idle = replace(cfg.turn_policy.idle, disconnect_after_idle_ms=idle_ms, disconnect_grace_ms=0)
    return replace(cfg, turn_policy=replace(cfg.turn_policy, idle=idle))


def _run(cfg, calls: dict, **kwargs):
    async def on_started() -> None:
        pass

    async def on_end(reason: str) -> None:
        calls.setdefault("end", []).append(reason)

    async def on_closed() -> None:
        calls.setdefault("closed", []).append(True)

    async def on_idle() -> None:
        calls.setdefault("idle", []).append(time.monotonic())

    return smarthome.run_smarthome_session(
        room=_Room(),
        cfg=cfg,
        prebuilt_vad=object(),
        owner_id="owner-test",
        device_ref="device-instance-test",
        session_id="esp32-test-00000001",
        on_started=on_started,
        on_end=on_end,
        on_closed=on_closed,
        on_idle=on_idle,
        **kwargs,
    )


async def test_an_unused_home_session_ends_with_an_idle_notice(home_runtime) -> None:
    calls: dict = {}

    await asyncio.wait_for(_run(_cfg(idle_ms=50), calls), timeout=2.0)

    assert calls["end"][0] == SESSION_END_IDLE_NORMAL
    assert len(calls["idle"]) == 1
    # The idle end already withdrew the dispatch; closing again would be a
    # second withdrawal of something that is gone.
    assert "closed" not in calls
    assert _Session.instances[0].closed
    assert _Session.instances[0].agent.options["llm"] is None
    assert _Session.instances[0].agent.options["tts"] is None


async def test_a_command_being_delivered_is_not_idleness(home_runtime, monkeypatch) -> None:
    delivered: list[float] = []

    async def slow_delivery(*_args, **_kwargs) -> None:
        await asyncio.sleep(0.3)
        delivered.append(time.monotonic())

    monkeypatch.setattr(smarthome, "handle_transcript", slow_delivery)
    calls: dict = {}
    running = asyncio.create_task(_run(_cfg(idle_ms=50), calls))
    while not _Session.instances or _Session.instances[0].agent is None:
        await asyncio.sleep(0)

    with pytest.raises(StopResponse):
        await _Session.instances[0].agent.on_user_turn_completed(
            None, SimpleNamespace(text_content="打开客厅灯")
        )
    await asyncio.wait_for(running, timeout=2.0)

    assert calls["idle"][0] > delivered[0]


async def test_noise_that_transcribes_to_nothing_is_not_activity(home_runtime) -> None:
    """Same rule as a Companion session: a noisy but silent panel still goes idle."""
    calls: dict = {}
    running = asyncio.create_task(_run(_cfg(idle_ms=50), calls))
    while not _Session.instances or _Session.instances[0].agent is None:
        await asyncio.sleep(0)
    handlers = _Session.instances[0].handlers["user_input_transcribed"]
    started = time.monotonic()
    while not running.done() and time.monotonic() - started < 1.0:
        for handler in handlers:
            handler(SimpleNamespace(transcript="  "))
        await asyncio.sleep(0.01)

    assert running.done()
    assert calls["end"][0] == SESSION_END_IDLE_NORMAL


async def test_a_presence_wake_keeps_the_window_its_governor_owns(home_runtime) -> None:
    """Same intent rule as a Companion session: presence is governed elsewhere."""
    calls: dict = {}
    running = asyncio.create_task(
        _run(_cfg(idle_ms=50), calls, session_intent=SESSION_INTENT_PRESENCE)
    )
    await asyncio.sleep(0.3)

    assert "idle" not in calls
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)


@pytest.mark.parametrize("assigned,owner,device", [
    (None, "owner-test", "device-instance-test"),
    ("companion-test", "other-owner", "device-instance-test"),
    ("companion-test", "owner-test", "other-device"),
])
async def test_invalid_identity_cannot_start_home_stt_or_command(
    home_runtime, monkeypatch, assigned, owner, device,
) -> None:
    runtime, mounts = AsyncMock(), AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-test", device_id="device-instance-test",
        device_ref=SimpleNamespace(device_instance_id="device-instance-test"),
        mount_revision=1, answering_companion_id=assigned,
    )
    runtime.resolve_companion.return_value = SimpleNamespace(
        owner_id=owner, companion_id="companion-test", device_id=device,
    )
    services = ChannelRuntimeServices(runtime=runtime, mounts=mounts)
    monkeypatch.setattr(factory, "_build_runtime_services", lambda *_args, **_kwargs: services)
    monkeypatch.setattr(_Room, "remote_participants", {
        "device-instance-test": SimpleNamespace(
            identity="device-instance-test",
            metadata=json.dumps({"kind": "device", "owner_id": "owner-test"}),
        ),
    }, raising=False)
    with pytest.raises(InteractionContextError):
        await _run(_cfg(idle_ms=50), {})
    assert not _Session.instances
    runtime.resolve_owner.assert_not_awaited()
    runtime.aclose.assert_awaited_once()


async def test_bound_home_terminal_uses_the_shared_resolver(home_runtime, monkeypatch) -> None:
    runtime, mounts = AsyncMock(), AsyncMock()
    mounts.resolve.return_value = DeviceConnectionContext(
        owner_id="owner-test", device_id="device-instance-test",
        device_ref=SimpleNamespace(device_instance_id="device-instance-test"),
        mount_revision=1, answering_companion_id="companion-test",
    )
    runtime.resolve_companion.return_value = SimpleNamespace(
        owner_id="owner-test", companion_id="companion-test", device_id="device-instance-test",
    )
    services = ChannelRuntimeServices(runtime=runtime, mounts=mounts)
    monkeypatch.setattr(factory, "_build_runtime_services", lambda *_args, **_kwargs: services)
    monkeypatch.setattr(_Room, "remote_participants", {
        "device-instance-test": SimpleNamespace(
            identity="device-instance-test",
            metadata=json.dumps({"kind": "device", "owner_id": "owner-test"}),
        ),
    }, raising=False)
    await asyncio.wait_for(_run(_cfg(idle_ms=50), {}), timeout=2)
    mounts.resolve.assert_awaited_once_with(owner_id="owner-test", device_id="device-instance-test")
    runtime.resolve_companion.assert_awaited_once_with("companion-test", device_id="device-instance-test")
    runtime.aclose.assert_awaited_once()
