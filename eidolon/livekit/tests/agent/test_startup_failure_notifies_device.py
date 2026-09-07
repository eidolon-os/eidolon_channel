"""A job that dies before its session runs must still tell the device.

Real-device incident (2026-09-07 22:11:19, room
eidolon-device-5582b08fb6be2decd55a46c3): the job was dispatched, joined the
room, logged ``interaction_mode=full_duplex`` and raised one millisecond later
while constructing ``AgentSession``. The device kept its microphone open and
kept showing 「正在聆听」 for as long as the Owner cared to speak, because
nothing ever told it otherwise.

``session_end{error}`` is the reason the contract already carries for a
"failure to be served", and ``_end_serving_cb`` was written to send it. It
cannot: the framework disconnects the room *before* it runs shutdown callbacks
(measured in the same log — room disconnected at 19,330, the callback logged at
19,331), so ``publish_data`` there always raises against a dead connection. The
raise then went to ``logger.debug``, which the worker does not record, so the
one surface that could have said "the device was never told" said nothing.

These tests pin both halves: the notice goes out from the failure path while the
room is still live, and a notice that cannot go out is logged loudly.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from eidolon_sdk.biz.contracts import (
    SESSION_CONTROL_TOPIC,
    SESSION_END_ERROR,
)

from eidolon.livekit.agent import server

_CONVERSATION_ID = "mobile-576a8d4c-331f7e69-00000001"


class _LocalParticipant:
    """Records what the agent told the device, in order."""

    def __init__(self, *, fail: bool = False) -> None:
        self.identity = "agent-AJ_m3dSsqvVzD3Y"
        self.published: list[tuple[dict, str]] = []
        self._fail = fail

    async def publish_data(self, payload: bytes, *, reliable: bool, topic: str) -> None:
        if self._fail:
            raise ConnectionError("room already disconnected")
        self.published.append((json.loads(payload), topic))


class _Room:
    def __init__(self, local: _LocalParticipant | None, *, connected: bool = True) -> None:
        self.name = "eidolon-device-5582b08fb6be2decd55a46c3"
        self.local_participant = local
        self.remote_participants = {"device-instance-08b2358f": SimpleNamespace(identity="dev")}
        self._connected = connected

    def isconnected(self) -> bool:
        return self._connected


class _Context:
    def __init__(self, local: _LocalParticipant | None, *, connected: bool = True) -> None:
        self.room = _Room(local, connected=connected)
        self.job = SimpleNamespace(
            metadata=json.dumps({"conversation_id": _CONVERSATION_ID}),
            agent_name="eidolon",
            dispatch_id="AD_dpPKgMs3ZdiN",
        )
        self.proc = SimpleNamespace(userdata={})
        self.shutdown_callbacks: list = []
        self.api = SimpleNamespace(
            agent_dispatch=SimpleNamespace(delete_dispatch=self._delete_dispatch)
        )
        self.deleted_dispatches: list[str] = []

    def add_shutdown_callback(self, cb) -> None:
        self.shutdown_callbacks.append(cb)

    async def _delete_dispatch(self, *, dispatch_id: str, room_name: str) -> None:
        self.deleted_dispatches.append(dispatch_id)


def _install(monkeypatch, *, run_raises: BaseException | None) -> list:
    """Stub out everything between the job entrypoint and ``pipeline.run``."""
    from eidolon.livekit.agent import factory as factory_mod
    from eidolon.livekit.agent import full_duplex as full_duplex_mod

    runs: list = []

    class _Pipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def run(self, room) -> None:
            runs.append(room)
            if run_raises is not None:
                raise run_raises

    monkeypatch.setattr(
        factory_mod.SharedStageFactory, "from_config", classmethod(lambda cls, *a, **k: object())
    )
    monkeypatch.setattr(full_duplex_mod, "StreamingPipeline", _Pipeline)

    # The metadata bus needs a connected room and a real device; the failure
    # under test happens after it, so resolve it to a full-duplex session.
    async def _metadata(ctx):
        return ("full_duplex", "user_initiated", False)

    monkeypatch.setattr(server, "_resolve_session_metadata", _metadata)
    return runs


def _session_ends(local: _LocalParticipant) -> list[dict]:
    return [
        payload
        for payload, topic in local.published
        if topic == SESSION_CONTROL_TOPIC and payload.get("type") == "session_end"
    ]


@pytest.mark.asyncio
async def test_a_job_that_dies_before_its_session_tells_the_device(monkeypatch) -> None:
    local = _LocalParticipant()
    ctx = _Context(local)
    _install(monkeypatch, run_raises=TypeError("got an unexpected keyword argument"))

    with pytest.raises(TypeError):
        await server.run_agent(ctx, server.AgentConfig())

    assert _session_ends(local) == [
        {
            "schema_v": 1,
            "type": "session_end",
            "conversation_id": _CONVERSATION_ID,
            "reason": SESSION_END_ERROR,
        }
    ]


@pytest.mark.asyncio
async def test_the_notice_goes_out_before_the_job_is_allowed_to_fail(monkeypatch) -> None:
    """Ordering is the whole fix: after the raise escapes, the room is gone."""
    local = _LocalParticipant()
    ctx = _Context(local)
    _install(monkeypatch, run_raises=RuntimeError("startup died"))

    with pytest.raises(RuntimeError):
        await server.run_agent(ctx, server.AgentConfig())
        pytest.fail("run_agent must not swallow the failure")

    # Published during run_agent, not deferred to a shutdown callback that the
    # framework only reaches after tearing the room down.
    assert len(_session_ends(local)) == 1
    assert ctx.shutdown_callbacks, "the backstop must still be registered"


@pytest.mark.asyncio
async def test_the_shutdown_backstop_does_not_tell_the_device_twice(monkeypatch) -> None:
    local = _LocalParticipant()
    ctx = _Context(local)
    _install(monkeypatch, run_raises=RuntimeError("startup died"))

    with pytest.raises(RuntimeError):
        await server.run_agent(ctx, server.AgentConfig())

    for cb in ctx.shutdown_callbacks:
        await cb("error")

    assert len(_session_ends(local)) == 1
    assert ctx.deleted_dispatches == ["AD_dpPKgMs3ZdiN"]


@pytest.mark.asyncio
async def test_a_notice_that_cannot_be_sent_is_logged_loudly(monkeypatch, caplog) -> None:
    """The silence that cost hours: this failure used to be logged at debug."""
    local = _LocalParticipant(fail=True)
    ctx = _Context(local)
    _install(monkeypatch, run_raises=RuntimeError("startup died"))

    with caplog.at_level(logging.INFO, logger="agent_server"):
        with pytest.raises(RuntimeError):
            await server.run_agent(ctx, server.AgentConfig())

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "session_end" in r.getMessage()
    ]
    assert warnings, "a device left uninformed must be visible at INFO or above"


@pytest.mark.asyncio
async def test_a_session_that_starts_is_not_told_it_failed(monkeypatch) -> None:
    local = _LocalParticipant()
    ctx = _Context(local)
    runs = _install(monkeypatch, run_raises=None)

    await server.run_agent(ctx, server.AgentConfig())

    assert runs, "the pipeline must actually have run"
    assert _session_ends(local) == []


@pytest.mark.asyncio
async def test_an_end_the_channel_can_no_longer_carry_is_not_a_warning(monkeypatch, caplog) -> None:
    """The happy path must stay quiet or the warning above stops being read.

    Measured on hardware (2026-09-07 23:21): an end the Owner asked for arrives
    as `ParticipantRemoved` first and `engine is closed` second, so the backstop
    tries to publish into a room that is already gone. That is every ordinary
    conversation, and `reason` cannot tell it apart — `user_left` is what
    _end_serving_cb assigns to any non-error shutdown, not a claim about who
    hung up. Nobody could have delivered this notice and the device already
    knows the channel dropped, so it is not a defect and must not read as one.
    """
    local = _LocalParticipant(fail=True)
    ctx = _Context(local, connected=False)
    _install(monkeypatch, run_raises=RuntimeError("startup died"))

    with caplog.at_level(logging.INFO, logger="agent_server"):
        with pytest.raises(RuntimeError):
            await server.run_agent(ctx, server.AgentConfig())

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "a channel that is already gone must not raise a warning"
    )
    assert [r for r in caplog.records if "already gone" in r.getMessage()], (
        "but it must still leave a trace that the notice never went out"
    )
