from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.runtime.resolver import DeviceTokenResolverError
from eidolon.livekit.agent import server
from eidolon.livekit.agent.server import (
    _resolve_runtime_session_id,
    _resolve_session_intent,
    _resolve_session_metadata,
    _session_lifecycle_payload,
)
from eidolon_sdk.biz.contracts import (
    SESSION_INTENT_PRESENCE,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
)


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
    def __init__(
        self,
        participants_after_connect: dict[str, SimpleNamespace],
        *,
        dispatch_metadata: str = '{"conversation_id":"conversation-1"}',
    ):
        self.room = _FakeRoom(participants_after_connect)
        self.job = SimpleNamespace(metadata=dispatch_metadata)
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1
        self.room._connected = True
        self.room.remote_participants = self.room._participants_after_connect


def test_runtime_session_id_comes_from_dispatch_metadata() -> None:
    ctx = _FakeContext({}, dispatch_metadata='{"conversation_id":"conversation-7"}')

    assert _resolve_runtime_session_id(ctx) == "conversation-7"


@pytest.mark.parametrize("metadata", ["", "[]", "{}", '{"conversation_id":"bad id"}'])
def test_invalid_dispatch_conversation_id_is_rejected(metadata: str) -> None:
    ctx = _FakeContext({}, dispatch_metadata=metadata)

    with pytest.raises(ValueError, match="dispatch"):
        _resolve_runtime_session_id(ctx)


def test_session_lifecycle_payload_keeps_the_conversation_correlation() -> None:
    started = json.loads(_session_lifecycle_payload("session_started", "conversation-7"))
    ended = json.loads(
        _session_lifecycle_payload("session_end", "conversation-7", reason="idle_normal_end")
    )

    assert started == {
        "schema_v": 1,
        "type": "session_started",
        "conversation_id": "conversation-7",
    }
    assert ended == {
        "schema_v": 1,
        "type": "session_end",
        "conversation_id": "conversation-7",
        "reason": "idle_normal_end",
    }


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

    mode, avatar = await _resolve_session_metadata(ctx)

    assert ctx.connect_calls == 1
    assert mode == "full_duplex"
    assert avatar is False


@pytest.mark.parametrize(
    "intent",
    [SESSION_INTENT_USER_INITIATED, SESSION_INTENT_PRESENCE, SESSION_INTENT_PROACTIVE],
)
def test_session_intent_comes_from_the_dispatch(intent: str) -> None:
    """Why a session exists is read off the bus only the Provider can write."""
    ctx = _FakeContext(
        {},
        dispatch_metadata=json.dumps(
            {"conversation_id": "conversation-1", "session_intent": intent}
        ),
    )

    assert _resolve_session_intent(ctx) == intent


@pytest.mark.asyncio
async def test_a_participant_cannot_declare_its_own_session_intent() -> None:
    """A body claiming a presence wake in its own metadata is ignored.

    `presence_initiated` buys an externally governed renewable Owner lease. If
    the participant bus could grant it, a compromised body would only have to
    say so. The credential it is issued carries no intent and forbids it
    rewriting its own metadata; this pins the other end — even a participant
    that somehow says it is not read.
    """
    box = _participant(
        "box-3",
        (
            '{"kind":"device","device_id":"box-3",'
            '"interaction_mode":"full_duplex",'
            '"session_intent":"presence_initiated"}'
        ),
        can_publish=True,
    )
    ctx = _FakeContext(
        {box.identity: box},
        dispatch_metadata='{"conversation_id":"conversation-1"}',
    )

    mode, _avatar = await _resolve_session_metadata(ctx)

    assert mode == "full_duplex"  # the actor still describes its own hardware
    assert _resolve_session_intent(ctx) == SESSION_INTENT_USER_INITIATED


@pytest.mark.parametrize(
    "metadata",
    [
        '{"conversation_id":"conversation-1"}',
        '{"conversation_id":"conversation-1","session_intent":"presence-initiated"}',
        '{"conversation_id":"conversation-1","session_intent":""}',
        "not-json",
        "",
    ],
)
def test_an_unstated_intent_is_an_ordinary_session(metadata: str) -> None:
    """Silence degrades toward less privilege, never more.

    The Provider's control contract rejects a misspelt intent outright, where
    an authenticated caller is there to be told. By the time a dispatch reaches
    this process the only thing left to do with a value nobody can explain is
    to assume the session is user-driven.
    """
    ctx = _FakeContext({}, dispatch_metadata=metadata)

    assert _resolve_session_intent(ctx) == SESSION_INTENT_USER_INITIATED


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


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['full_duplex', 'half_duplex', 'ptt'])
@pytest.mark.parametrize('owner', ['channel', 'livekit_native_adaptive'])
async def test_job_constructs_intent_only_for_its_selected_mode(monkeypatch, mode, owner):
    """Global opt-in must not allocate a model for a non-consuming session."""
    from dataclasses import replace
    from unittest.mock import AsyncMock, MagicMock

    from eidolon.livekit.agent import full_duplex
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.half_duplex import pipeline as ptt_module
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig

    cfg = EffectiveAgentConfig()
    cfg = replace(cfg, turn_policy=replace(cfg.turn_policy, interruption_owner=owner,
        interrupt=replace(cfg.turn_policy.interrupt, intent_provider='llm')))
    created = []
    build_model = MagicMock(return_value=SimpleNamespace(aclose=AsyncMock()))
    monkeypatch.setattr(SharedStageFactory, '_build_llm', build_model)

    def factory_from_config(session_cfg, **kwargs):
        factory = SimpleNamespace(
            interrupt_classifier=SharedStageFactory.build_interrupt_classifier(session_cfg))
        created.append((session_cfg, factory))
        return factory

    monkeypatch.setattr(SharedStageFactory, 'from_config', factory_from_config)
    resolve = AsyncMock(return_value=(mode, False))
    monkeypatch.setattr(server, '_resolve_session_metadata', resolve)
    streaming = MagicMock(return_value=SimpleNamespace(run=AsyncMock()))
    ptt = MagicMock(return_value=SimpleNamespace(run=AsyncMock()))
    monkeypatch.setattr(full_duplex, 'StreamingPipeline', streaming)
    monkeypatch.setattr(ptt_module, 'HalfDuplexPttPipeline', ptt)
    ctx = _FakeContext({})
    ctx.room.name = 'mode-boundary'
    ctx.proc = SimpleNamespace(userdata={})
    ctx.add_shutdown_callback = MagicMock()

    await server.run_agent(ctx, cfg)

    resolve.assert_awaited_once_with(ctx)
    assert len(created) == 1
    _, factory = created[0]
    consumes = mode == 'full_duplex' and owner == 'channel'
    assert build_model.call_count == int(consumes)
    assert (factory.interrupt_classifier is not None) is consumes
    assert cfg.turn_policy.interrupt.intent_provider == 'llm', 'shared config was mutated'
    selected = ptt if mode == 'ptt' else streaming
    unused = streaming if mode == 'ptt' else ptt
    unused.assert_not_called()
    assert selected.call_args.args[0] is factory
    if mode != 'ptt':
        assert selected.call_args.kwargs['allow_interruptions'] is (mode == 'full_duplex')
    selected.return_value.run.assert_awaited_once_with(ctx.room)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent", [SESSION_INTENT_PRESENCE, SESSION_INTENT_PROACTIVE, SESSION_INTENT_USER_INITIATED]
)
async def test_the_dispatched_intent_reaches_the_pipeline(monkeypatch, intent):
    """The last hop: what the Provider stamped is what the session runs under.

    Resolving the intent correctly and then building the pipeline with a
    different one would leave every other test in this file green while the
    wake still behaved like an ordinary session — which is exactly the failure
    this whole change exists to remove. So this exercises the real
    `_resolve_session_intent` (only the other bus is stubbed) and reads back
    what the pipeline was actually constructed with.
    """
    from unittest.mock import AsyncMock, MagicMock

    from eidolon.livekit.agent import full_duplex
    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig

    monkeypatch.setattr(
        SharedStageFactory, "from_config", lambda *a, **k: SimpleNamespace()
    )
    streaming = MagicMock(return_value=SimpleNamespace(run=AsyncMock()))
    monkeypatch.setattr(full_duplex, "StreamingPipeline", streaming)
    # Only the participant bus is stubbed; the intent must come the real way.
    monkeypatch.setattr(
        server, "_resolve_session_metadata", AsyncMock(return_value=("full_duplex", False))
    )
    ctx = _FakeContext(
        {},
        dispatch_metadata=json.dumps(
            {"conversation_id": "conversation-1", "session_intent": intent}
        ),
    )
    ctx.room.name = "intent-reaches-pipeline"
    ctx.proc = SimpleNamespace(userdata={})
    ctx.add_shutdown_callback = MagicMock()

    await server.run_agent(ctx, EffectiveAgentConfig())

    assert streaming.call_args.kwargs["session_intent"] == intent


@pytest.mark.asyncio
async def test_failed_actor_resolution_does_not_allocate_provider_clients(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.common.config.schema import EffectiveAgentConfig

    build = MagicMock()
    monkeypatch.setattr(SharedStageFactory, 'from_config', build)
    monkeypatch.setattr(server, '_resolve_session_metadata', AsyncMock(
        side_effect=DeviceTokenResolverError('runtime actor unavailable')))
    ctx = _FakeContext({})
    ctx.room.name = 'missing-actor'
    ctx.proc = SimpleNamespace(userdata={})
    with pytest.raises(DeviceTokenResolverError, match='runtime actor unavailable'):
        await server.run_agent(ctx, EffectiveAgentConfig())
    build.assert_not_called()
