from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from livekit import rtc
from .test_invitation_delivery import setup


@pytest.mark.parametrize("endpoint", ["device", "agent", "listener", "unrelated"])
async def test_disconnect_ends_only_current_reserved_session(monkeypatch, endpoint):
    adapter, grant, connection, _, _ = await setup(monkeypatch)
    ended = Mock()
    remove = adapter.observe_session_end(grant.handle, "native-session", ended)
    if endpoint == "listener":
        connection.drop()
    else:
        connection.handlers["participant_disconnected"](SimpleNamespace(
            identity=grant.handle["device"] if endpoint == "device" else "another",
            kind=(rtc.ParticipantKind.PARTICIPANT_KIND_AGENT if endpoint == "agent"
                  else rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD)))
    assert ended.call_count == (0 if endpoint == "unrelated" else 1)
    remove()
    await adapter.shutdown()


async def test_restarted_listener_withdraws_orphan_even_when_source_policy_matches(monkeypatch):
    import json
    from .test_livekit_adapter import _adapter, _spec, FakeDispatch
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1000)
    handle = dict(grant.handle, output_template={'inputs': {'microphone': True},
        'outputs': {'speech': False}, 'policy_revision': 1})
    room = handle['room']
    client.agent_dispatch.dispatches[room] = [FakeDispatch('orphan', handle['agent'], metadata=json.dumps({
        'output_plan': handle['output_template'], 'presentation_endpoint': {'room': 'speaker'}}))]
    await adapter._reconcile_output_dispatches(handle)
    assert client.agent_dispatch.deleted == [(room, 'orphan')]
    await adapter.shutdown()


async def test_old_remote_publisher_blocks_new_single_session(monkeypatch):
    from eidolon.channel_provider.contracts import InvalidTransition
    adapter, grant, connection, _, _ = await setup(monkeypatch)
    connection.remote_participants = {'old': SimpleNamespace(identity='presentation-old-job',
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)}
    with pytest.raises(InvalidTransition, match='still closing'):
        await adapter.open_session(grant.handle, 'new-single', session_intent='user_initiated')
    assert not adapter._api.agent_dispatch.created
    await adapter.shutdown()


@pytest.mark.parametrize('kind', ['presentation_endpoint', 'team_dispatch'])
@pytest.mark.parametrize('current', [True, False])
async def test_channel_refresh_preserves_only_owned_scene_dispatch(monkeypatch, kind, current):
    import json
    from .test_livekit_adapter import _adapter, _spec, FakeDispatch
    adapter, client = _adapter()
    grant = await adapter.open(_spec(), issued_at_ms=1000)
    handle = dict(grant.handle, output_template={'inputs': {'microphone': True},
        'outputs': {'speech': False}, 'policy_revision': 1})
    room = handle['room']
    client.agent_dispatch.dispatches[room] = [FakeDispatch('scene-worker', handle['agent'], metadata=json.dumps({
        'conversation_id': 'native-session', kind: {'room': 'speaker'}}))]
    remove = adapter.observe_session_end(handle, 'native-session' if current else 'another-session', Mock())
    await adapter._reconcile_output_dispatches(handle)
    assert client.agent_dispatch.deleted == ([] if current else [(room, 'scene-worker')])
    remove()
    await adapter.shutdown()


def _agent(identity):
    return SimpleNamespace(identity=identity, kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT)


def _session_end(connection, sender, conversation_id):
    import json
    from eidolon_sdk.biz.contracts import (
        SESSION_CONTROL_TOPIC, SESSION_CONVERSATION_ID_FIELD, SESSION_END_TYPE,
        WIRE_SCHEMA_VERSION)
    connection.handlers["data_received"](SimpleNamespace(
        topic=SESSION_CONTROL_TOPIC, participant=sender,
        data=json.dumps({"schema_v": WIRE_SCHEMA_VERSION, "type": SESSION_END_TYPE,
                         SESSION_CONVERSATION_ID_FIELD: conversation_id,
                         "reason": "user_left"}).encode()))


async def _next_conversation_with_previous_agent_still_in_room(monkeypatch):
    """The 2026-11-11 order: the earlier conversation was closed, its agent is
    still leaving, and the device has already asked for the next one."""
    adapter, grant, connection, _, _ = await setup(monkeypatch)
    previous, current = _agent("agent-AJ_previous"), _agent("agent-AJ_current")
    connection.remote_participants[previous.identity] = previous
    ended = Mock()
    remove = adapter.observe_session_end(grant.handle, "next-conversation", ended)
    # Registered first, then the dispatch is placed and its agent joins.
    await adapter.open_session(grant.handle, "next-conversation",
                               session_intent="user_initiated", device_requested=True)
    connection.remote_participants[current.identity] = current
    return adapter, grant, connection, ended, remove, previous, current


async def test_an_earlier_conversations_agent_leaving_late_does_not_end_the_next(monkeypatch):
    adapter, _, connection, ended, remove, previous, _ = (
        await _next_conversation_with_previous_agent_still_in_room(monkeypatch))
    del connection.remote_participants[previous.identity]
    connection.handlers["participant_disconnected"](previous)
    _session_end(connection, previous, "previous-conversation")
    ended.assert_not_called()
    remove()
    await adapter.shutdown()


async def test_the_next_conversations_own_agent_crashing_still_ends_it(monkeypatch):
    adapter, _, connection, ended, remove, previous, current = (
        await _next_conversation_with_previous_agent_still_in_room(monkeypatch))
    connection.handlers["participant_disconnected"](previous)
    connection.handlers["participant_disconnected"](current)  # no session_end first
    ended.assert_called_once_with("TEAM_WORKER_DISCONNECTED")
    remove()
    await adapter.shutdown()


async def test_the_next_conversation_still_ends_normally(monkeypatch):
    adapter, _, connection, ended, remove, previous, current = (
        await _next_conversation_with_previous_agent_still_in_room(monkeypatch))
    connection.handlers["participant_disconnected"](previous)
    _session_end(connection, current, "next-conversation")
    ended.assert_called_once_with()
    remove()
    await adapter.shutdown()


@pytest.mark.parametrize("endpoint", ["device", "listener"])
async def test_device_or_listener_loss_ends_the_next_conversation_with_a_lingering_agent(
        monkeypatch, endpoint):
    from eidolon.channel_provider.ports import SESSION_END_DEVICE_LEFT
    adapter, grant, connection, ended, remove, _, _ = (
        await _next_conversation_with_previous_agent_still_in_room(monkeypatch))
    if endpoint == "listener":
        connection.drop()
        ended.assert_called_once_with("CHANNEL_LISTENER_DISCONNECTED")
    else:
        connection.handlers["participant_disconnected"](SimpleNamespace(
            identity=grant.handle["device"], kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD))
        ended.assert_called_once_with(SESSION_END_DEVICE_LEFT)
    remove()
    await adapter.shutdown()


async def test_an_agent_counts_as_earlier_only_for_the_observer_it_predates(monkeypatch):
    adapter, grant, connection, _, _ = await setup(monkeypatch)
    first = _agent("agent-AJ_first")
    ended_first, ended_second = Mock(), Mock()
    remove = adapter.observe_session_end(grant.handle, "first", ended_first)
    connection.remote_participants[first.identity] = first
    connection.handlers["participant_disconnected"](first)
    ended_first.assert_called_once_with("TEAM_WORKER_DISCONNECTED")
    remove()
    remove = adapter.observe_session_end(grant.handle, "second", ended_second)
    connection.handlers["participant_disconnected"](first)
    ended_second.assert_not_called()
    remove()
    await adapter.shutdown()
