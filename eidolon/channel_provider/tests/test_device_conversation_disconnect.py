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
