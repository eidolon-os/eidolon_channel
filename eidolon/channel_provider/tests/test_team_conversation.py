import pytest

from eidolon_sdk.biz.control.coordination_stream import OpenScene
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import ProvisionRequest
from .helpers import provision_payload, encoded
from .test_device_conversation import Adapter, wait_for
from .test_service import _service


class TeamAdapter(Adapter):
    def observe_team_state(self, handle, conversation_id, session_id, callback):
        self.report_round = callback
        return lambda: None

    async def require_team_input(self, handle):
        pass

    async def open_team_session(self, opened, handles, sessions):
        self.team = (opened, handles, sessions)
        self.launched.set()


async def test_non_ptt_input_is_rejected_before_any_device_wakes():
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from eidolon.channel_provider.contracts import InvalidTransition
    from eidolon.channel_provider.team_conversation import TeamConversation
    backend = TeamAdapter()
    backend.require_team_input = AsyncMock(side_effect=InvalidTransition('TEAM_INPUT_REQUIRES_PTT'))
    visit = TeamConversation(SimpleNamespace(selection=SimpleNamespace(session_id='team')),
                             backend, ({'device': 'input'}, {'device': 'output'}))
    await visit.run()
    assert visit.state == 'failed'
    assert visit.error == 'TEAM_INPUT_REQUIRES_PTT'
    assert not backend.sent and not visit.sessions
    assert not backend.launched.is_set()


@pytest.mark.parametrize('metadata,accepted', [
    ('{"interaction_mode":"ptt"}', True),
    ('{"interaction_mode":"full_duplex"}', False),
    ('{}', False), ('[]', False), ('broken', False),
])
async def test_team_input_preflight_uses_actual_selected_actor(metadata, accepted):
    from types import SimpleNamespace
    from .test_livekit_adapter import _adapter
    from eidolon.channel_provider.contracts import InvalidTransition
    adapter, client = _adapter()
    client.room.participants = [
        SimpleNamespace(identity='bridge', metadata='{"interaction_mode":"ptt"}'),
        SimpleNamespace(identity='input', metadata=metadata),
    ]
    if accepted:
        await adapter.require_team_input({'room': 'room', 'device': 'input'})
    else:
        with pytest.raises(InvalidTransition, match='TEAM_INPUT_REQUIRES_PTT'):
            await adapter.require_team_input({'room': 'room', 'device': 'input'})
    await adapter.shutdown()


@pytest.mark.parametrize("close_failure", [False, True])
async def test_three_device_team_http_start_status_close_and_shared_occupancy(tmp_path, close_failure):
    from aiohttp.test_utils import TestClient, TestServer
    from eidolon.channel_provider.http import create_app
    backend = TeamAdapter()
    service, _, _ = _service(tmp_path, [1700000000000], backend)
    refs = []
    for name in ('input', 'a', 'b'):
        req = ProvisionRequest.parse(encoded(provision_payload(device_id=named_device_instance_id(name))))
        await service.provision(req)
        refs.append(req.device_ref)
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', selection=dict(scenario='ip_role_group', session_id='team', input_device=refs[0],
            members=[dict(companion_id=key, output_device=ref) for key, ref in zip(('a', 'b'), refs[1:])],
            goal="讨论旅行", reply_budget=4)))
    token = 'test-team-service-token-at-least-32-bytes'
    async with TestClient(TestServer(create_app(service=service, bearer_token=token))) as client:
        body = opened.model_dump(mode='json')
        assert (await client.post('/v1/role-groups/open', json=body)).status == 401
        assert not backend.sent
        headers = {'Authorization': 'Bearer ' + token}
        response = await client.post('/v1/role-groups/open', json=body, headers=headers)
        assert response.status == 200, await response.text()
        visit = service._team_conversations[('owner_1', 'team')]
        await wait_for(lambda: visit.state == 'ready')
        assert len(backend.team[1]) == len(backend.team[2]) == 3
        assert len(service._transport_scopes) == 3
        assert not backend.directed and not backend.sessions_opened
        query = dict(owner_id='owner_1', session_id='team')
        response = await client.post('/v1/role-groups/status', json=query, headers=headers)
        assert (await response.json())['state'] == 'ready'
        if close_failure:
            async def offline_input(handle, *args, **kwargs):
                if handle['device'] == refs[0].device_instance_id:
                    raise RuntimeError('input ACK timeout')
            backend.end_prepared_session.side_effect = offline_input
            response = await client.post('/v1/role-groups/close', json=query, headers=headers)
            assert response.status == 200, await response.text()
            assert (await response.json())['state'] == 'closing'
            await wait_for(lambda: visit.close_task.done())
            assert len(service._transport_scopes) == 3
            response = await client.post('/v1/role-groups/status', json=query, headers=headers)
            status = await response.json()
            assert status['state'] == 'failed'
            assert refs[0].device_instance_id in status['error']
            backend.end_prepared_session.side_effect = None
        response = await client.post('/v1/role-groups/close', json=query, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())['state'] == 'closing'
        await wait_for(lambda: visit.close_task.done())
        status = visit.snapshot()
        assert status['state'] == 'closed'
        assert status['error'] == ''
        assert not service._transport_scopes
        assert backend.end_prepared_session.await_count == (6 if close_failure else 3)


async def test_real_adapter_creates_single_explicit_team_dispatch():
    from .test_livekit_adapter import _adapter
    from eidolon_sdk.biz.presentation import InputSelection, OutputSelection, SessionOutputPlan
    from eidolon.livekit.common.team_dispatch import TeamDispatch
    adapter, client = _adapter()
    refs = [ProvisionRequest.parse(encoded(provision_payload(
        device_id=named_device_instance_id(name)))).device_ref for name in ('input', 'a', 'b')]
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', selection=dict(scenario='ip_role_group', session_id='demo', input_device=refs[0], members=[
            dict(companion_id=key, output_device=ref) for key, ref in zip(('a','b'), refs[1:])])) )
    handles = tuple(dict(device=ref.device_instance_id, room='room-' + str(i), agent='eidolon',
        output_template=SessionOutputPlan(session_id='unused', policy_revision=1,
            inputs=InputSelection(microphone=True), outputs=OutputSelection(speech=True))
                .model_dump(mode='json', exclude={'session_id'})) for i, ref in enumerate(refs))
    sessions = {h['device']: 'native-' + str(i) for i, h in enumerate(handles)}
    await adapter.open_team_session(opened, handles, sessions)
    assert len(client.agent_dispatch.created) == 1
    import json
    metadata = json.loads(client.agent_dispatch.created[0][2])
    team = TeamDispatch.model_validate(metadata['team_dispatch'])
    assert not team.input_plan.outputs.can_respond
    assert all(not e.plan.inputs.microphone for e in team.endpoints)
    assert team.opened.selection == opened.selection
    assert 'presentation_endpoint' not in metadata and 'target_companion_id' not in metadata
    await adapter.open_team_session(opened, handles, sessions)
    assert len(client.agent_dispatch.created) == 1  # Idempotent dispatch.
    await adapter.shutdown()

async def test_legacy_order_checkpoint_recovers_closure_only(tmp_path):
    from eidolon.channel_provider.contracts import BackendUnavailable
    backend = TeamAdapter()
    service, store, _ = _service(tmp_path, [1700000000000], backend)
    refs = []
    for name in ('input', 'a'):
        req = ProvisionRequest.parse(encoded(provision_payload(device_id=named_device_instance_id(name))))
        await service.provision(req)
        refs.append(req.device_ref)
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', selection=dict(
        scenario='ip_role_group', session_id='legacy', input_device=refs[0],
        members=[dict(companion_id='a', output_device=refs[1])])) )
    await service.open_team_conversation(opened, authenticated_owner_id='owner_1')
    visit = service._team_conversations[('owner_1', 'legacy')]
    await wait_for(lambda: visit.state == 'ready')
    pending = refs[0].device_instance_id
    async def lost_ack(handle, *args, **kwargs):
        if handle['device'] == pending:
            raise BackendUnavailable('lost terminal ACK')
    backend.end_prepared_session.side_effect = lost_ack
    backend.observers[pending]()
    with pytest.raises(BackendUnavailable):
        await visit.task
    owner, session, scenario, data = store.prepared_scenes()[0]
    data['version'] = 1
    data['selection'].pop('schema_version', None)
    data['selection']['mock_order'] = ['a']
    data['selection']['selection']['schema_version'] = 1
    data['selection']['selection']['discussion'] = True
    data['selection']['selection'].pop('goal', None)
    store.save_prepared_scene(owner, session, scenario, data)
    restored_backend = TeamAdapter()
    restored, _, _ = _service(tmp_path, [1700000000000], restored_backend)
    await restored.start()
    recovered = restored._team_conversations[(owner, session)]
    assert len(restored._transport_scopes) == 2
    await recovered.task
    assert recovered.state == 'closed'
    assert not restored._transport_scopes
    assert not hasattr(restored_backend, 'team')  # no old dialogue resumed
    assert not restored_backend.sent
    assert restored_backend.end_prepared_session.await_count == 1
    assert 'mock_order' not in store.prepared_scenes()[0][3]['selection']
    assert store.prepared_scenes()[0][3]['version'] == 2
    await restored.shutdown()
    backend.end_prepared_session.side_effect = None
    await service.shutdown()


async def test_distinct_devices_cannot_run_same_companion_twice(tmp_path):
    from eidolon.channel_provider.contracts import InvalidTransition
    backend = TeamAdapter()
    service, _, _ = _service(tmp_path, [1700000000000], backend)
    refs = []
    for name in ('input-1', 'output-1', 'input-2', 'output-2'):
        request = ProvisionRequest.parse(encoded(provision_payload(
            device_id=named_device_instance_id(name))))
        await service.provision(request)
        refs.append(request.device_ref)
    def opened(session, index):
        return OpenScene.model_validate(dict(type='open', owner_id='owner_1', selection=dict(
            scenario='ip_role_group', session_id=session, input_device=refs[index],
            members=[dict(companion_id='same-partner', output_device=refs[index + 1])])) )
    first, second = opened('first', 0), opened('second', 2)
    await service.open_team_conversation(first, authenticated_owner_id='owner_1')
    visit = service._team_conversations[('owner_1', 'first')]
    try:
        with pytest.raises(InvalidTransition, match='first'):
            await service.open_team_conversation(second, authenticated_owner_id='owner_1')
        await wait_for(lambda: visit.state == 'ready')
        with pytest.raises(InvalidTransition, match='first'):
            await service.open_team_conversation(second, authenticated_owner_id='owner_1')
        await service.device_conversation('first', authenticated_owner_id='owner_1', close=True, team=True)
        await wait_for(lambda: visit.closure_complete)
        backend.launched.clear()
        await service.open_team_conversation(second, authenticated_owner_id='owner_1')
        next_visit = service._team_conversations[('owner_1', 'second')]
        await wait_for(lambda: next_visit.state == 'ready')
        await service.device_conversation('second', authenticated_owner_id='owner_1', close=True, team=True)
        await wait_for(lambda: next_visit.closure_complete)
    finally:
        for current in service._team_conversations.values():
            if not current.closure_complete:
                current.stopped.set()
                await current.task


async def test_close_before_delayed_open_is_durable_and_owner_scoped(tmp_path):
    from eidolon.channel_provider.contracts import InvalidTransition, UnknownChannel
    backend = TeamAdapter()
    service, store, _ = _service(tmp_path, [1700000000000], backend)
    refs = []
    for name in ('cancel-input', 'cancel-output'):
        request = ProvisionRequest.parse(encoded(provision_payload(device_id=named_device_instance_id(name))))
        await service.provision(request)
        refs.append(request.device_ref)
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', selection=dict(
        scenario='ip_role_group', session_id='delayed', input_device=refs[0],
        members=[dict(companion_id='a', output_device=refs[1])])) )
    # Missing status alone is not closure evidence.
    with pytest.raises(UnknownChannel):
        await service.device_conversation('delayed', authenticated_owner_id='owner_1', team=True)
    closed = await service.device_conversation('delayed', authenticated_owner_id='owner_1', close=True, team=True)
    assert closed['state'] == 'closed'
    assert store.scene_cancelled('owner_1', 'delayed', 'ip_role_group')
    assert not store.scene_cancelled('other', 'delayed', 'ip_role_group')
    with pytest.raises(InvalidTransition, match='cancelled'):
        await service.open_team_conversation(opened, authenticated_owner_id='owner_1')
    restored, _, _ = _service(tmp_path, [1700000000000], TeamAdapter())
    assert (await restored.device_conversation('delayed', authenticated_owner_id='owner_1', team=True)) == closed
    with pytest.raises(InvalidTransition, match='cancelled'):
        await restored.open_team_conversation(opened, authenticated_owner_id='owner_1')
    assert not service._transport_scopes and not backend.sent


async def test_unknown_team_close_and_status_satisfy_hub_response_contract(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from eidolon.channel_provider.http import create_app
    from eidolon_sdk.biz.control.coordination import RoleGroupStatus
    service, _, _ = _service(tmp_path, [1700000000000], TeamAdapter())
    token = 'test-team-service-token-at-least-32-bytes'
    async with TestClient(TestServer(create_app(service=service, bearer_token=token))) as client:
        for action in ('close', 'status', 'close'):
            response = await client.post('/v1/role-groups/' + action,
                json={'owner_id': 'owner_1', 'session_id': 'refused-before-dispatch'},
                headers={'Authorization': 'Bearer ' + token})
            assert response.status == 200
            result = RoleGroupStatus.model_validate(await response.json())
            assert result.state == 'closed'
            assert result.error == ''
            assert result.session_id == 'refused-before-dispatch'


async def test_round_status_is_scoped_to_owned_worker_and_monotonic():
    import json
    from types import SimpleNamespace
    from livekit import rtc
    from eidolon_sdk.biz.contracts import SESSION_CONTROL_TOPIC
    from eidolon_sdk.biz.control.coordination import TeamRoundStatus
    from eidolon.channel_provider.team_conversation import TeamConversation
    from .test_livekit_adapter import _adapter
    adapter, _ = _adapter()
    visit = TeamConversation(SimpleNamespace(selection=SimpleNamespace(session_id='team')),
                             adapter, ({'device': 'input', 'room': 'room'},))
    visit.state = 'ready'
    remove = adapter.observe_team_state(visit.handles[0], 'native', 'team', visit._round_state)
    def send(session='team', conversation='native', epoch=1, error='', agent=True):
        status = TeamRoundStatus(session_id=session, conversation_id=conversation,
                                 epoch=epoch, error=error)
        packet = SimpleNamespace(topic=SESSION_CONTROL_TOPIC, data=status.model_dump_json().encode(),
            participant=SimpleNamespace(identity='agent-job' if agent else 'input',
                kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT if agent else 0))
        adapter._observe_team_state(packet, room='room')
    send(error='DECISION_TRANSPORT_ERROR', agent=False)
    send(session='other', error='DECISION_TRANSPORT_ERROR')
    send(conversation='old', error='DECISION_TRANSPORT_ERROR')
    assert visit.round_epoch == 0
    send(error='DECISION_TRANSPORT_ERROR')
    assert visit.snapshot()['round_error'] == 'DECISION_TRANSPORT_ERROR'
    send(epoch=2)
    send(epoch=1, error='DECISION_TRANSPORT_ERROR')
    assert visit.round_epoch == 2 and not visit.round_error
    remove()
    send(epoch=3, error='DECISION_TRANSPORT_ERROR')
    assert visit.round_epoch == 2
    await adapter.shutdown()
