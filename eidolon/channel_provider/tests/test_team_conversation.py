import pytest

from eidolon_sdk.biz.control.coordination_stream import OpenScene
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id
from eidolon.channel_provider.contracts import ProvisionRequest
from .helpers import provision_payload, encoded
from .test_device_conversation import Adapter, wait_for
from .test_service import _service


class TeamAdapter(Adapter):
    async def open_team_session(self, opened, handles, sessions):
        self.team = (opened, handles, sessions)
        self.launched.set()


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
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', mock_order=['a', 'b'],
        selection=dict(scenario='ip_role_group', session_id='team', input_device=refs[0],
            members=[dict(companion_id=key, output_device=ref) for key, ref in zip(('a', 'b'), refs[1:])],
            discussion=True, reply_budget=4)))
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
            backend.end_prepared_session.side_effect = [RuntimeError('input ACK timeout'), None, None]
            response = await client.post('/v1/role-groups/close', json=query, headers=headers)
            assert response.status == 503, await response.text()
            assert len(service._transport_scopes) == 3
            response = await client.post('/v1/role-groups/status', json=query, headers=headers)
            status = await response.json()
            assert status['state'] == 'failed'
            assert refs[0].device_instance_id in status['error']
            backend.end_prepared_session.side_effect = None
        response = await client.post('/v1/role-groups/close', json=query, headers=headers)
        assert response.status == 200, await response.text()
        status = await response.json()
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
    opened = OpenScene.model_validate(dict(type='open', owner_id='owner_1', mock_order=['b', 'a'],
        selection=dict(scenario='ip_role_group', session_id='demo', input_device=refs[0], members=[
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
    assert team.opened.mock_order == ('b', 'a')
    assert 'presentation_endpoint' not in metadata and 'target_companion_id' not in metadata
    await adapter.open_team_session(opened, handles, sessions)
    assert len(client.agent_dispatch.created) == 1  # Idempotent dispatch.
    await adapter.shutdown()
