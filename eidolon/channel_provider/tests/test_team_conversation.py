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


async def test_three_device_team_http_start_status_close_and_shared_occupancy(tmp_path):
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
        response = await client.post('/v1/role-groups/close', json=query, headers=headers)
        assert (await response.json())['state'] == 'closed'
        assert not service._transport_scopes
        assert backend.end_prepared_session.await_count == 3
