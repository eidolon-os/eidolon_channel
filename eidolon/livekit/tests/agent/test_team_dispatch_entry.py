import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from eidolon_sdk.biz.presentation import InputSelection, OutputSelection, SessionOutputPlan
from eidolon.livekit.common.config.schema import EffectiveAgentConfig
from eidolon.livekit.common.team_dispatch import TeamDispatch
from eidolon.livekit.agent import server
from .test_team_worker import arguments


@pytest.mark.asyncio
async def test_production_dispatch_routes_only_explicit_team(monkeypatch):
    opened, kwargs = arguments()
    plan = SessionOutputPlan(session_id='native-input', policy_revision=1,
        inputs=InputSelection(microphone=True), outputs=OutputSelection())
    team = TeamDispatch(opened=opened, input_plan=plan,
        endpoints=tuple(o.endpoint for o in kwargs['outputs']))
    metadata = dict(conversation_id=plan.session_id, output_plan=plan.model_dump(mode='json'),
                    team_dispatch=team.model_dump(mode='json'))
    ctx = SimpleNamespace(room=SimpleNamespace(name='input'),
                          job=SimpleNamespace(metadata=json.dumps(metadata)))
    runner = AsyncMock()
    monkeypatch.setattr('eidolon.livekit.agent.coordination.entrypoint.run_team_dispatch', runner)
    await server.run_agent(ctx, EffectiveAgentConfig())
    runner.assert_awaited_once()
    metadata['target_companion_id'] = 'solo-target'
    ctx.job.metadata = json.dumps(metadata)
    with pytest.raises(ValueError, match='SCOPE_MISMATCH'):
        await server.run_agent(ctx, EffectiveAgentConfig())
    assert runner.await_count == 1


async def test_provider_owned_team_teardown_never_emits_competing_session_end(monkeypatch):
    from eidolon.livekit.agent.coordination import entrypoint
    from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint
    opened, kwargs = arguments()
    plan = SessionOutputPlan(session_id='native-input', policy_revision=1,
        inputs=InputSelection(microphone=True), outputs=OutputSelection())
    team = TeamDispatch(opened=opened, input_plan=plan,
        endpoints=tuple(o.endpoint for o in kwargs['outputs']))
    rooms = []
    def room():
        value = SimpleNamespace(local_participant=SimpleNamespace(publish_data=AsyncMock()),
            name='test', disconnect=AsyncMock())
        rooms.append(value)
        return value
    async def connect(self, **kwargs):
        return room()
    monkeypatch.setenv('EIDOLON_AGENT_ADMIN_API_TOKEN', 'test-credential')
    monkeypatch.setattr(PresentationEndpoint, 'connect', connect)
    monkeypatch.setattr(server, '_resolve_session_metadata', AsyncMock(return_value=('ptt', None)))
    monkeypatch.setattr(entrypoint.SharedStageFactory, 'from_config', lambda *a, **kw:
        SimpleNamespace(tts=SimpleNamespace(tts=object()), aclose=AsyncMock()))
    class Worker:
        cleanup_ok = True
        def __init__(self, *args, on_ready, **kwargs):
            self.on_ready = on_ready
        async def run(self, *args, **kwargs):
            await self.on_ready()
    monkeypatch.setattr(entrypoint, 'TeamWorker', Worker)
    ctx = SimpleNamespace(room=room(), proc=SimpleNamespace(userdata={}))
    await entrypoint.run_team_dispatch(ctx, EffectiveAgentConfig(), team.model_dump(mode='json'))
    from eidolon_sdk.biz.contracts import SESSION_STARTED_TYPE
    for value in rooms:
        calls = value.local_participant.publish_data.call_args_list
        assert len(calls) == 1
        assert json.loads(calls[0].args[0])['type'] == SESSION_STARTED_TYPE
