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
