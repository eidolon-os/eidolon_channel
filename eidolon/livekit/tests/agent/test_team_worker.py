import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from eidolon_sdk.biz.presentation import InputSelection, OutputSelection, SessionOutputPlan
from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint
from eidolon.livekit.common.config.schema import ObservabilityConfig, TurnPolicyConfig
from eidolon.livekit.agent.coordination import worker
from .test_role_group_client import opening


def arguments():
    opened = opening()
    def room(identity):
        return SimpleNamespace(name=identity, isconnected=lambda: True,
            local_participant=SimpleNamespace(publish_data=AsyncMock()),
            remote_participants={identity: SimpleNamespace(identity=identity)})
    outputs = tuple(worker.TeamOutput(member.companion_id, PresentationEndpoint(
        room=member.output_device.device_instance_id,
        participant_identity=member.output_device.device_instance_id,
        plan=SessionOutputPlan(session_id='native-' + member.companion_id,
            policy_revision=1, inputs=InputSelection(microphone=False),
            outputs=OutputSelection(speech=True))), room(member.output_device.device_instance_id),
        object(), AsyncMock(return_value=True)) for member in opened.selection.members)
    return opened, dict(input_room=room(opened.selection.input_device.device_instance_id),
        input_factory=SimpleNamespace(outputs=OutputSelection(), stt=object()),
        outputs=outputs, stop=AsyncMock(return_value=True), on_ready=AsyncMock())


@pytest.fixture
def runtime(monkeypatch):
    sessions, pipelines, clients = [], [], []
    launched, finish = asyncio.Event(), asyncio.Event()
    class Session:
        def __init__(self, **kwargs):
            self.handlers = {}
            self.aclose = AsyncMock()
            self.start = AsyncMock()
            sessions.append(self)
        def on(self, event, callback):
            self.handlers[event] = callback
    class Client:
        def __init__(self, opened, **kwargs):
            self.ready, self.closed = asyncio.Event(), asyncio.Event()
            self.cleanup_ok = False
            self.kwargs = kwargs
            clients.append(self)
        async def run(self, *args, **kwargs):
            self.ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleanup_ok = True
                self.closed.set()
    class Pipeline:
        def __init__(self, factory, **kwargs):
            self.factory, self.kwargs = factory, kwargs
            self.shutdown = AsyncMock()
            pipelines.append(self)
        async def run(self, room):
            assert self.kwargs['destination'].ready.is_set()
            await self.kwargs['on_session_started']()
            launched.set()
            await finish.wait()
    monkeypatch.setattr(worker, 'AgentSession', Session)
    monkeypatch.setattr(worker, 'RoleGroupClient', Client)
    monkeypatch.setattr(worker, 'HalfDuplexPttPipeline', Pipeline)
    # TTS objects here are sentinels; native AgentSession/TTS are covered by the
    # native output integration tests, this test checks scene composition.
    monkeypatch.setattr(worker, 'PolicyBoundAgent', lambda **kwargs: kwargs)
    return sessions, pipelines, clients, launched, finish


@pytest.mark.asyncio
async def test_team_composes_one_input_and_exact_native_outputs(runtime):
    sessions, pipelines, clients, launched, finish = runtime
    opened, kwargs = arguments()
    original = TurnPolicyConfig()
    policy = replace(original, ptt=replace(original.ptt, segment_min_audio_ms=120))
    observations = ObservabilityConfig()
    team = worker.TeamWorker(opened, **kwargs, turn_policy=policy,
        observability=observations, audio_sample_rate=24_000)
    task = asyncio.create_task(team.run(None, agent_url='http://agent', service_token='test'))
    try:
        await asyncio.wait_for(launched.wait(), 2)
        assert len(sessions) == 2 and len(pipelines) == 1
        assert pipelines[0].factory is kwargs['input_factory']
        input_options = pipelines[0].kwargs
        assert input_options['turn_policy'].idle.disconnect_after_idle_ms == 0
        assert policy.idle.disconnect_after_idle_ms == 60_000
        assert input_options['turn_policy'].ptt.segment_min_audio_ms == 120
        assert input_options['observability'] is observations
        assert input_options['audio_sample_rate'] == 24_000
        for session, output in zip(sessions, kwargs['outputs']):
            call = session.start.await_args
            assert call.args[0]['llm'] is None and call.args[0]['stt'] is None
            assert call.args[0]['tts'] is output.tts
            opts = call.kwargs['room_options']
            assert opts.audio_input is False and opts.text_input is False
            assert opts.audio_output is True
            assert call.kwargs['room'] is output.room
        kwargs['on_ready'].assert_awaited_once()
        assert callable(clients[0].kwargs['present'])
        finish.set()
        await task
        assert team.cleanup_ok
        assert all(s.aclose.await_count == 1 for s in sessions)
        with pytest.raises(RuntimeError, match='reused'):
            await team.run(None, agent_url='http://agent', service_token='test')
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_output_loss_stops_team_input_and_cleans_all_sessions(runtime):
    sessions, pipelines, clients, launched, finish = runtime
    opened, kwargs = arguments()
    team = worker.TeamWorker(opened, **kwargs)
    task = asyncio.create_task(team.run(None, agent_url='http://agent', service_token='test'))
    await asyncio.wait_for(launched.wait(), 2)
    sessions[0].handlers['close'](None)
    await asyncio.wait_for(task, 2)
    assert clients[0].closed.is_set()
    pipelines[0].shutdown.assert_awaited_once()
    assert all(s.aclose.await_count == 1 for s in sessions)


def test_missing_member_or_speaking_input_rejected_before_any_start():
    opened, kwargs = arguments()
    with pytest.raises(ValueError, match='membership'):
        worker.TeamWorker(opened, **(kwargs | {'outputs': kwargs['outputs'][:1]}))
    kwargs['input_factory'].outputs = OutputSelection(speech=True)
    with pytest.raises(ValueError, match='must not respond'):
        worker.TeamWorker(opened, **kwargs)


async def test_team_maps_real_playback_and_round_completion_to_device_ui(runtime, monkeypatch):
    sessions, pipelines, clients, launched, finish = runtime
    async def speech(start, text, speaking):
        speaking()
        return True
    monkeypatch.setattr(worker, 'NativeSpeechPresenter', lambda endpoints: speech)
    opened, kwargs = arguments()
    team = worker.TeamWorker(opened, **kwargs)
    task = asyncio.create_task(team.run(None, agent_url='http://agent', service_token='test'))
    try:
        await asyncio.wait_for(launched.wait(), 2)
        device = kwargs['outputs'][0].endpoint.participant_identity
        assert await clients[0].kwargs['present'](SimpleNamespace(device_id=device), None, lambda: None)
        def states(room):
            return [json.loads(c.args[0])['state']
                    for c in room.local_participant.publish_data.await_args_list]
        assert states(kwargs['input_room']) == ['thinking', 'speaking']
        assert states(kwargs['outputs'][0].room) == ['thinking', 'speaking', 'listening']
        assert states(kwargs['outputs'][1].room) == []
        await clients[0].kwargs['on_state'](SimpleNamespace(state='idle'))
        assert states(kwargs['input_room'])[-1] == 'listening'
    finally:
        finish.set()
        await task
