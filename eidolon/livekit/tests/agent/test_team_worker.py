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
            self.closing = asyncio.Event()
            self.kwargs = kwargs
            clients.append(self)
        async def run(self, *args, **kwargs):
            self.ready.set()
            try:
                await self.closing.wait()
            finally:
                self.cleanup_ok = True
                self.closed.set()
        def close(self):
            self.closing.set()
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
        await clients[0].kwargs['on_state'](SimpleNamespace(
            outcome='finished', error_code='', capture_id='one'))
        assert states(kwargs['input_room'])[-1] == 'waiting'
    finally:
        finish.set()
        await task


@pytest.mark.parametrize('trigger', ['input_end', 'speaking_end', 'worker_cancel', 'output_loss', 'peer_loss'])
@pytest.mark.parametrize('stop_ok', [True, False])
async def test_worker_drains_control_before_releasing_outputs(runtime, monkeypatch, trigger, stop_ok):
    """Actual worker/client over TCP; only peer and media are test adapters."""
    import aiohttp
    from aiohttp import web
    from eidolon_sdk.biz.control.coordination_stream import ROLE_GROUP_STREAM_PATH
    from eidolon.livekit.agent.coordination.client import RoleGroupClient
    from .test_role_group_client import frame
    sessions, pipelines, clients, launched, finish = runtime
    monkeypatch.setattr(worker, 'RoleGroupClient', RoleGroupClient)
    opened, kwargs = arguments()
    entered, release, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()
    speaking, revoked = asyncio.Event(), asyncio.Event()
    receipts, calls = [], []
    async def present(start, text, on_speaking):
        on_speaking()
        speaking.set()
        try:
            await asyncio.Event().wait()
        finally:
            revoked.set()
    monkeypatch.setattr(worker, 'NativeSpeechPresenter', lambda endpoints: present)
    async def stop(device):
        calls.append(device)
        if trigger == 'speaking_end' and not team.client._close_requested:
            return True  # Initial PTT barrier, before closing during speech.
        entered.set()
        await release.wait()
        assert not any(s.aclose.called for s in sessions)
        return stop_ok
    kwargs['stop'] = stop
    async def peer(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()
        await ws.send_str(frame('prepared', policy='semantic-step-v2', physical_devices_ready=False))
        if trigger == 'speaking_end':
            assert (await ws.receive_json())['type'] == 'press'
            await ws.send_str(frame('capturing', capture_id='one', epoch=1))
            assert (await ws.receive_json())['type'] == 'release'
            member = opened.selection.members[0]
            await ws.send_str(frame('reply_start', request_id='reply', turn_id='reply',
                device_id=member.output_device.device_instance_id,
                companion_id=member.companion_id, epoch=1))
            assert (await ws.receive_json())['type'] == 'speaking'
        if trigger == 'peer_loss':
            await disconnect.wait()
            await ws.close(code=1011)
            return ws
        request = await ws.receive_json()
        assert request['type'] == 'close'
        for m in opened.selection.members:
            await ws.send_str(frame('stop', request_id=m.companion_id,
                device_id=m.output_device.device_instance_id, epoch=1))
        for _ in opened.selection.members:
            receipts.append(await ws.receive_json())
        await ws.close()
        return ws
    app = web.Application()
    app.router.add_get(ROLE_GROUP_STREAM_PATH, peer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    team = worker.TeamWorker(opened, **kwargs)
    try:
        async with aiohttp.ClientSession() as http:
            task = asyncio.create_task(team.run(http, agent_url=f'http://127.0.0.1:{port}', service_token='test'))
            try:
                await asyncio.wait_for(launched.wait(), 2)
                if trigger == 'speaking_end':
                    team.client.press('one')
                    team.client.release('one')
                    await asyncio.wait_for(speaking.wait(), 2)
                if trigger == 'worker_cancel':
                    task.cancel()
                elif trigger == 'output_loss':
                    sessions[0].handlers['close'](None)
                elif trigger == 'peer_loss':
                    disconnect.set()
                else:
                    finish.set()
                await asyncio.wait_for(entered.wait(), 2)
                assert not task.done()
                assert not any(s.aclose.called for s in sessions)
                with pytest.raises(ConnectionError):
                    team.client.press('late-input')
                # A repeated end is idempotent while receipts are in flight.
                if trigger != 'peer_loss':
                    team.client.close()
                release.set()
                result = (await asyncio.gather(task, return_exceptions=True))[0]
                if trigger == 'worker_cancel':
                    assert isinstance(result, asyncio.CancelledError)
                elif trigger == 'peer_loss':
                    assert isinstance(result, ConnectionError)
                else:
                    assert result is None
                assert team.cleanup_ok is stop_ok
                assert set(calls) == set(team.client.members)
                assert all(s.aclose.await_count == 1 for s in sessions)
                if trigger == 'speaking_end':
                    assert revoked.is_set()
                if trigger != 'peer_loss':
                    assert len(receipts) == 2
                    assert all(r['result'] == ('completed' if stop_ok else 'failed') for r in receipts)
                else:
                    assert not receipts  # Physical fallback is not a peer receipt.
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        await runner.cleanup()
