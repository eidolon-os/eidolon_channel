"""Home transport forwards transcripts and lifecycle without interpreting them."""
import json

import httpx
import pytest

from eidolon_sdk.biz.smarthome import HomeSessionScope

from eidolon.livekit.agent import smarthome


@pytest.mark.asyncio
async def test_transcript_keeps_session_scope_and_forwards_clarification(monkeypatch):
    requests = []
    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith('/command'):
            return httpx.Response(200, json={
                'turn_id': body['turn_id'], 'utterance': body['utterance'],
                'outcome': 'clarification', 'message': '要调节哪个设备？',
            })
        return httpx.Response(200, json={})
    original = httpx.AsyncClient
    monkeypatch.setenv('EIDOLON_AGENT_ADMIN_API_TOKEN', 'test-token')
    monkeypatch.setenv('EIDOLON_CHANNEL_PROVIDER_TOKEN', 'test-token')
    monkeypatch.setattr(smarthome.httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(serve), **kw))
    scope = HomeSessionScope(owner_id='o', companion_id='c', device_ref='d', session_id='voice-1')
    await smarthome.handle_transcript(scope, '调一下')
    await smarthome.end_home_session(scope)
    assert requests[0][1]['session_id'] == 'voice-1'
    assert requests[0][1]['companion_id'] == 'c'
    assert requests[1][1]['result']['outcome'] == 'clarification'
    assert requests[2] == ('/api/admin/smarthome/session/end', {
        'owner_id': 'o', 'companion_id': 'c', 'device_ref': 'd', 'session_id': 'voice-1',
    })


@pytest.mark.asyncio
async def test_turn_stage_logs_correlate_agent_and_delivery(monkeypatch,caplog):
    monkeypatch.setenv('EIDOLON_AGENT_ADMIN_API_TOKEN','test-token')
    monkeypatch.setenv('EIDOLON_CHANNEL_PROVIDER_TOKEN','test-token')
    def serve(request):
        body=json.loads(request.content)
        if request.url.path.endswith('/command'):
            return httpx.Response(200,json={'turn_id':body['turn_id'],'utterance':body['utterance'],
                                           'outcome':'answered','message':'完成'})
        return httpx.Response(200,json={})
    caplog.set_level('INFO',logger=smarthome.logger.name)
    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        await smarthome.handle_transcript(HomeSessionScope(owner_id='o',companion_id='c',device_ref='d',session_id='s'),'开灯',client=client)
    assert all(x in caplog.text for x in ['stage=handler_started','stage=agent','stage=result_delivery','delivered elapsed_ms='])
