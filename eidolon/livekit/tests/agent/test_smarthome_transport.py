"""Home transport forwards transcripts and lifecycle without interpreting them."""
import json

import httpx
import pytest

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
    await smarthome.handle_transcript('o', 'd', '调一下', session_id='voice-1')
    await smarthome.end_home_session('o', 'd', 'voice-1')
    assert requests[0][1]['session_id'] == 'voice-1'
    assert requests[1][1]['result']['outcome'] == 'clarification'
    assert requests[2] == ('/api/admin/smarthome/session/end', {
        'owner_id': 'o', 'device_ref': 'd', 'session_id': 'voice-1',
    })
