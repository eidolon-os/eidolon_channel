import asyncio
import json
from types import SimpleNamespace
import pytest
from livekit import rtc
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon.livekit.agent.session.playback_stop import stop_playback
from eidolon.livekit.agent.session.playback_stop import PlaybackStopError


@pytest.mark.asyncio
async def test_stop_uses_authenticated_device_terminal_ack():
    room = rtc.EventEmitter()
    sent = asyncio.Queue()
    async def publish(data, **kwargs):
        assert kwargs['destination_identities'] == ['device']
        await sent.put(json.loads(data))
    room.local_participant = SimpleNamespace(publish_data=publish)
    task = asyncio.create_task(stop_playback(room, 'device', 'native', 1))
    command = await sent.get()
    def ack(identity, status):
        room.emit('data_received', SimpleNamespace(topic=CONTROL_TOPIC,
            participant=SimpleNamespace(identity=identity), data=json.dumps(dict(v=1,
                kind='ack', device_id='device', ref=command['id'], op='playback.stop',
                status=status)).encode()))
    ack('other-device', 'completed')
    ack('device', 'accepted')
    await asyncio.sleep(0)
    assert not task.done()
    ack('device', 'completed')
    assert await task


@pytest.mark.parametrize('sdk_stopped,code,tolerated', [
    (True, 'UNSUPPORTED_OPERATION', True),
    (False, 'UNSUPPORTED_OPERATION', False),
    (True, 'PLAYBACK_FAILED', False),
])
async def test_unsupported_requires_already_stopped_sdk_output(sdk_stopped, code, tolerated):
    room = rtc.EventEmitter()
    async def publish(data, **kwargs):
        command = json.loads(data)
        room.emit('data_received', SimpleNamespace(topic=CONTROL_TOPIC,
            participant=SimpleNamespace(identity='device'), data=json.dumps(dict(v=1,
                kind='ack', device_id='device', ref=command['id'], op='playback.stop',
                status='error', code=code)).encode()))
    room.local_participant = SimpleNamespace(publish_data=publish)
    if tolerated:
        assert await stop_playback(room, 'device', 'session', 1,
                                   sdk_output_stopped=sdk_stopped) is False
    else:
        with pytest.raises(PlaybackStopError, match='TEAM_STOP_REJECTED'):
            await stop_playback(room, 'device', 'session', 1, sdk_output_stopped=sdk_stopped)
