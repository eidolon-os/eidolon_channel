import asyncio
import json
from types import SimpleNamespace
import pytest
from livekit import rtc
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon.livekit.agent.session.playback_stop import stop_playback


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
