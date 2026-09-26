"""Acknowledged stop on an existing room, using the shared control protocol."""
import asyncio
import json
from uuid import uuid4
from eidolon_sdk.biz.contracts import CONTROL_TOPIC
from eidolon_sdk.biz.control.protocol import command_status_from_ack
from eidolon.livekit.control_receipts import read_control_receipt
from .client_control import build_session_client_control_envelope


async def stop_playback(room, device: str, session_id: str, policy_revision: int) -> bool:
    command = build_session_client_control_envelope(op='playback.stop', reason='scene_interrupt',
        payload={'session_id': session_id, 'policy_revision': policy_revision})
    command['id'] = 'stop:' + uuid4().hex
    done = asyncio.get_running_loop().create_future()

    def receipt(packet):
        body = read_control_receipt(packet, device=device, max_bytes=4096)
        if (body is None or body['ref'] != command['id']
                or body.get('op') != 'playback.stop' or done.done()):
            return
        status = command_status_from_ack(body['status'])
        if status == 'succeeded':
            done.set_result(True)
        elif status not in ('accepted', 'running'):
            done.set_result(False)

    room.on('data_received', receipt)
    try:
        async with asyncio.timeout(2):
            await room.local_participant.publish_data(json.dumps(command).encode(),
                reliable=True, topic=CONTROL_TOPIC, destination_identities=[device])
            return await done
    except (TimeoutError, ConnectionError):
        return False
    finally:
        room.off('data_received', receipt)
        if not done.done():
            done.cancel()
