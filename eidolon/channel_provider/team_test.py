"""Operator test client for the production team API, using normal service config.

Run with the installed Channel environment. Inventory is read-only; start/close
use Provider APIs. No credentials, room tokens or opaque handles are printed.
"""
import argparse
import asyncio
import json
from uuid import uuid4
import aiohttp
from livekit import api
from eidolon_sdk.biz.control.coordination_stream import OpenScene
from .config import load_provider_config
from .store import ChannelProviderStore
from eidolon.livekit.common.config import load_agent_config
from eidolon.livekit.agent.runtime.kernel_bodies import KernelBodyHttpClient


async def inventory(config):
    settings = load_agent_config()
    mounts = KernelBodyHttpClient(base_url=settings.runtime_authority.kernel_api_url)
    client = api.LiveKitAPI(config.livekit.api_url, api_key=config.livekit.api_key,
                           api_secret=config.livekit.api_secret)
    rows = []
    try:
        for record in ChannelProviderStore(config.storage.path).observable_provisions():
            room = json.loads(record.handle_json)['room']
            participants = await client.room.list_participants(api.ListParticipantsRequest(room=room))
            participant = next((p for p in participants.participants if p.identity == record.device_id), None)
            if participant is None:
                continue
            metadata = json.loads(participant.metadata or '{}')
            connection = await mounts.resolve(owner_id=record.owner_id, device_id=record.device_id)
            rows.append(dict(device_id=record.device_id, owner_id=record.owner_id,
                device_ref=record.device_ref.model_dump(mode='json'),
                manifest_id=metadata.get('manifest_id'), interaction_mode=metadata.get('interaction_mode'),
                companion_id=connection.answering_companion_id))
        return rows
    finally:
        await client.aclose()
        await mounts.close()


async def run(args):
    config = load_provider_config()
    if args.action == 'inventory':
        return {'devices': await inventory(config)}
    if args.action == 'start':
        rows = {row['device_id']: row for row in await inventory(config)}
        source = rows[args.input]
        selected = [rows[device] for device in args.output]
        if source['interaction_mode'] != 'ptt':
            raise ValueError('input is not a PTT device')
        if any(row['owner_id'] != source['owner_id'] or not row['companion_id'] for row in selected):
            raise ValueError('outputs require same-Owner Companion bindings')
        opened = OpenScene.model_validate(dict(type='open', owner_id=source['owner_id'],
            mock_order=[row['companion_id'] for row in selected], selection=dict(
                scenario='ip_role_group', session_id=args.session or 'team-' + uuid4().hex,
                input_device=source['device_ref'], discussion=args.discussion,
                reply_budget=args.budget, members=[dict(companion_id=row['companion_id'],
                    output_device=row['device_ref']) for row in selected])))
        body = opened.model_dump(mode='json')
        action = 'open'
    else:
        body = dict(owner_id=args.owner, session_id=args.session)
        action = args.action
    async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + config.bearer_token}) as http:
        async with http.post(f'http://127.0.0.1:{config.http.port}/v1/role-groups/{action}', json=body) as response:
            result = await response.json()
            if response.status != 200:
                raise RuntimeError(f'team API rejected request: HTTP {response.status}')
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('inventory')
    start = commands.add_parser('start')
    start.add_argument('--input', required=True)
    start.add_argument('--output', action='append', required=True)
    start.add_argument('--discussion', action='store_true')
    start.add_argument('--budget', type=int, default=4)
    start.add_argument('--session')
    for name in ('status', 'close'):
        command = commands.add_parser(name)
        command.add_argument('--owner', required=True)
        command.add_argument('--session', required=True)
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
