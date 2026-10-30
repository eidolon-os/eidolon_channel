"""Read-only inventory and saved-Team diagnostics through Owner Management.

Runtime commands require EIDOLON_CONTROLLER_ACCESS_TOKEN and --url pointing to
that Controller's Host. They use the same saved revision and admission path as
Mobile. Provider service credentials are used only for read-only inventory.
"""
import argparse
import asyncio
import json
import os
from urllib.parse import quote
import aiohttp
from livekit import api
from eidolon_sdk.biz.control.coordination import RoleGroupStatus
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
    if args.action == 'inventory':
        return {'devices': await inventory(load_provider_config())}
    token = os.environ.get('EIDOLON_CONTROLLER_ACCESS_TOKEN', '').strip()
    if not token:
        raise ValueError('EIDOLON_CONTROLLER_ACCESS_TOKEN is required')
    team = quote(args.team, safe='')
    session_id = args.session
    path = f'/api/management/v1/teams/{team}/runs'
    body = None
    method = 'POST'
    if args.action == 'start':
        body = dict(expected_revision=args.revision, session_id=session_id)
    else:
        path += '/' + quote(session_id, safe='')
        if args.action == 'close':
            path += '/close'
        else:
            method = 'GET'
    async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + token},
                                    timeout=aiohttp.ClientTimeout(total=60)) as http:
        async with http.request(method, args.url.rstrip('/') + path, json=body,
                                allow_redirects=False) as response:
            if response.status != 200:
                raise RuntimeError(f'saved Team API refused: HTTP {response.status}; '
                                   f'session={session_id}; query this identity before retrying')
            result = RoleGroupStatus.model_validate(await response.json())
            if result.session_id != session_id or result.scenario != 'ip_role_group':
                raise ValueError('Team API returned a different run')
            return result.model_dump(mode='json')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='action', required=True)
    commands.add_parser('inventory')
    for name in ('start', 'status', 'close'):
        command = commands.add_parser(name)
        command.add_argument('--url', required=True, help='Authenticated Host Management URL')
        command.add_argument('--team', required=True, help='Saved Team ID')
        command.add_argument('--session', required=True, help='Stable run ID, also used after an uncertain response')
        if name == 'start':
            command.add_argument('--revision', required=True, type=int)
    return result


def main():
    print(json.dumps(asyncio.run(run(parser().parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
