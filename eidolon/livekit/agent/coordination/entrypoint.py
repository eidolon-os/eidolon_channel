"""Production team dispatch composition; other scene entrypoints do not call it."""
import asyncio
import os
import aiohttp
from eidolon_sdk.biz.contracts import SESSION_CONTROL_TOPIC
from eidolon.livekit.common.team_dispatch import TeamDispatch
from ..factory import SharedStageFactory
from ..session.playback_stop import stop_playback
from .worker import TeamWorker, TeamOutput


async def run_team_dispatch(ctx, cfg, raw):
    from ..server import _session_lifecycle_payload
    from eidolon_sdk.biz.contracts import SESSION_STARTED_TYPE, SESSION_END_TYPE
    team = TeamDispatch.model_validate(raw)
    url = os.environ.get('EIDOLON_TEAM_AGENT_URL', 'http://127.0.0.1:8081')
    token = os.environ.get('EIDOLON_AGENT_ADMIN_API_TOKEN', '')
    if not token:
        raise ValueError('team Agent service credential is not configured')
    from ..server import _resolve_session_metadata
    mode, _ = await _resolve_session_metadata(ctx)
    if mode != 'ptt':
        raise ValueError('team input device must explicitly support PTT')
    rooms, factories, outputs = [], [], []
    worker = None
    source = team.opened.selection.input_device.device_instance_id
    try:
        for member, endpoint in zip(team.opened.selection.members, team.endpoints):
            async with asyncio.timeout(10):
                room = await endpoint.connect(core=cfg.core,
                    worker_identity=f'presentation-team-{team.opened.selection.session_id}-{member.companion_id}')
            rooms.append(room)
            factory = SharedStageFactory.from_config(cfg, livekit_room=room,
                livekit_session_key=room.name, runtime_session_id=endpoint.plan.session_id,
                output_plan=endpoint.plan, target_companion_id=member.companion_id)
            factories.append(factory)
            outputs.append(TeamOutput(member.companion_id, endpoint, room, factory.tts.tts))
        source_factory = SharedStageFactory.from_config(cfg, livekit_room=ctx.room,
            livekit_session_key=ctx.room.name, runtime_session_id=team.input_plan.session_id,
            output_plan=team.input_plan, prebuilt_vad=getattr(ctx.proc, 'userdata', {}).get('vad'))
        factories.append(source_factory)
        by_device = {output.endpoint.participant_identity: output for output in outputs}

        async def stop(device):
            output = by_device[device]
            plan = output.endpoint.plan
            return await stop_playback(output.room, device, plan.session_id, plan.policy_revision)

        async def lifecycle(kind):
            await asyncio.gather(*(
                output.room.local_participant.publish_data(_session_lifecycle_payload(kind,
                    output.endpoint.plan.session_id, output_plan=(
                        output.endpoint.plan if kind == SESSION_STARTED_TYPE else None),
                    reason='user_left' if kind == SESSION_END_TYPE else None),
                    reliable=True, topic=SESSION_CONTROL_TOPIC,
                    destination_identities=[output.endpoint.participant_identity]) for output in outputs))
            await ctx.room.local_participant.publish_data(_session_lifecycle_payload(kind,
                team.input_plan.session_id, output_plan=(
                    team.input_plan if kind == SESSION_STARTED_TYPE else None),
                reason='user_left' if kind == SESSION_END_TYPE else None),
                reliable=True, topic=SESSION_CONTROL_TOPIC, destination_identities=[source])

        worker = TeamWorker(team.opened, input_room=ctx.room, input_factory=source_factory,
            outputs=tuple(outputs), stop=stop, on_ready=lambda: lifecycle(SESSION_STARTED_TYPE))
        async with aiohttp.ClientSession() as http:
            try:
                await worker.run(http, agent_url=url, service_token=token)
            finally:
                if worker.cleanup_ok:
                    await lifecycle(SESSION_END_TYPE)
        if not worker.cleanup_ok:
            raise RuntimeError('team device cleanup did not complete')
    finally:
        await asyncio.gather(*(f.aclose() for f in factories), return_exceptions=True)
        await asyncio.gather(*(room.disconnect() for room in rooms), return_exceptions=True)
