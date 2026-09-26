"""Compose the team scene from existing PTT, Companion stream and native RoomIO.

The caller supplies Provider-authorized, already-connected standing rooms and
configured stages. This module neither reserves devices nor builds transports.
Only the explicit ip_role_group selection can instantiate this composition.
"""
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from livekit.agents.voice import AgentSession
from livekit.agents.voice.room_io import RoomOptions
from eidolon_sdk.biz.control.coordination_stream import OpenScene, ReplyStart
from eidolon.livekit.common.presentation_endpoint import PresentationEndpoint
from ..half_duplex.pipeline import HalfDuplexPttPipeline
from ..session.policy_bound_agent import PolicyBoundAgent
from .client import RoleGroupClient
from .native_output import NativeSpeechPresenter, SpeechEndpoint


@dataclass(frozen=True)
class TeamOutput:
    companion_id: str
    endpoint: PresentationEndpoint
    room: object
    tts: object  # Existing configured LiveKit TTS plugin, not a new provider.
    confirm_playback: Callable[[ReplyStart], Awaitable[bool]] | None = None


class TeamWorker:
    def __init__(self, opened: OpenScene, *, input_room, input_factory,
                 outputs: tuple[TeamOutput, ...], stop: Callable[[str], Awaitable[bool]],
                 on_ready: Callable[[], Awaitable[None]]):
        expected = {(m.companion_id, m.output_device.device_instance_id)
                    for m in opened.selection.members}
        actual = {(o.companion_id, o.endpoint.participant_identity) for o in outputs}
        if actual != expected or len(outputs) != len(expected):
            raise ValueError('team outputs must exactly match authorized membership')
        if input_factory.outputs.can_respond or input_factory.outputs.audio_cue:
            raise ValueError('team input must not respond')
        if input_factory.stt is None:
            raise ValueError('team requires its existing ASR stage')
        if not input_room.isconnected() or opened.selection.input_device.device_instance_id not in {
            p.identity for p in input_room.remote_participants.values()
        }:
            raise ValueError('team input device is not present')
        for output in outputs:
            if (output.room.name != output.endpoint.room or not output.room.isconnected()
                    or not output.endpoint.plan.outputs.speech or output.tts is None
                    or output.endpoint.participant_identity not in {
                        p.identity for p in output.room.remote_participants.values()}):
                raise ValueError('team output is not ready for native speech')
        self.opened = opened
        self.input_room, self.input_factory = input_room, input_factory
        self.outputs, self.stop, self.on_ready = outputs, stop, on_ready
        self.client = None
        self._started = False
        self.cleanup_ok = False

    async def run(self, http, *, agent_url: str, service_token: str):
        if self._started:
            raise RuntimeError('team worker cannot be reused')
        self._started = True
        sessions = []
        tasks = []
        pipeline = None
        ended = asyncio.Event()
        try:
            endpoints = {}
            async with asyncio.timeout(10):
                for output in self.outputs:
                    session = AgentSession(turn_handling={
                        'turn_detection': 'manual', 'interruption': {'enabled': False}})
                    sessions.append(session)
                    session.on('close', lambda event: ended.set())
                    await session.start(PolicyBoundAgent(instructions='',
                        outputs=output.endpoint.plan.outputs, llm=None, stt=None, tts=output.tts),
                        room=output.room, room_options=RoomOptions(
                            participant_identity=output.endpoint.participant_identity,
                            audio_input=False, video_input=False, text_input=False,
                            audio_output=True, text_output=output.endpoint.plan.outputs.dialogue_text,
                            close_on_disconnect=True))
                    endpoints[(output.companion_id, output.endpoint.participant_identity)] = (
                        SpeechEndpoint(session, output.confirm_playback))
            self.client = RoleGroupClient(self.opened,
                present=NativeSpeechPresenter(endpoints), stop=self.stop)
            transport = asyncio.create_task(self.client.run(
                http, base_url=agent_url, token=service_token))
            tasks.append(transport)
            ready = asyncio.create_task(self.client.ready.wait())
            lost_output = asyncio.create_task(ended.wait())
            tasks.extend((ready, lost_output))
            async with asyncio.timeout(10):
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if transport in done:
                    await transport
                    raise ConnectionError('team transport ended during preparation')
                if lost_output in done:
                    raise ConnectionError('team output ended during preparation')
            pipeline = HalfDuplexPttPipeline(self.input_factory, destination=self.client,
                welcome_message=None, on_session_started=self.on_ready)
            recording = asyncio.create_task(pipeline.run(self.input_room))
            tasks.append(recording)
            done, _ = await asyncio.wait((recording, transport, lost_output),
                                        return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            # Revoke the scene before closing outputs; the client's finally
            # independently sends stop to every member, even on transport loss.
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            cleanup = [s.aclose() for s in sessions]
            if pipeline is not None:
                cleanup.append(pipeline.shutdown())
            results = await asyncio.gather(*cleanup, return_exceptions=True)
            self.cleanup_ok = bool(self.client and self.client.cleanup_ok) and not any(
                isinstance(result, BaseException) for result in results)
