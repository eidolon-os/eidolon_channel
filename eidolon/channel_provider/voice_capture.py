"""Short-lived, owner-scoped management recordings over the conversation transport.

No Agent dispatch, identity database, or disk recording. Only the expected
publisher's single audio track is collected; final PCM is returned once.
"""
import asyncio
import base64
import uuid
from dataclasses import dataclass, field

from aiohttp import web
from livekit import rtc
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from eidolon_sdk.integrations.livekit import build_livekit_token


class CaptureOwner(BaseModel):
    model_config = ConfigDict(extra='forbid')
    owner_id: str = Field(min_length=1, max_length=128)


@dataclass
class Capture:
    owner: str
    room: object
    publisher: str
    pcm: bytearray = field(default_factory=bytearray)
    task: object = None
    expiry: object = None
    failed: bool = False
    finishing: bool = False


class VoiceCapture:
    def __init__(self, config, client_urls):
        self.config, self.client_urls = config, client_urls
        self.sessions = {}

    async def start(self, owner):
        # Reserve before any network await, so concurrent callers cannot overbook.
        if len(self.sessions) >= 4 or any(s.owner == owner for s in self.sessions.values()):
            raise web.HTTPConflict(text='录音仍在进行，请稍后重试')
        key = uuid.uuid4().hex
        room_name = 'voice-capture-' + key
        publisher = 'microphone-' + key
        room = rtc.Room()
        session = Capture(owner, room, publisher)
        self.sessions[key] = session
        def token(identity, publish):
            return build_livekit_token(api_key=self.config.api_key,
                api_secret=self.config.api_secret, room_name=room_name, identity=identity,
                dispatch_agent=False, can_publish=publish, can_subscribe=not publish,
                can_publish_data=False)
        @room.on('track_subscribed')
        def track_subscribed(track, publication, participant):
            if participant.identity != publisher or track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if session.task is not None:
                session.failed = True
                return
            async def collect():
                stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=20)
                try:
                    async for event in stream:
                        data = bytes(event.frame.data)
                        if len(session.pcm) + len(data) > 32000 * 12:
                            session.failed = True
                            break
                        session.pcm.extend(data)
                except Exception:
                    session.failed = True
                finally:
                    await stream.aclose()
            session.task = asyncio.create_task(collect())
        async def expire():
            await asyncio.sleep(45)
            await self.cancel(owner, key)
        try:
            await asyncio.wait_for(room.connect(self.config.api_url.replace('https://', 'wss://').replace('http://', 'ws://'), token('receiver-' + key, False)), 8)
            session.expiry = asyncio.create_task(expire())
            return {'capture_id':key, 'server_urls':self.client_urls(), 'token':token(publisher, True)}
        except BaseException:
            await self.cancel(owner, key)
            raise

    def get(self, owner, key):
        session = self.sessions.get(key)
        if session is None or session.owner != owner:
            raise web.HTTPNotFound()
        return session

    async def cancel(self, owner, key):
        session = self.get(owner, key)
        self.sessions.pop(key)
        if session.expiry is not None and session.expiry is not asyncio.current_task():
            session.expiry.cancel()
        if session.task is not None:
            session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)
        try:
            await session.room.disconnect()
        finally:
            session.pcm.clear()

    async def finish(self, owner, key):
        session = self.get(owner, key)
        if session.finishing:
            raise web.HTTPConflict()
        session.finishing = True
        if session.expiry is not None:
            session.expiry.cancel()
        try:
            # The sender stops publishing before finish. Drain media already in flight.
            await asyncio.sleep(.3)
            if session.failed or len(session.pcm) < 32000 * 3:
                raise web.HTTPUnprocessableEntity(text='录音不完整或不足三秒，请重新录制')
            return {'pcm_base64':base64.b64encode(session.pcm).decode('ascii')}
        finally:
            if self.sessions.get(key) is session:
                await self.cancel(owner, key)

    async def close(self):
        for key, session in list(self.sessions.items()):
            await self.cancel(session.owner, key)


def register_voice_capture(app, *, capture, authorized):
    async def handle(request):
        if not authorized(request): raise web.HTTPUnauthorized()
        if capture is None: raise web.HTTPServiceUnavailable()
        try: owner = CaptureOwner.model_validate_json(await request.read()).owner_id
        except (ValidationError, ValueError): raise web.HTTPBadRequest() from None
        key = request.match_info.get('capture_id')
        if key is None:
            result = await capture.start(owner)
        elif request.method == 'DELETE':
            await capture.cancel(owner, key)
            result = {}
        else:
            result = await capture.finish(owner, key)
        return web.json_response(result, headers={'Cache-Control':'no-store'})
    app.router.add_post('/v1/voiceprints/captures', handle)
    app.router.add_post('/v1/voiceprints/captures/{capture_id}/finish', handle)
    app.router.add_delete('/v1/voiceprints/captures/{capture_id}', handle)
    async def cleanup(_app):
        if capture is not None: await capture.close()
    app.on_cleanup.append(cleanup)
