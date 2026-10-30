"""Bounded Owner-scoped cache around existing TTS; no room or dialogue runtime."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from eidolon_sdk.biz.control.voice_preview import VoicePreviewStatus

TEXT = '你好，很高兴认识你。接下来，让我们一起聊聊今天的生活吧。'


class VoicePreviewCache:
    def __init__(self, root, *, profile, synthesize, max_bytes=64 * 1024 * 1024):
        self.root = Path(root)
        self.profile, self.synthesize, self.max_bytes = profile, synthesize, max_bytes
        if max_bytes <= 0:
            raise ValueError("preview cache capacity must be positive")
        self._slots = asyncio.Semaphore(1)
        self._tasks = {}
        self._results = {}

    def _identity(self, owner):
        profile = self.profile()
        encoded = json.dumps({'owner': owner, 'profile': profile, 'text': TEXT,
            'version': 1, 'format': 'pcm_s16le', 'sample_rate': 16000}, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest(), profile

    def _read(self, key):
        path = self.root / (key + '.json')
        try:
            data = json.loads(path.read_bytes())
            pcm = base64.b64decode(data['pcm_s16le_base64'], validate=True)
            if len(pcm) % 2 or not 48000 <= len(pcm) <= 960000:
                return None
            os.utime(path, None)
            return data
        except (OSError, ValueError, KeyError):
            return None

    async def request(self, owner, request):
        key, profile = self._identity(owner)
        cached = self._read(key)
        operation = (key, request.request_id)
        if cached is None and self._results.get(operation) == 'ready':
            self._results.pop(operation)
        if key not in self._tasks and operation not in self._results:
            if cached is not None and (not request.regenerate or cached['request_id'] == request.request_id):
                self._results[operation] = 'ready'
            else:
                task = asyncio.create_task(self._generate(key, operation, profile))
                self._tasks[key] = task
        if key in self._tasks:
            self._results[operation] = 'generating'
        state = 'generating' if key in self._tasks else self._results.get(operation, 'failed')
        return VoicePreviewStatus(key=key, state=state, audio_available=cached is not None,
                                  voice=profile['voice'])

    async def _generate(self, key, operation, profile):
        try:
            async with self._slots, asyncio.timeout(60):
                pcm = await self.synthesize(TEXT)
            if len(pcm) % 2 or not 48000 <= len(pcm) <= 960000:
                raise ValueError('invalid preview audio')
            data = json.dumps({'request_id': operation[1],
                'pcm_s16le_base64': base64.b64encode(pcm).decode()}).encode()
            if len(data) > self.max_bytes:
                raise ValueError('preview exceeds cache capacity')
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.preview-', dir=self.root)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.root / (key + '.json'))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            files = sorted(self.root.glob('*.json'), key=lambda p: p.stat().st_mtime)
            total = sum(p.stat().st_size for p in files)
            for path in files:
                if total <= self.max_bytes:
                    break
                if path.stem == key or path.stem in self._tasks:
                    continue
                total -= path.stat().st_size
                path.unlink(missing_ok=True)
            self._results[operation] = 'ready'
        except Exception:
            # Keep the old complete sample on regeneration failure. Provider
            # errors/credentials never become public response text.
            self._results[operation] = 'failed'
        finally:
            outcome = self._results.get(operation, 'failed')
            for waiter in self._results:
                if waiter[0] == key and self._results[waiter] == 'generating':
                    self._results[waiter] = outcome
            self._tasks.pop(key, None)
            while len(self._results) > 256:
                self._results.pop(next(iter(self._results)))

    def audio(self, owner, key):
        current, _ = self._identity(owner)
        if current != key:
            raise KeyError('preview not available in current voice scope')
        data = self._read(key)
        if data is None:
            raise KeyError('preview not available')
        return {'pcm_base64': data['pcm_s16le_base64']}

    async def close(self):
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class ConfiguredVoicePreview:
    """Reuse the synthesis factory used by TeamWorker, without starting its worker."""
    def __init__(self, config):
        self.config = config

    def profile(self):
        provider = self.config.providers.tts_provider
        settings = getattr(self.config, 'local_tts' if provider in {'local', 'local_tts'} else provider + '_tts')
        # Only synthesis-affecting fields. Never credentials, pool or transport settings.
        fields = ('voice', 'voice_id', 'model', 'speed', 'speech_rate', 'pitch',
                  'volume', 'sample_rate', 'audio_format', 'language', 'style')
        return {'provider': provider, 'implementation': 1,
                **{name: getattr(settings, name) for name in fields if hasattr(settings, name)},
                'voice': getattr(settings, 'voice', None) or getattr(settings, 'voice_id', None) or 'Host 默认声音'}

    async def synthesize(self, text):
        from livekit import rtc
        from eidolon.livekit.agent.factory import SharedStageFactory
        stage = SharedStageFactory._build_tts(self.config)
        chunks, resampler = [], None
        try:
            async for frame in stage.synthesize(text):
                if frame.num_channels != 1:
                    raise ValueError('preview requires mono synthesis')
                if frame.sample_rate == 16000:
                    frames = [frame]
                else:
                    if resampler is None:
                        resampler = rtc.AudioResampler(input_rate=frame.sample_rate,
                            output_rate=16000, num_channels=1)
                    frames = resampler.push(frame)
                chunks.extend(bytes(part.data) for part in frames)
                if sum(map(len, chunks)) > 960000:
                    raise ValueError('preview too long')
            if resampler is not None:
                chunks.extend(bytes(frame.data) for frame in resampler.flush())
            return b''.join(chunks)
        finally:
            await stage.shutdown()
            await stage.tts.aclose()


def register_voice_preview(app, *, cache, authorized):
    from aiohttp import web
    from pydantic import ValidationError
    from eidolon_sdk.biz.control.voice_preview import VoicePreviewRequest

    async def request_preview(request):
        if not authorized(request):
            return web.Response(status=401)
        if cache is None:
            return web.Response(status=503)
        try:
            body = VoicePreviewRequest.model_validate(await request.json())
            result = await cache.request(request.match_info['owner'], body)
            return web.json_response(result.model_dump(mode='json'))
        except (ValueError, ValidationError):
            return web.Response(status=422)

    async def audio(request):
        if not authorized(request):
            return web.Response(status=401)
        if cache is None:
            return web.Response(status=503)
        try:
            return web.json_response(cache.audio(request.match_info['owner'], request.match_info['key']),
                                     headers={'Cache-Control': 'no-store'})
        except KeyError:
            return web.Response(status=404)

    app.router.add_post('/v1/owners/{owner}/voice-preview', request_preview)
    app.router.add_get('/v1/owners/{owner}/voice-preview/{key}/audio', audio)
    if cache is not None:
        async def shutdown(_app):
            await cache.close()
        app.on_cleanup.append(shutdown)
