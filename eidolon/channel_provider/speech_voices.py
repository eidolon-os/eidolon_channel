"""channel-provider routes for voices: the Host catalog, resolution, previews and new voices.

Admin and the voice worker are the only callers, both with the provider token.
Ownership is Admin's to prove before it asks for anything Owner-specific here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import socket

from aiohttp import web
from eidolon_sdk.biz.speech_voice import (
    CloneRequest,
    DesignRequest,
    EnrolledVoice,
    ResolveRequest,
    VoiceRef,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eidolon.speech_voice.adapter import EnrollmentRejected, VoiceEnrollmentAdapter
from eidolon.speech_voice.catalog import HostVoiceCatalogService
from eidolon.speech_voice.preview import PREVIEW_TEXTS, RENDITION_TEXT, PreviewBusy, VoicePreviews
from eidolon.speech_voice.reference_audio import check_reference
from eidolon.speech_voice.resolve import resolve
from eidolon.speech_voice.wav import wav_bytes

MAX_UPLOAD = 12 * 1024 * 1024
_REFRESH = web.AppKey("speech_voice_refresh", asyncio.Task)


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ref: VoiceRef
    #: 1 or 2: one of the fixed sentences; 0: the rendition text a new voice reads.
    sample: int = Field(ge=0, le=len(PREVIEW_TEXTS))


class ConfiguredSynthesis:
    """Synthesize with the worker's own TTS build, speaking a given voice. No room needed."""

    def __init__(self, config) -> None:
        self.config = config

    async def __call__(self, ref: VoiceRef, text: str) -> tuple[bytes, int]:
        from eidolon.livekit.agent.factory import SharedStageFactory
        from eidolon.livekit.plugins.tts.voice_catalogs import config_speaking

        stage = SharedStageFactory._build_tts(config_speaking(self.config, ref))
        chunks, rate = [], None
        try:
            async for frame in stage.synthesize(text):
                if frame.num_channels != 1 or (rate is not None and frame.sample_rate != rate):
                    raise ValueError("preview synthesis must be mono at one rate")
                rate = frame.sample_rate
                chunks.append(bytes(frame.data))
        finally:
            await stage.shutdown()
            await stage.tts.aclose()
        return b"".join(chunks), rate or 16000


def host_propagator() -> str:
    """An opaque name for this Host in AIGC identifiers — never the provider account."""
    host = os.environ.get("EIDOLON_HOST_ID") or socket.gethostname()
    return "eidolon-host-" + hashlib.sha256(host.encode()).hexdigest()[:16]


def register_speech_voices(app, *, catalog: HostVoiceCatalogService, previews: VoicePreviews,
                           authorized, problem) -> None:
    def error(code: str, status: int, detail: str, retryable: bool = False) -> web.Response:
        category = {401: "auth", 409: "conflict", 422: "invalid", 429: "unavailable"}.get(status, "unavailable")
        return problem(code=code, category=category, retryable=retryable, status=status, detail=detail)

    def guarded(handler):
        async def wrapped(request: web.Request) -> web.StreamResponse:
            if not authorized(request):
                return error("UNAUTHENTICATED", 401, "provider token required")
            try:
                return await handler(request)
            except ValidationError as exc:
                return error("INVALID_ARGUMENT", 422, exc.errors(include_url=False)[0]["msg"])
            except EnrollmentRejected as exc:
                return error("ENROLLMENT_REJECTED", 422, f"{exc.code}: {exc}")
            except PreviewBusy:
                return error("PREVIEW_BUSY", 429, "previews are busy; try again shortly", True)
        return wrapped

    def json_model(model, *, headers=None) -> web.Response:
        return web.json_response(model.model_dump(mode="json"), headers=headers)

    def enrollment() -> VoiceEnrollmentAdapter | None:
        adapter = catalog.adapter
        return adapter if isinstance(adapter, VoiceEnrollmentAdapter) else None

    async def read_limited(request: web.Request) -> bytes:
        # The app-wide body limit is small on purpose; only audio may be larger.
        data = bytearray()
        async for chunk in request.content.iter_chunked(64 * 1024):
            data += chunk
            if len(data) > MAX_UPLOAD:
                raise web.HTTPRequestEntityTooLarge(max_size=MAX_UPLOAD, actual_size=len(data))
        return bytes(data)

    async def get_catalog(request: web.Request) -> web.StreamResponse:
        current = catalog.catalog()
        etag = f'"{current.catalog_revision}"'
        if request.headers.get("If-None-Match") == etag:
            return web.Response(status=304, headers={"ETag": etag})
        return json_model(current, headers={"ETag": etag})

    async def refresh(_request: web.Request) -> web.Response:
        return json_model(await catalog.refresh_on_request())

    async def resolve_voice(request: web.Request) -> web.Response:
        body = ResolveRequest.model_validate_json(await request.read())
        return json_model(resolve(body.speech_voice, catalog.catalog(), catalog.overlay))

    async def preview(request: web.Request) -> web.Response:
        body = PreviewRequest.model_validate_json(await request.read())
        current = catalog.catalog()
        default = next((v.ref for v in current.voices if v.is_host_default), None)
        if default is None or (body.ref.provider, body.ref.model) != (default.provider, default.model):
            return error("VOICE_NOT_ON_THIS_HOST", 409, "this Host does not speak that model")
        text = RENDITION_TEXT if body.sample == 0 else PREVIEW_TEXTS[body.sample - 1]
        data = await previews.wav(body.ref, text, keep=body.sample != 0)
        return web.Response(body=data, content_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})

    async def check(request: web.Request) -> web.Response:
        data = await read_limited(request)
        start = request.query.get("start_ms")
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: check_reference(data, start_ms=int(start) if start else None))
        return json_model(result)

    async def design(request: web.Request) -> web.Response:
        adapter = enrollment()
        if adapter is None or not adapter.enrollment_capabilities().design:
            return error("DESIGN_UNAVAILABLE", 409, "this Host's TTS cannot design voices")
        body = DesignRequest.model_validate_json(await request.read())
        made = await adapter.design(prompt=body.prompt, preview_text=body.preview_text)
        wav = wav_bytes(made.preview_pcm, made.sample_rate, aigc={
            "ContentProducer": f"eidolon/{made.binding.provider}",
            "ProduceID": made.binding.provider_voice_id[-32:], "ContentPropagator": host_propagator(),
            "PropagateID": made.binding.provider_voice_id[-32:],
        })
        return json_model(EnrolledVoice(binding=made.binding, preview_wav_base64=base64.b64encode(wav).decode()))

    async def clone(request: web.Request) -> web.Response:
        adapter = enrollment()
        if adapter is None or not adapter.enrollment_capabilities().clone:
            return error("CLONE_UNAVAILABLE", 409, "this Host's TTS cannot clone voices")
        body = CloneRequest.model_validate_json(await read_limited(request))
        binding = await adapter.clone(reference_wav=base64.b64decode(body.reference_wav_base64, validate=True))
        return json_model(EnrolledVoice(binding=binding))

    def own(request: web.Request) -> VoiceEnrollmentAdapter | None:
        adapter = enrollment()
        if adapter is None or request.query.get("provider") != getattr(adapter, "provider", None):
            return None
        return adapter

    async def status(request: web.Request) -> web.Response:
        adapter = own(request)
        if adapter is None:
            return error("PROVIDER_NOT_CURRENT", 409, "that provider is not this Host's TTS")
        return json_model(await adapter.status(request.match_info["voice_id"]))

    async def delete(request: web.Request) -> web.Response:
        adapter = own(request)
        if adapter is None:
            return error("PROVIDER_NOT_CURRENT", 409, "that provider is not this Host's TTS")
        await adapter.delete(request.match_info["voice_id"])
        return web.Response(status=204)

    async def list_own(request: web.Request) -> web.Response:
        adapter = own(request)
        if adapter is None:
            return error("PROVIDER_NOT_CURRENT", 409, "that provider is not this Host's TTS")
        return web.json_response(list(await adapter.list_own()))

    base = "/v1/speech-voices"
    app.router.add_get(base, guarded(get_catalog))
    app.router.add_post(base + "/refresh", guarded(refresh))
    app.router.add_post(base + "/resolve", guarded(resolve_voice))
    app.router.add_post(base + "/previews", guarded(preview))
    app.router.add_post(base + "/references/check", guarded(check))
    app.router.add_post(base + "/enrollments/design", guarded(design))
    app.router.add_post(base + "/enrollments/clone", guarded(clone))
    app.router.add_get(base + "/enrollments", guarded(list_own))
    app.router.add_get(base + "/enrollments/{voice_id}", guarded(status))
    app.router.add_delete(base + "/enrollments/{voice_id}", guarded(delete))

    async def start(_app):
        app[_REFRESH] = asyncio.create_task(catalog.run(), name="speech-voice-catalog")

    async def stop(_app):
        task = app.get(_REFRESH)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
