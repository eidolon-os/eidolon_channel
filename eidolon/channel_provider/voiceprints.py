"""Authenticated, bounded extraction endpoint. No profiles are stored here."""

import base64
import binascii
from aiohttp import web
from pydantic import ValidationError
from eidolon_sdk.biz.voiceprint import VoiceAudioRequest
from eidolon.livekit.plugins.speaker_verification.extractor import (
    shared_voice_extractor,
    InsufficientSpeech,
)


def install_voiceprint_routes(app, authorized, extractor=None):
    engine = extractor or shared_voice_extractor()

    async def extract(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        # Increase only this route's body limit; existing control routes retain theirs.
        bounded = request.clone(client_max_size=1_300_000)
        try:
            body = VoiceAudioRequest.model_validate(await bounded.json())
            pcm = base64.b64decode(body.pcm_base64, validate=True)
            result = await engine.extract(pcm)
            return web.json_response(result.model_dump(mode="json"))
        except (ValidationError, ValueError, binascii.Error, InsufficientSpeech):
            # Never echo raw input or pydantic errors containing biometric data.
            return web.json_response({"detail": "clear mono 16 kHz speech required"}, status=422)
        except Exception:
            return web.json_response({"detail": "voice extraction unavailable"}, status=503)

    app.router.add_post("/v1/voiceprints/extract", extract)
