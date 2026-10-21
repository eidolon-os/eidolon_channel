"""Internal authenticated extraction surface; not enrollment or authorization."""

import asyncio
import base64
import binascii
import logging

from aiohttp import web
from eidolon_sdk.biz.voiceprint import (
    SpeakerExtractionReadiness,
    VoiceAudioRequest,
    EnrollmentEvaluationRequest,
)
from pydantic import ValidationError

from eidolon.livekit.plugins.speaker_verification.inference import (
    InferenceBusy,
    InferenceTimeout,
    ModelUnavailable,
)
from eidolon.livekit.plugins.speaker_verification.reference_extractor import InvalidSpeakerAudio
from eidolon.livekit.plugins.speaker_verification.enrollment import evaluate_enrollment


def register_speaker_extraction(app, *, engine, authorized, problem):
    warm_task = None

    def error(code, status, retryable=False):
        category = "auth" if status == 401 else ("unavailable" if retryable else "invalid")
        return problem(
            code=code, category=category, retryable=retryable, status=status, detail=code.lower()
        )

    async def readiness(request):
        if not authorized(request):
            return error("UNAUTHENTICATED", 401)
        state = engine.state if engine is not None else "disabled"
        result = SpeakerExtractionReadiness(state=state, extraction_ready=state == "ready")
        return web.json_response(result.model_dump(mode="json"))

    async def extract(request):
        if not authorized(request):
            return error("UNAUTHENTICATED", 401)
        if request.content_type != "application/json":
            return error("INVALID_CONTENT_TYPE", 415)
        if engine is None or engine.state != "ready":
            return error("MODEL_NOT_READY", 503, True)
        try:
            # Only this route needs 30s PCM base64. Preserve core control-plane limits.
            audio_request = request.clone(client_max_size=1_310_720)
            body = VoiceAudioRequest.model_validate_json(await audio_request.read())
            pcm = base64.b64decode(body.pcm_base64, validate=True)
        except (ValidationError, binascii.Error, ValueError):
            # ValidationError strings can include the entire supplied recording.
            return error("INVALID_AUDIO_REQUEST", 422)
        try:
            result = await engine.extract(pcm)
            return web.json_response(result.model_dump(mode="json"))
        except InvalidSpeakerAudio as exc:
            return error(str(exc).upper(), 422)
        except InferenceBusy:
            return error("INFERENCE_BUSY", 429, True)
        except InferenceTimeout:
            return error("INFERENCE_TIMEOUT", 504, True)
        except ModelUnavailable:
            return error("MODEL_NOT_READY", 503, True)
        except Exception:
            return error("INFERENCE_FAILED", 503, True)

    async def warm():
        try:
            await engine.start()
        except ModelUnavailable:
            logging.getLogger(__name__).warning("Speaker extraction model unavailable")

    async def evaluate(request):
        if not authorized(request):
            return error("UNAUTHENTICATED", 401)
        if request.content_type != "application/json":
            return error("INVALID_CONTENT_TYPE", 415)
        try:
            body = EnrollmentEvaluationRequest.model_validate_json(await request.read())
            result = evaluate_enrollment(body)
        except (ValueError, ValidationError):
            return error("INVALID_ENROLLMENT_EVIDENCE", 422)
        return web.json_response(result.model_dump(mode="json"))

    async def start(_app):
        nonlocal warm_task
        if engine is not None:
            warm_task = asyncio.create_task(warm())

    async def close(_app):
        if warm_task is not None:
            warm_task.cancel()
            await asyncio.gather(warm_task, return_exceptions=True)
        if engine is not None:
            await engine.close()

    app.on_startup.append(start)
    app.on_cleanup.append(close)
    app.router.add_get("/v1/voiceprints/readiness", readiness)
    app.router.add_post("/v1/voiceprints/extract", extract)
    app.router.add_post("/v1/voiceprints/evaluate-enrollment", evaluate)
