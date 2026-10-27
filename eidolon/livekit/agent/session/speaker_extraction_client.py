"""Use the existing Channel Provider extractor; no worker-local model or store."""

import base64
import json

from eidolon_sdk.biz.voiceprint import (
    ExtractedSample,
    SpeakerExtractionRequest,
    SpeakerExtractionReadiness,
    SpeakerAudioQuality,
)
from eidolon_sdk.core.http.client import ServiceHTTPClient, authorization_header
from eidolon_sdk.core.http.errors import ServiceUpstreamError


class SpeakerExtractionRejected(ServiceUpstreamError):
    """Allowlisted extraction outcome, without raw validation inputs or errors."""

    def __init__(self, reason, quality):
        super().__init__(422, reason)
        self.reason, self.quality = reason, quality

    @property
    def audio_quality_failure(self):
        return self.reason in _QUALITY_REASONS


_QUALITY_REASONS = {"audio_too_short", "speech_too_short", "digital_silence", "audio_clipped"}
_REJECTION_REASONS = _QUALITY_REASONS | {"invalid_pcm", "invalid_audio_request"}


class SpeakerExtractionClient(ServiceHTTPClient):
    """The caller owns the pooled HTTP client and its lifespan. No POST retries."""

    def __init__(self, http_client, base_url: str, *, token: str):
        super().__init__(http_client, base_url, default_headers=authorization_header(token))

    async def extract(self, pcm: bytes, *, purpose="enrollment", min_speech_ms=1500) -> ExtractedSample:
        if not isinstance(pcm, bytes) or len(pcm) % 2 or not 9600 <= len(pcm) <= 960000:
            raise ValueError("bounded mono 16 kHz PCM required")
        request = SpeakerExtractionRequest(
            pcm_base64=base64.b64encode(pcm).decode("ascii"),
            purpose=purpose, min_speech_ms=min_speech_ms,
        )
        try:
            response = await self._request(
                "POST", "/v1/voiceprints/extract", json=request.model_dump(mode="json"), timeout=8
            )
        except ServiceUpstreamError as exc:
            if exc.status_code == 422:
                # Do not forward arbitrary upstream text or validation inputs to the LLM/log.
                try:
                    body = json.loads(exc.message)
                    reason = str(body.get("code", "")).lower()
                    if reason in _REJECTION_REASONS:
                        quality = (SpeakerAudioQuality.model_validate(body["audio_quality"])
                                   if body.get("audio_quality") is not None else None)
                        raise SpeakerExtractionRejected(reason, quality) from None
                except (ValueError, TypeError, AttributeError):
                    pass
            raise
        return ExtractedSample.model_validate(response.json())

    async def readiness(self) -> SpeakerExtractionReadiness:
        response = await self._request("GET", "/v1/voiceprints/readiness", timeout=2)
        return SpeakerExtractionReadiness.model_validate(response.json())
