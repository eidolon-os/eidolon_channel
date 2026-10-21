"""Use the existing Channel Provider extractor; no worker-local model or store."""

import base64

from eidolon_sdk.biz.voiceprint import ExtractedSample, VoiceAudioRequest
from eidolon_sdk.core.http.client import ServiceHTTPClient, authorization_header


class SpeakerExtractionClient(ServiceHTTPClient):
    """The caller owns the pooled HTTP client and its lifespan. No POST retries."""

    def __init__(self, http_client, base_url: str, *, token: str):
        super().__init__(http_client, base_url, default_headers=authorization_header(token))

    async def extract(self, pcm: bytes) -> ExtractedSample:
        if not isinstance(pcm, bytes) or len(pcm) % 2 or not 48000 <= len(pcm) <= 960000:
            raise ValueError("bounded mono 16 kHz PCM required")
        request = VoiceAudioRequest(pcm_base64=base64.b64encode(pcm).decode("ascii"))
        response = await self._request("POST", "/v1/voiceprints/extract",
            json=request.model_dump(mode="json"), timeout=8)
        return ExtractedSample.model_validate(response.json())
