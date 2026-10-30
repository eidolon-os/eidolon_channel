"""Bailian (DashScope) voices: the catalog adapter and voice design / cloning.

System voices have no list API, so ``list_voices`` reads ``voices.yaml`` and
keeps what the configured model can speak in the region ``api_url`` reaches.
Custom voices go through the customization endpoint on the same host; every
one created here carries this Host's prefix so it can be found again.

References: CosyVoice / Qwen-Audio-TTS voice lists, the WebSocket API (regional
domains), and the voice clone / voice design HTTP API on help.aliyun.com.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from urllib.parse import urlparse

import aiohttp
from eidolon_sdk.biz.speech_voice import EnrollmentCapabilities, OwnerVoiceBinding, VoiceRef

from eidolon.speech_voice.adapter import (
    DesignedVoice,
    EnrollmentRejected,
    ProviderVoices,
    VoiceCatalogAdapter,
    VoiceEnrollmentAdapter,
)
from eidolon.speech_voice.wav import wav_pcm

from .config import BailianTTSConfig

BEIJING, SINGAPORE = "cn-beijing", "ap-southeast-1"
_REGION_HOSTS = {"dashscope.aliyuncs.com": BEIJING, "dashscope-intl.aliyuncs.com": SINGAPORE}
#: Which models can make a custom voice, by region (voice clone / voice design guides, 2026-10-30).
_CLONE = {
    BEIJING: {"qwen-audio-3.1-tts-plus", "qwen-audio-3.1-tts-flash", "qwen-audio-3.0-tts-plus",
              "qwen-audio-3.0-tts-flash", "cosyvoice-v3.5-plus", "cosyvoice-v3.5-flash",
              "cosyvoice-v3-plus", "cosyvoice-v3-flash", "cosyvoice-v2"},
    SINGAPORE: {"qwen-audio-3.1-tts-plus", "qwen-audio-3.1-tts-flash", "qwen-audio-3.0-tts-plus",
                "qwen-audio-3.0-tts-flash", "cosyvoice-v3-plus", "cosyvoice-v3-flash"},
}
_DESIGN = {
    BEIJING: {"qwen-audio-3.1-tts-plus", "qwen-audio-3.1-tts-flash", "qwen-audio-3.0-tts-plus",
              "qwen-audio-3.0-tts-flash", "cosyvoice-v3.5-plus", "cosyvoice-v3.5-flash",
              "cosyvoice-v3-plus", "cosyvoice-v3-flash"},
}
_STATUS = {"OK": "ready", "DEPLOYING": "deploying", "UNDEPLOYED": "rejected"}
_TIMEOUT = aiohttp.ClientTimeout(total=60)


def region_of(api_url: str) -> str | None:
    host = urlparse(api_url).hostname or ""
    if host in _REGION_HOSTS:
        return _REGION_HOSTS[host]
    for region in (BEIJING, SINGAPORE):
        if host.endswith(f".{region}.maas.aliyuncs.com"):
            return region
    return None


def host_prefix() -> str:
    """``e`` + 9 hex of this Host's id: what Bailian allows (letters/digits, ≤10)."""
    host = os.environ.get("EIDOLON_HOST_ID") or socket.gethostname()
    return "e" + hashlib.sha256(host.encode()).hexdigest()[:9]


class BailianVoiceCatalogAdapter(VoiceCatalogAdapter, VoiceEnrollmentAdapter):
    provider = "bailian"
    catalog_file = "voices.yaml"
    config_attr = "bailian_tts"
    plugin_cfg: BailianTTSConfig

    # ------------------------------------------------------------------ catalog

    def host_default(self) -> VoiceRef:
        return VoiceRef(provider=self.provider, model=self.plugin_cfg.model, voice_id=self.plugin_cfg.voice)

    @property
    def region(self) -> str | None:
        return region_of(self.plugin_cfg.api_url)

    async def list_voices(self) -> ProviderVoices:
        region, model = self.region, self.plugin_cfg.model
        # An address we cannot place: only voices both regions serve are safe.
        allowed = {region} if region else {BEIJING, SINGAPORE}
        return ProviderVoices(voices=tuple(
            d.voice for d in self.describe()
            if model in d.voice.models and allowed <= set(d.voice.ext.get("regions", ()))
        ))

    def fingerprint(self) -> str:
        return f"{self.provider}|{self.plugin_cfg.model}|{self.region}"

    def with_voice(self, voice_id: str) -> BailianTTSConfig:
        return replace(self.plugin_cfg, voice=voice_id)

    # ------------------------------------------------------------------ custom voices

    def enrollment_capabilities(self) -> EnrollmentCapabilities:
        region, model = self.region, self.plugin_cfg.model
        return EnrollmentCapabilities(
            design=model in _DESIGN.get(region, ()), clone=model in _CLONE.get(region, ()),
            processor="阿里云百炼",
        )

    @property
    def _customization_url(self) -> str:
        return f"https://{urlparse(self.plugin_cfg.api_url).hostname}/api/v1/services/audio/tts/customization"

    async def _call(self, action: str, parameters: dict | None = None, **fields) -> dict:
        body = {"model": "voice-enrollment", "input": {"action": action, **fields}}
        if parameters:
            body["parameters"] = parameters
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as http, http.post(
            self._customization_url, json=body,
            headers={"Authorization": f"Bearer {self.plugin_cfg.api_key}"},
        ) as response:
            data = await response.json(content_type=None) or {}
            if response.status == 200:
                return data.get("output") or {}
            code = data.get("code") or f"http_{response.status}"
            if 400 <= response.status < 500:
                raise EnrollmentRejected(data.get("message") or code, code=code)
            raise RuntimeError(f"Bailian customization {action} failed: {response.status} {code}")

    def _binding(self, voice_id: str, status: str, detail: str | None = None) -> OwnerVoiceBinding:
        return OwnerVoiceBinding(
            provider=self.provider, model=self.plugin_cfg.model, provider_voice_id=voice_id,
            status=status, status_detail=detail, last_checked_at=datetime.now(UTC),
        )

    async def design(self, *, prompt: str, preview_text: str) -> DesignedVoice:
        output = await self._call(
            "create_voice", parameters={"sample_rate": 24000, "response_format": "wav"},
            target_model=self.plugin_cfg.model, prefix=host_prefix(),
            voice_prompt=prompt, preview_text=preview_text, language_hints=["zh"],
        )
        pcm, rate = wav_pcm(base64.b64decode(output["preview_audio"]["data"]))
        return DesignedVoice(binding=await self.status(output["voice_id"]), preview_pcm=pcm, sample_rate=rate)

    async def clone(self, *, reference_wav: bytes) -> OwnerVoiceBinding:
        output = await self._call(
            "create_voice", target_model=self.plugin_cfg.model, prefix=host_prefix(),
            audio="data:audio/wav;base64," + base64.b64encode(reference_wav).decode("ascii"),
            language_hints=["zh"],
        )
        return self._binding(output["voice_id"], "deploying")

    async def status(self, provider_voice_id: str) -> OwnerVoiceBinding:
        try:
            output = await self._call("query_voice", voice_id=provider_voice_id)
        except EnrollmentRejected as exc:
            if exc.code == "BadRequest.ResourceNotExist":
                return self._binding(provider_voice_id, "failed", "服务商那边已没有这个声音")
            raise
        # ``resource_link`` (the provider's copy of the reference audio) never leaves here.
        status = _STATUS.get(output.get("status", ""), "deploying")
        return self._binding(provider_voice_id, status)

    async def delete(self, provider_voice_id: str) -> None:
        try:
            await self._call("delete_voice", voice_id=provider_voice_id)
        except EnrollmentRejected as exc:
            if exc.code != "BadRequest.ResourceNotExist":
                raise

    async def list_own(self) -> Sequence[str]:
        found: list[str] = []
        page = 0
        while True:
            output = await self._call("list_voice", prefix=host_prefix(), page_index=page, page_size=100)
            voices = output.get("voice_list") or []
            found += [v["voice_id"] for v in voices if v.get("voice_id")]
            if len(voices) < 100:
                return found
            page += 1
