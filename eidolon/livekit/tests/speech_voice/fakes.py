"""A provider-neutral adapter for tests, described by a catalog file written on the fly."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import yaml
from eidolon_sdk.biz.speech_voice import EnrollmentCapabilities, OwnerVoiceBinding, VoiceRef

from eidolon.speech_voice.adapter import (
    DesignedVoice,
    ProviderVoices,
    VoiceCatalogAdapter,
    VoiceEnrollmentAdapter,
)


@dataclass
class FakeConfig:
    model: str = "m1"
    voice: str = "warm-f"


VOICES = [
    {"voice_id": "warm-f", "display_name": "暖暖", "summary": "温暖女声", "gender": "female", "age_band": "youth",
     "languages": ["zh-CN", "en"], "styles": ["warm", "gentle"], "rank": 101, "models": ["m1", "m2"]},
    {"voice_id": "bright-f", "display_name": "亮亮", "gender": "female", "age_band": "youth",
     "languages": ["zh-CN"], "styles": ["bright", "lively"], "rank": 102, "models": ["m1"]},
    {"voice_id": "gentle-f", "display_name": "柔柔", "gender": "female", "age_band": "youth",
     "languages": ["zh-CN"], "styles": ["gentle", "sweet"], "rank": 103, "models": ["m2"]},
    {"voice_id": "calm-m", "display_name": "稳稳", "gender": "male", "age_band": "adult",
     "languages": ["zh-CN"], "styles": ["calm"], "rank": 104, "models": ["m1"]},
    {"voice_id": "warm-f-v2", "display_name": "暖暖", "gender": "female", "age_band": "youth",
     "languages": ["zh-CN"], "styles": ["warm"], "lineage": "warm-f", "rank": 105, "models": ["m2"]},
    {"voice_id": "yue-f", "display_name": "粤粤", "gender": "female", "age_band": "youth",
     "languages": ["yue"], "styles": ["warm"], "rank": 106, "models": ["m1"]},
    {"voice_id": "robot", "display_name": "机机", "gender": "unspecified", "age_band": "youth",
     "languages": ["zh-CN"], "styles": ["character"], "rank": 107, "models": ["m1"]},
]


def write_catalog(path: Path, provider: str = "fake", voices=VOICES) -> Path:
    path.write_text(yaml.safe_dump({
        "schema": "speech_voice_catalog/v1", "provider": provider,
        "source": {"title": "t", "url": "https://example.invalid", "retrieved_on": "2026-10-31"},
        "models": ["m1", "m2"], "voices": voices,
    }, allow_unicode=True), encoding="utf-8")
    return path


def adapter_class(catalog: Path, *, provider: str = "fake", fail: list | None = None):
    """A fresh adapter class per catalog (``describe`` caches per class)."""

    class Fake(VoiceCatalogAdapter, VoiceEnrollmentAdapter):
        config_attr = "fake_tts"
        catalog_file = catalog.name
        failures = fail if fail is not None else []
        designed: ClassVar[list] = []
        deleted: ClassVar[list] = []

        @classmethod
        def catalog_path(cls) -> Path:
            return catalog

        def host_default(self) -> VoiceRef:
            return VoiceRef(provider=self.provider, model=self.plugin_cfg.model, voice_id=self.plugin_cfg.voice)

        async def list_voices(self) -> ProviderVoices:
            if self.failures:
                raise self.failures.pop(0)
            return ProviderVoices(voices=tuple(
                d.voice for d in self.describe() if self.plugin_cfg.model in d.voice.models))

        def fingerprint(self) -> str:
            return f"{self.provider}|{self.plugin_cfg.model}"

        def with_voice(self, voice_id: str):
            return replace(self.plugin_cfg, voice=voice_id)

        def enrollment_capabilities(self) -> EnrollmentCapabilities:
            return EnrollmentCapabilities(design=True, clone=True, processor="假服务")

        def _binding(self, voice_id: str, status: str) -> OwnerVoiceBinding:
            return OwnerVoiceBinding(provider=self.provider, model=self.plugin_cfg.model,
                                     provider_voice_id=voice_id, status=status)

        async def design(self, *, prompt: str, preview_text: str) -> DesignedVoice:
            self.designed.append(prompt)
            return DesignedVoice(binding=self._binding("m1-vd-e1-x", "ready"),
                                 preview_pcm=b"\x01\x00" * 24000, sample_rate=24000)

        async def clone(self, *, reference_wav: bytes) -> OwnerVoiceBinding:
            return self._binding("m1-e1-c", "deploying")

        async def status(self, provider_voice_id: str) -> OwnerVoiceBinding:
            return self._binding(provider_voice_id, "ready")

        async def delete(self, provider_voice_id: str) -> None:
            self.deleted.append(provider_voice_id)

        async def list_own(self):
            return ["m1-e1-c", "m1-e1-orphan"]

    Fake.provider = provider
    return Fake
