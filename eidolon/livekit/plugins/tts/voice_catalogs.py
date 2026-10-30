"""Which TTS providers take part in voice selection, beside the providers themselves.

Keyed by the same names as ``TTS_PROVIDERS``. A provider listed as pending is a
TTS this Host can run whose voice adapter is not written yet; it speaks with its
configured voice. Adding a TTS provider means deciding which list it joins —
the registry test fails until someone does.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from importlib import import_module
from typing import Any

from eidolon_sdk.biz.speech_voice import VoiceRef

from eidolon.speech_voice.adapter import VoiceCatalogAdapter

logger = logging.getLogger(__name__)

VOICE_CATALOG_ADAPTERS: dict[str, str] = {
    "bailian": "eidolon.livekit.plugins.tts.bailian.voice_catalog:BailianVoiceCatalogAdapter",
}
PENDING_PROVIDERS = frozenset({"sensetime", "local_tts"})


def adapter_class(provider: str | None) -> type[VoiceCatalogAdapter] | None:
    target = VOICE_CATALOG_ADAPTERS.get(provider or "")
    if target is None:
        return None
    module, name = target.split(":")
    return getattr(import_module(module), name)


def all_adapter_classes() -> tuple[type[VoiceCatalogAdapter], ...]:
    return tuple(adapter_class(provider) for provider in VOICE_CATALOG_ADAPTERS)


def current_adapter(cfg: Any) -> VoiceCatalogAdapter | None:
    """The adapter for the TTS this config runs, built from its plugin config."""
    cls = adapter_class(cfg.providers.tts_provider)
    return cls(getattr(cfg, cls.config_attr)) if cls is not None else None


def config_speaking(cfg: Any, voice: VoiceRef) -> Any:
    """The config with ``voice`` written into its TTS plugin config by that provider's adapter.

    A voice for another provider or model can only come from a provider and a
    worker that disagree about the config (a restart in between); the
    configured voice is then the one this TTS can certainly speak.
    """
    adapter = current_adapter(cfg)
    if adapter is None:
        return cfg
    configured = adapter.host_default()
    if (voice.provider, voice.model) != (configured.provider, configured.model):
        logger.error("speech voice for %s/%s does not fit %s/%s; using the configured voice",
                     voice.provider, voice.model, configured.provider, configured.model)
        return cfg
    return replace(cfg, **{adapter.config_attr: adapter.with_voice(voice.voice_id)})
