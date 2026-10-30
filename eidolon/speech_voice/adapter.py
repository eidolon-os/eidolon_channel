"""The ports a TTS provider implements to take part in voice selection.

``VoiceCatalogAdapter`` is what every provider answers: which voices exist,
which this Host can produce now, and how a voice is written into its plugin
config. ``VoiceEnrollmentAdapter`` is optional: making new voices from a
description or a recording. All implementations are peers; nothing here, or in
the code that uses it, branches on which provider is behind it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

from eidolon_sdk.biz.speech_voice import EnrollmentCapabilities, OwnerVoiceBinding, VoiceRef

from .catalog_file import CatalogVoice, load_catalog_file


@dataclass(frozen=True)
class VoiceDescriptor:
    """One described voice of one provider, independent of this Host."""

    provider: str
    voice: CatalogVoice


@dataclass(frozen=True)
class ProviderVoices:
    """What ``list_voices`` found: described voices this Host can produce now."""

    voices: tuple[CatalogVoice, ...]
    #: Voices the provider offers that the catalog file does not describe.
    uncurated: int = 0


class VoiceCatalogAdapter(ABC):
    provider: ClassVar[str]
    #: Path of the catalog file, relative to the adapter's module directory.
    catalog_file: ClassVar[str]
    #: Name of this provider's plugin config on the agent config.
    config_attr: ClassVar[str]

    def __init__(self, plugin_cfg: Any) -> None:
        self.plugin_cfg = plugin_cfg

    @classmethod
    def catalog_path(cls) -> Path:
        import sys

        return Path(sys.modules[cls.__module__].__file__).with_name(cls.catalog_file)

    @classmethod
    def describe(cls) -> tuple[VoiceDescriptor, ...]:
        """Every voice the catalog file describes. Reads no config, no credential, no network."""
        return _describe(cls)

    @abstractmethod
    def host_default(self) -> VoiceRef:
        """The voice the plugin config names."""

    @abstractmethod
    async def list_voices(self) -> ProviderVoices:
        """Described voices this Host can produce under its current config."""

    @abstractmethod
    def fingerprint(self) -> str:
        """What ``list_voices`` depends on; a stored result is reused only while it matches."""

    @abstractmethod
    def with_voice(self, voice_id: str) -> Any:
        """A copy of the plugin config that speaks with ``voice_id``."""


@cache
def _describe(cls: type[VoiceCatalogAdapter]) -> tuple[VoiceDescriptor, ...]:
    catalog = load_catalog_file(cls.catalog_path(), provider=cls.provider)
    return tuple(VoiceDescriptor(provider=cls.provider, voice=v) for v in catalog.voices)


@dataclass(frozen=True)
class DesignedVoice:
    binding: OwnerVoiceBinding
    preview_pcm: bytes
    sample_rate: int


class VoiceEnrollmentAdapter(ABC):
    """Making a new voice on the current provider model."""

    @abstractmethod
    def enrollment_capabilities(self) -> EnrollmentCapabilities: ...

    @abstractmethod
    async def design(self, *, prompt: str, preview_text: str) -> DesignedVoice: ...

    @abstractmethod
    async def clone(self, *, reference_wav: bytes) -> OwnerVoiceBinding: ...

    @abstractmethod
    async def status(self, provider_voice_id: str) -> OwnerVoiceBinding: ...

    @abstractmethod
    async def delete(self, provider_voice_id: str) -> None:
        """A voice the provider no longer has counts as deleted."""

    @abstractmethod
    async def list_own(self) -> Sequence[str]:
        """Provider voice ids this Host created."""


class EnrollmentRejected(Exception):
    """The provider refused the request itself (bad audio, quota, policy)."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code
