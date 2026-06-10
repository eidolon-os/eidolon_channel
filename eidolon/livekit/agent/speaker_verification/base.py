"""Provider contract for speaker verification engines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .signal import SpeakerSignal
from .store import VoiceprintProfile


class SpeakerVerificationProvider(Protocol):
    """Minimal async contract shared by noop and real voiceprint providers."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def enroll(
        self,
        *,
        tenant_id: str,
        user_id: str,
        audio_segments: Sequence[bytes],
        sample_rate: int,
    ) -> VoiceprintProfile:
        """Create a profile from enrollment PCM/WAV bytes."""

    async def verify(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        profiles: Sequence[VoiceprintProfile],
        audio_ms: int | None = None,
    ) -> SpeakerSignal:
        """Return the best speaker match for one voiced turn."""
