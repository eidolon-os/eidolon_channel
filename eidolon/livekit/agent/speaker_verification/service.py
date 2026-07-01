"""Observe-only orchestration around speaker verification providers."""

from __future__ import annotations

import time
from collections.abc import Sequence

from eidolon.livekit.common.speaker_verification import (
    SpeakerSignal,
    SpeakerVerificationProvider,
    VoiceprintProfile,
)

from .store import VoiceprintStore


class SpeakerVerificationService:
    """Load profiles, invoke provider, and normalize graceful failures."""

    def __init__(
        self,
        *,
        provider: SpeakerVerificationProvider,
        store: VoiceprintStore,
    ) -> None:
        self._provider = provider
        self._store = store

    async def enroll_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        audio_segments: Sequence[bytes],
        sample_rate: int = 16000,
    ) -> VoiceprintProfile:
        profile = await self._provider.enroll(
            tenant_id=tenant_id,
            user_id=user_id,
            audio_segments=audio_segments,
            sample_rate=sample_rate,
        )
        self._store.save_profile(profile)
        return profile

    async def verify_turn(
        self,
        *,
        tenant_id: str,
        user_id: str,
        audio: bytes,
        sample_rate: int = 16000,
        audio_ms: int | None = None,
    ) -> SpeakerSignal:
        start = time.monotonic()
        profiles = self._store.list_user_profiles(tenant_id=tenant_id, user_id=user_id)
        if not profiles:
            return SpeakerSignal.error_signal(
                provider=self._provider.provider,
                model=self._provider.model,
                error="profile_not_found",
                latency_ms=_elapsed_ms(start),
                audio_ms=audio_ms,
            )
        try:
            signal = await self._provider.verify(
                audio=audio,
                sample_rate=sample_rate,
                profiles=profiles,
                audio_ms=audio_ms,
            )
        except Exception as exc:  # noqa: BLE001 - provider failure must degrade
            return SpeakerSignal.error_signal(
                provider=self._provider.provider,
                model=self._provider.model,
                error=f"provider_error:{type(exc).__name__}",
                latency_ms=_elapsed_ms(start),
                audio_ms=audio_ms,
                profile_id=profiles[0].profile_id if profiles else None,
            )
        if signal.latency_ms is None:
            return SpeakerSignal(
                **{**signal.as_timeline_attrs(), "latency_ms": _elapsed_ms(start)}
            )
        return signal


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000.0
