"""No-op provider used before a real voiceprint model is configured."""

from __future__ import annotations

from collections.abc import Sequence

from eidolon.livekit.common.speaker_verification import (
    SpeakerSignal,
    VoiceprintProfile,
    default_profile_id,
)


class NoopSpeakerVerificationProvider:
    """Creates placeholder profiles and always degrades on verification."""

    provider = "noop"
    model = "noop"

    async def enroll(
        self,
        *,
        tenant_id: str,
        user_id: str,
        audio_segments: Sequence[bytes],
        sample_rate: int,
    ) -> VoiceprintProfile:
        duration_ms = _duration_ms(audio_segments, sample_rate=sample_rate)
        return VoiceprintProfile(
            profile_id=default_profile_id(user_id),
            tenant_id=tenant_id,
            user_id=user_id,
            provider=self.provider,
            model=self.model,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            quality={
                "accepted_segments": len([s for s in audio_segments if s]),
                "rejected_segments": len([s for s in audio_segments if not s]),
            },
            metadata={"status": "placeholder"},
        )

    async def verify(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        profiles: Sequence[VoiceprintProfile],
        audio_ms: int | None = None,
    ) -> SpeakerSignal:
        profile = profiles[0] if profiles else None
        return SpeakerSignal.error_signal(
            provider=self.provider,
            model=self.model,
            error="provider_disabled",
            audio_ms=audio_ms,
            profile_id=profile.profile_id if profile is not None else None,
        )


def _duration_ms(audio_segments: Sequence[bytes], *, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    # 16-bit mono PCM duration. WAV containers are accepted by future real
    # providers, but noop intentionally treats bytes as opaque-ish test input.
    total_samples = sum(len(segment) // 2 for segment in audio_segments)
    return int(total_samples * 1000 / sample_rate)
