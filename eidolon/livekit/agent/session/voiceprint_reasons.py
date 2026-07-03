"""Stable voiceprint reason codes used by turn ownership gates."""

from __future__ import annotations

VOICEPRINT_ALLOWED_PREFIX = "voiceprint_allowed"
VOICEPRINT_BLOCKED_PREFIX = "voiceprint_blocked"
VOICEPRINT_ERROR_PREFIX = "voiceprint_error"
VOICEPRINT_INCONCLUSIVE_PREFIX = "voiceprint_inconclusive"
VOICEPRINT_VERIFY_ERROR_FAILOPEN_PREFIX = "verify_error_failopen"

VOICEPRINT_REASON_CACHED_OWNER_CONTEXT = "cached_owner_context"
VOICEPRINT_REASON_OWNER_HIGH_CONFIDENCE = "owner_high_confidence"
VOICEPRINT_REASON_OWNER_KNOWN_SHORT_AUDIO = "owner_known_short_audio"
VOICEPRINT_REASON_SCORE_MISSING = "score_missing"
VOICEPRINT_REASON_SPEAKER_NOT_OWNER = "speaker_not_owner"
VOICEPRINT_REASON_TRUSTED_PAIRED_DEVICE = "trusted_paired_device"

VOICEPRINT_REASON_AUDIO_TOO_SHORT = "audio_too_short"
VOICEPRINT_REASON_INSUFFICIENT_AUDIO = "insufficient_audio"
VOICEPRINT_REASON_TOO_SHORT = "too_short"

VOICEPRINT_CONTEXT_ERROR_PREFIX = "context_error"
VOICEPRINT_OWNER_ABOVE_PROVIDER_THRESHOLD_PREFIX = "owner_above_provider_threshold"

VOICEPRINT_INCONCLUSIVE_REASONS = frozenset(
    {
        VOICEPRINT_REASON_AUDIO_TOO_SHORT,
        VOICEPRINT_REASON_INSUFFICIENT_AUDIO,
        VOICEPRINT_REASON_TOO_SHORT,
    }
)


def voiceprint_allowed_reason(reason: str) -> str:
    return f"{VOICEPRINT_ALLOWED_PREFIX}:{reason}"


def voiceprint_blocked_reason(reason: str) -> str:
    return f"{VOICEPRINT_BLOCKED_PREFIX}:{reason}"


def voiceprint_error_reason(error_type: str) -> str:
    return f"{VOICEPRINT_ERROR_PREFIX}:{error_type}"


def voiceprint_failopen_reason(error: str) -> str:
    return f"{VOICEPRINT_VERIFY_ERROR_FAILOPEN_PREFIX}:{error}"


def voiceprint_inconclusive_reason(reason: str) -> str:
    return f"{VOICEPRINT_INCONCLUSIVE_PREFIX}:{reason}"


def voiceprint_owner_above_provider_threshold_reason(
    *,
    score: float,
    threshold: float,
) -> str:
    return f"{VOICEPRINT_OWNER_ABOVE_PROVIDER_THRESHOLD_PREFIX}:{score:.3f}<{threshold:.3f}"


def is_voiceprint_inconclusive_reason(reason: str) -> bool:
    return reason.lower() in VOICEPRINT_INCONCLUSIVE_REASONS
