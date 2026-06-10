"""Speaker verification primitives for observe-only voiceprint support."""

from .base import SpeakerVerificationProvider
from .service import SpeakerVerificationService
from .signal import SpeakerSignal
from .store import (
    VoiceprintEmbedding,
    VoiceprintProfile,
    VoiceprintStore,
    default_profile_id,
    default_voiceprint_root,
)

__all__ = [
    "SpeakerSignal",
    "SpeakerVerificationProvider",
    "SpeakerVerificationService",
    "VoiceprintProfile",
    "VoiceprintEmbedding",
    "VoiceprintStore",
    "default_profile_id",
    "default_voiceprint_root",
]
