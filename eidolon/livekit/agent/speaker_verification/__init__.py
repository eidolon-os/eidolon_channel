"""Speaker verification primitives for observe-only voiceprint support."""

from eidolon.livekit.common.speaker_verification import (
    SpeakerSignal,
    SpeakerVerificationProvider,
    VoiceprintEmbedding,
    VoiceprintProfile,
    default_profile_id,
    default_voiceprint_root,
)
from .service import SpeakerVerificationService
from .store import VoiceprintStore

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
