"""Speaker verification provider implementations."""

from .modelscope_campplus import ModelScopeCampPlusSpeakerVerificationProvider
from .noop import NoopSpeakerVerificationProvider

__all__ = [
    "ModelScopeCampPlusSpeakerVerificationProvider",
    "NoopSpeakerVerificationProvider",
]
