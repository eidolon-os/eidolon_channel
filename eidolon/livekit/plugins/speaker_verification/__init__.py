"""Speaker verification provider plugins."""

from .model_paths import default_campplus_model_dir
from .providers import (
    ModelScopeCampPlusSpeakerVerificationProvider,
    NoopSpeakerVerificationProvider,
)

__all__ = [
    "ModelScopeCampPlusSpeakerVerificationProvider",
    "NoopSpeakerVerificationProvider",
    "default_campplus_model_dir",
]
