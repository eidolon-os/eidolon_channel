"""Speaker verification provider plugins."""

from .model_paths import default_campplus_model_dir
from .providers import (
    ModelScopeCampPlusSpeakerVerificationProvider,
)

__all__ = [
    "ModelScopeCampPlusSpeakerVerificationProvider",
    "default_campplus_model_dir",
]
