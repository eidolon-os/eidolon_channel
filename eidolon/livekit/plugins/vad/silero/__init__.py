"""Silero VAD provider: the model the Reachy native app uses, on the Channel state machine."""

from .classifier import MODEL_SHA256, SileroClassifier
from .vad import SileroVAD

__all__ = ["MODEL_SHA256", "SileroClassifier", "SileroVAD"]
