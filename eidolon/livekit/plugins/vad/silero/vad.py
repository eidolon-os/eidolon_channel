"""Silero provider for the shared window-classifier VAD."""

from __future__ import annotations

import functools
import logging

from ..streaming import StreamingVAD, StreamingVADOptions
from .classifier import SileroClassifier

logger = logging.getLogger("eidolon.livekit.plugins.vad.silero")


@functools.cache
def _shared_classifier(model_path: str | None) -> SileroClassifier:
    # The worker prewarm and every session factory build through load(); keyed
    # by path so a second model file is never served the first one's session.
    classifier = SileroClassifier.load(model_path)
    logger.info("Silero VAD ONNX model loaded model=%s", model_path or "bundled")
    return classifier


class SileroVAD(StreamingVAD):
    """Silero VAD under the same state machine and pipeline hooks as FireRed."""

    logger = logger
    # Light smoothing, as livekit-plugins-silero: Silero's own recurrent state
    # already smooths; heavier filtering only delays barge-in onset.
    smoothing_alpha = 0.35

    @classmethod
    def load(cls, *, model_path: str | None = None, **options: float) -> SileroVAD:
        """Build a VAD over the process-wide Silero model.

        ``options`` are :class:`StreamingVADOptions` fields.
        """
        return cls(processor=_shared_classifier(model_path), opts=StreamingVADOptions(**options))
