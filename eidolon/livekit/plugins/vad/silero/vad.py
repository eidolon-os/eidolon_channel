"""Silero provider for the shared window-classifier VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from ..streaming import StreamingVAD
from .classifier import SileroClassifier

logger = logging.getLogger("eidolon.livekit.plugins.vad.silero")


@dataclass
class SileroVadConfig:
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.5
    prefix_padding_duration: float = 0.3
    max_buffered_speech: float = 60.0
    activation_threshold: float = 0.5
    sample_rate: Literal[16000] = 16000


class SileroVAD(StreamingVAD):
    """Silero VAD under the same state machine and pipeline hooks as FireRed."""

    logger = logger
    # Light smoothing, as livekit-plugins-silero: Silero's own recurrent state
    # already smooths; heavier filtering only delays barge-in onset.
    smoothing_alpha = 0.35

    _classifier: SileroClassifier | None = None

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.5,
        prefix_padding_duration: float = 0.3,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[16000] = 16000,
        model_path: str | None = None,
    ) -> SileroVAD:
        if sample_rate != 16000:
            raise ValueError("Silero VAD is configured for 16 kHz only")
        if cls._classifier is None:
            cls._classifier = SileroClassifier.load(model_path)
            logger.info("Silero VAD ONNX model preloaded")
        opts = SileroVadConfig(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
        )
        return cls(processor=cls._classifier, opts=opts)
