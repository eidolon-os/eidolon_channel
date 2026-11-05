# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FireRedChat pVAD provider for the shared window-classifier VAD.

The speech/silence state machine lives in :mod:`eidolon.livekit.plugins.vad.streaming`;
this module only owns the FireRed processor and its speaker conditioning.

Provider notes (vs upstream livekit-plugins-fireredchat-pvad):
- ONNX session is eagerly loaded when VAD.load() is called (no lazy loading).
- Multiple streams share a single PvadProcessor (model weights are read-only;
  only streaming state buffers are per-stream).
- ExpFilter alpha 0.5 (original: 0.8).
- update_speaker() must be called explicitly by the caller (no implicit call on first frame).
- min_speech_duration defaults to 0.1s (100ms) for faster speech onset detection.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ..streaming import StreamingVAD, StreamingVADStream
from .config import FireredPvadConfig
from .log import logger
from .processor import PvadProcessor


class VADStream(StreamingVADStream):
    """A stream whose PvadProcessor also carries a speaker conditioning vector.

    Each stream has its own state buffers (mel_buffer, gru_buffer, spkemb) but
    shares the ONNX session; update_speaker() updates only this stream.
    """

    def update_speaker(self, audio_16k_samples: Optional[np.ndarray] = None) -> None:
        """Update the speaker embedding for adaptive VAD.

        After the first user utterance, call this with ~1–5 seconds of audio
        from the primary speaker to improve VAD accuracy for that speaker.

        Args:
            audio_16k_samples: Audio as a float32 numpy array at 16 kHz.
                Values should be in [-1, 1]. If None, the method is a no-op.
                Recommended length: 1–5 seconds.
        """
        if audio_16k_samples is not None and audio_16k_samples.size > 0:
            self._processor.update_speaker_embedding(audio_16k_samples)
            logger.info("VADStream: speaker embedding updated")


class VAD(StreamingVAD):
    """FireRedChat pVAD VAD implementation.

    Use ``VAD.load()`` to construct an instance, then call ``vad.stream()`` to
    create a ``VADStream`` for processing audio.
    """

    logger = logger
    smoothing_alpha = 0.5

    # Resource owner only. Each stream receives fresh recurrent/identity state.
    _processor: PvadProcessor | None = None

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.4,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[16000] = 16000,
        force_cpu: bool = True,
        model_dir: str | None = None,
    ) -> VAD:
        """Load and initialize the FireRedChat pVAD model.

        Args:
            min_speech_duration: Minimum speech duration to trigger START_OF_SPEECH.
            min_silence_duration: Silence duration to trigger END_OF_SPEECH.
            prefix_padding_duration: Audio padding before detected speech onset.
            max_buffered_speech: Max speech audio buffered before forcing END_OF_SPEECH.
            activation_threshold: Probability threshold [0.0, 1.0].
            sample_rate: Audio sample rate. Only 16000 is supported.
            force_cpu: Force CPU inference.
            model_dir: Override path to the model directory. Supports
                EIDOLON_FIRERED_PVAD_MODEL_DIR env var.
        """
        if sample_rate != 16000:
            raise ValueError("FireRedChat pVAD only supports 16 kHz sample rates")

        # Created once and reused across all instances. Load the ONNX model now,
        # not on the first frame, to avoid a ~4s cold start in the first turn.
        if cls._processor is None:
            cls._processor = PvadProcessor(
                model_dir=model_dir,
                force_cpu=force_cpu,
            )
            logger.info("FireRedChat pVAD shared processor initialized")
            _ = cls._processor.onnx_session
            logger.info("FireRedChat pVAD ONNX model preloaded")

        opts = FireredPvadConfig(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
            force_cpu=force_cpu,
            model_dir=model_dir,
        )
        return cls(processor=cls._processor, opts=opts)

    def _new_stream(self, *, notify_pipeline: bool) -> VADStream:
        return VADStream(
            vad=self, processor=self._processor.new_stream(), opts=self._opts,
            notify_pipeline=notify_pipeline,
        )
