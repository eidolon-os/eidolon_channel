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

"""Configuration dataclass for the FireRedChat pVAD VAD plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FireredPvadConfig:
    """Configuration options for the FireRedChat pVAD VAD plugin.

    Attributes:
        min_speech_duration: Minimum duration of speech to trigger START_OF_SPEECH.
            Default: 0.1s (100ms). Shorter values = faster response, more false triggers.
        min_silence_duration: Silence duration to trigger END_OF_SPEECH after speech ends.
            Default: 0.4s (400ms).
        prefix_padding_duration: Audio context padding added before detected speech onset.
            This captures the speech lead-in to avoid clipping. Default: 0.5s (500ms).
        max_buffered_speech: Maximum speech audio buffered before forcing END_OF_SPEECH.
            Default: 60.0s.
        activation_threshold: Probability threshold [0.0, 1.0] for considering a frame as
            speech. Higher = less sensitive. Default: 0.5.
        sample_rate: Audio sample rate. Only 16000 is supported. Default: 16000.
        force_cpu: Force CPU inference even if CUDA is available. Default: True.
        model_dir: Override path to the model directory containing pvad.onnx and
            spkrec-ecapa-voxceleb/. Defaults to the bundled resources/ directory.
    """

    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.4
    prefix_padding_duration: float = 0.5
    max_buffered_speech: float = 60.0
    activation_threshold: float = 0.5
    sample_rate: Literal[16000] = 16000
    force_cpu: bool = True
    model_dir: Optional[str] = None
