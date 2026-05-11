# Copyright 2025 Eidolon Team
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

"""
Multilingual EOT model.

Uses the multilingual_best_model_q8.onnx model for mixed-language text.
"""

from ..config import EidolonEOTConfig
from .base import EidolonEOTModel


class MultilingualModel(EidolonEOTModel):
    """
    Multilingual EOT model.

    Uses the FireRed Chat Turn Detector multilingual model
    (multilingual_best_model_q8.onnx) which supports Chinese and English.

    Usage::

        from eidolon.channel.livekit.plugins.eot import MultilingualModel

        session = AgentSession(turn_detection=MultilingualModel())
    """

    def __init__(self, **kwargs: object) -> None:
        config = EidolonEOTConfig(prefer_multilingual=True, **kwargs)
        super().__init__(config)
