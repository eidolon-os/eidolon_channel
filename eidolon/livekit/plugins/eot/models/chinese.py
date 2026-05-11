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
Chinese EOT model.

Uses the chinese_best_model_q8.onnx model for Chinese-only text.
"""

from ..config import EidolonEOTConfig
from .base import EidolonEOTModel


class ChineseModel(EidolonEOTModel):
    """
    Chinese-specific EOT model.

    Uses the FireRed Chat Turn Detector Chinese model (chinese_best_model_q8.onnx).

    Usage::

        from eidolon.livekit.plugins.eot import ChineseModel

        session = AgentSession(turn_detection=ChineseModel())
    """

    def __init__(self, **kwargs: object) -> None:
        config = EidolonEOTConfig(prefer_multilingual=False, **kwargs)
        super().__init__(config)
