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
Eidolon EOT plugin for LiveKit Agents.

============================================================================
ARCHITECTURE DECISION (ADR — why this isn't using framework primitives)
============================================================================

The framework's ``turn_detection`` parameter accepts:
  * ``"stt"`` / ``"vad"`` / ``"realtime_llm"`` — built-in modes
  * ``"manual"`` — caller drives ``commit_user_turn``
  * a custom ``_TurnDetector`` instance with
    ``predict_end_of_turn(chat_ctx, timeout) -> float``

This plugin IS a custom ``_TurnDetector`` (the right extension shape).
What it adds beyond a plain probability score:

1. **PolicyChain** — provider-neutral fusion of learned EOT probability,
   VAD, STT finality, timing and streaming duplicate guards. Transcript
   wording is evaluated only by the learned model.

2. **Soft/hard interrupt staging** — soft pause first, hard cut only
   if user keeps talking. Framework has
   ``InterruptionOptions.false_interruption_timeout`` for similar
   intent, but it's a single timer; our soft-interrupt is a state
   machine that can be cancelled by silence.

3. **Two-path scoring** — same model serves
   ``predict_end_of_turn`` (framework's endpointing) and
   ``should_interrupt`` (our PolicyChain decision), with shared cache
   and slightly different post-processing.

When framework's interrupt path is active alongside this plugin,
results conflict. We disable framework auto-interrupt with the public
``AgentSession.turn_handling.interruption.enabled = False`` config, while
keeping this plugin as the EOT endpointing model.

============================================================================

Provides end-of-turn detection using the FireRed Chat Turn Detector model.

Usage::

    from eidolon.livekit.plugins.eot import ChineseModel

    session = AgentSession(
        turn_detection=ChineseModel(
            model_dir="/path/to/model",        # Override default bundled model
            prefer_multilingual=False,         # True for multilingual model
            max_threshold=2.2,
        ),
    )

Available models:
    - ChineseModel: Chinese-only model (chinese_best_model_q8.onnx)
    - MultilingualModel: Multilingual model (multilingual_best_model_q8.onnx)
"""

from __future__ import annotations

from .config import EidolonEOTConfig
from .models.base import EidolonEOTModel
from .models.chinese import ChineseModel
from .models.multilingual import MultilingualModel
from .version import __version__

# Re-export impl classes for testing without triggering the livekit.agent import chain.
# All tests should import from 'eidolon.livekit.plugins.eot' (this package)
# rather than 'eidolon.livekit.plugins.eot.impl' which goes through
# plugins/__init__.py and crashes on psutil.
from .impl.eot_policy import (
    PolicyChain,
    CutDecision,
    CutPolicy,
    InterruptCooldownPolicy,
    MinIntervalPolicy,
    DuplicateTextPolicy,
    MinSpeakingDurationPolicy,
    VADStabilityPolicy,
    ASRStabilityPolicy,
    MaxDurationPolicy,
    VADStalePolicy,
    ASRFinalCutPolicy,
    EOTScorePolicy,
    EOTScoreSemanticPolicy,
)
from .impl.state import (
    TurnDetectionStateManager,
    VADState,
    ASRState,
    SentenceState,
    InterruptState,
    NoiseState,
)
from .impl.turn_end_policy import TurnEndPolicy
from .impl.context_enhanced_eot import ContextEnhancedEot
from .impl.utils import (
    longest_common_prefix_len,
    extract_incremental_text,
    is_similar_text,
    compute_text_hash,
    format_duration,
)

__all__ = [
    "EidolonEOTConfig",
    "EidolonEOTModel",
    "ChineseModel",
    "MultilingualModel",
    "register_plugin",
    "__version__",
    # impl classes
    "PolicyChain",
    "CutDecision",
    "CutPolicy",
    "InterruptCooldownPolicy",
    "MinIntervalPolicy",
    "DuplicateTextPolicy",
    "MinSpeakingDurationPolicy",
    "VADStabilityPolicy",
    "ASRStabilityPolicy",
    "MaxDurationPolicy",
    "VADStalePolicy",
    "ASRFinalCutPolicy",
    "EOTScorePolicy",
    "EOTScoreSemanticPolicy",
    "TurnDetectionStateManager",
    "VADState",
    "ASRState",
    "SentenceState",
    "InterruptState",
    "NoiseState",
    "TurnEndPolicy",
    "ContextEnhancedEot",
    "longest_common_prefix_len",
    "extract_incremental_text",
    "is_similar_text",
    "compute_text_hash",
    "format_duration",
]


def __getattr__(name: str):
    # Lazy-load the plugin wrapper class to avoid registration on import.
    # Use register_plugin() from the main thread instead.
    if name == "EidolonEOTPlugin":
        from ._plugin import EidolonEOTPlugin

        return EidolonEOTPlugin

    if name == "register_plugin":
        from ._plugin import register_plugin

        return register_plugin

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
