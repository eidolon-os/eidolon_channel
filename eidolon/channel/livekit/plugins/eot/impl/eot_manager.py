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
EOT manager: singleton + factory for creating EOT backend.
Merged from pipeline's eot_manager.py and eot_factory.py.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .eot_backend import EotBackend, OnnxEotBackend

if TYPE_CHECKING:
    from ..config import EidolonEOTConfig


class EotManager:
    """
    EOT detection upper-level entry: delegates to factory-created backend.

    Thread-safe registry keyed on (model_dir, prefer_multilingual) so Chinese
    and Multilingual models can coexist without silently ignoring the second config.
    """

    _instances: dict[tuple, "EotManager"] = {}
    _lock: Final[threading.Lock] = threading.Lock()

    DEFAULT_THRESHOLD: Final[float] = 0.42
    _CACHE_TTL: Final[float] = 0.05  # 50ms — same text within one transcript event cycle

    def __new__(cls, config: "EidolonEOTConfig | None" = None) -> "EotManager":
        key = cls._config_key(config)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    inst._config = config
                    cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, config: "EidolonEOTConfig | None" = None) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._backend: EotBackend = _create_eot_backend(self._config)
            self._cache_text: str = ""
            self._cache_score: float = 0.0
            self._cache_time: float = 0.0
            self._initialized = True

    @classmethod
    def _config_key(cls, config: "EidolonEOTConfig | None") -> tuple:
        if config is None:
            return (None, False)
        return (config.model_dir, config.prefer_multilingual)

    def is_complete(
        self, text: str, threshold: float = DEFAULT_THRESHOLD
    ) -> tuple[bool, float]:
        """
        Determine if text is complete (EOT).

        Args:
            text: Input text.
            threshold: Completeness threshold.

        Returns:
            (is_complete, model_score [0,1])
        """
        score = self.p_complete_score(text)
        return (score >= threshold, score)

    def p_complete_score(self, text: str) -> float:
        """
        Return [0,1] sentence completeness score.

        Uses a short-lived cache (50ms TTL) to avoid redundant ONNX inference
        when the same text is scored multiple times within one transcript event
        cycle (update_asr, should_interrupt, predict_end_of_turn).

        Args:
            text: Input text.

        Returns:
            Completeness score [0,1], empty text returns 0.0.
        """
        if not isinstance(text, str):
            return 0.0

        stripped = text.strip()
        if not stripped:
            return 0.0

        now = time.time()
        if stripped == self._cache_text and (now - self._cache_time) < self._CACHE_TTL:
            return self._cache_score

        score = float(self._backend.score(stripped))
        self._cache_text = stripped
        self._cache_score = score
        self._cache_time = now
        return score


def _resolve_default_model_dir() -> Path:
    """Resolve the default model directory relative to this file."""
    # eot_backend.py is at eidolon/channel/livekit/plugins/eot/impl/eot_backend.py
    # data/ is at eidolon/channel/livekit/plugins/eot/data/
    base = Path(__file__).parent.parent
    return base / "data" / "model" / "firered_chat_turn_detector"


def _create_eot_backend(config: "EidolonEOTConfig | None" = None) -> EotBackend:
    """
    Create an EOT backend.

    Priority:
    1. config.model_dir (if provided)
    2. EIDOLON_EOT_MODEL_DIR env var
    3. Bundled model at data/model/firered_chat_turn_detector/

    Args:
        config: Optional EidolonEOTConfig instance.

    Returns:
        EotBackend instance.
    """
    prefer_multilingual = False
    model_dir: str | Path | None = None

    if config is not None:
        prefer_multilingual = config.prefer_multilingual
        model_dir = config.model_dir

    if model_dir is None:
        model_dir = os.environ.get("EIDOLON_EOT_MODEL_DIR")

    if model_dir is None:
        model_dir = _resolve_default_model_dir()
    else:
        model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"EOT model directory not found: {model_dir}")

    try:
        return OnnxEotBackend(model_dir, prefer_multilingual=prefer_multilingual)
    except Exception as e:
        raise RuntimeError(f"Failed to load EOT backend from {model_dir}: {e}") from e
