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
EOT ONNX backend using the FireRed Chat Turn Detector model.

Uses the same model as the official FireRed plugin:
    - chinese_best_model_q8.onnx
    - multilingual_best_model_q8.onnx
    - tokenizer/

Reference: https://huggingface.co/FireRedTeam/FireRedChat-turn-detector
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..log import logger

MAX_HISTORY_TOKENS = 128


def _softmax_last(logits: np.ndarray) -> float:
    """Apply softmax on the last dimension and return the EOU probability."""
    if logits.size == 0:
        return 0.0
    x = np.asarray(logits, dtype=np.float64)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return float(probs.flatten()[-1])


class EotBackend:
    """
    Abstract EOT backend: input text, output [0,1] completeness score.
    Subclasses must implement score(text).
    """

    def score(self, text: str) -> float:
        raise NotImplementedError


class OnnxEotBackend(EotBackend):
    """
    FireRed Chat Turn Detector backend.

    Uses the ONNX model to compute end-of-utterance probability for input text.
    Supports both Chinese and multilingual models.

    Args:
        model_dir: Path to the model directory containing the ONNX file and tokenizer.
        prefer_multilingual: If True, use multilingual_best_model_q8.onnx;
            otherwise use chinese_best_model_q8.onnx.
    """

    def __init__(self, model_dir: str | Path, prefer_multilingual: bool = False):
        self._model_dir = Path(model_dir)
        self._prefer_multilingual = prefer_multilingual
        self._session = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        """Load ONNX session and tokenizer lazily."""
        if self._prefer_multilingual:
            onnx_path = self._model_dir / "multilingual_best_model_q8.onnx"
        else:
            onnx_path = self._model_dir / "chinese_best_model_q8.onnx"
        tokenizer_path = self._model_dir / "tokenizer"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            logger.info(
                f"OnnxEotBackend: loaded {onnx_path.name} "
                f"with providers {self._session.get_providers()}"
            )
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for OnnxEotBackend. "
                "Install it with: pip install onnxruntime"
            ) from e

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                truncation_side="left",
            )
            logger.info(f"OnnxEotBackend: loaded tokenizer from {tokenizer_path}")
        except ImportError as e:
            raise ImportError(
                "transformers is required for OnnxEotBackend. "
                "Install it with: pip install transformers"
            ) from e

    def score(self, text: str) -> float:
        """
        Compute EOT probability for the given text.

        Args:
            text: Input text (may contain punctuation).

        Returns:
            EOU probability in [0, 1]. Higher means more likely to be end of utterance.
        """
        if not text or not text.strip():
            return 0.0

        try:
            # Step 1: Clean text - remove punctuation
            cleaned = re.sub(r"[，。？！,. ?!「」『』（）【】\/\\\]\[\{\}\"']", "", text)

            # Step 2: Tokenize (truncate from left, keep latest tokens)
            inputs = self._tokenizer(
                cleaned,
                return_tensors="np",
                truncation=True,
                max_length=MAX_HISTORY_TOKENS,
                add_special_tokens=True,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Step 3: ONNX inference
            logits = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                },
            )[0]

            # Step 4: Softmax on last dimension
            return _softmax_last(logits)

        except Exception as e:
            logger.warning(f"OnnxEotBackend.score failed for '{text}': {e}")
            return 0.0
