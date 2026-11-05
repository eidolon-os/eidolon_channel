"""Silero VAD window classifier (same model and inference as the Reachy native app)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

MODEL_FILENAME = "silero_vad_16k_op15.onnx"
MODEL_SHA256 = "7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49"
DEFAULT_MODEL_PATH = Path(__file__).parent / "resources" / MODEL_FILENAME

_STATE_SHAPE = (2, 1, 128)
_CONTEXT_SAMPLES = 64


class SileroClassifier:
    """Recurrent per-stream state over one shared, read-only ONNX session.

    Each call consumes one 512-sample (32 ms) window prefixed with the last 64
    samples of the previous window, exactly as upstream ``silero_vad`` does.
    """

    sample_rate = 16000
    window_size_samples = 512

    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self._sr = np.array(self.sample_rate, dtype=np.int64)
        self.reset()

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> SileroClassifier:
        import onnxruntime as ort

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        data = path.read_bytes()
        if model_path is None and hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            # A git-lfs pointer or truncated copy must fail loudly, not run as noise.
            raise ValueError(f"Silero VAD model checksum mismatch: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            data, sess_options=options, providers=["CPUExecutionProvider"]
        )
        return cls(session)

    def new_stream(self) -> SileroClassifier:
        return SileroClassifier(self._session)

    def reset(self) -> None:
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        chunk = np.asarray(window, dtype=np.float32).reshape(1, -1)
        if chunk.shape[1] != self.window_size_samples:
            raise ValueError("Silero VAD expects 512-sample windows at 16 kHz")
        probability, state = self._session.run(
            None,
            {
                "input": np.concatenate((self._context, chunk), axis=1),
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._state = np.asarray(state, dtype=np.float32)
        self._context = chunk[:, -_CONTEXT_SAMPLES:].copy()
        return float(np.asarray(probability).item())
