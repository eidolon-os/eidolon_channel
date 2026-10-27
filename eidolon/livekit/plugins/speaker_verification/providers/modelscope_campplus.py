"""ModelScope CAMPPlus numerical embedding adapter; no identity or persistence."""

from pathlib import Path
from typing import Any


class ModelScopeCampPlusSpeakerVerificationProvider:
    model = "campplus_zh_16k_common"

    def __init__(self, *, model_dir: str | Path):
        self.model_dir = Path(model_dir).expanduser()
        self._pipeline: Any | None = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            if not self.model_dir.is_dir():
                raise RuntimeError(f"3D-Speaker model dir not found: {self.model_dir}")
            from modelscope.pipelines import pipeline

            self._pipeline = pipeline(
                task="speaker-verification",
                model=str(self.model_dir),
            )
        return self._pipeline

    def extract_embedding(self, pcm: bytes, sample_rate: int):
        """Reference preprocessing for explicit PCM, without temporary WAV files.

        Public numerical adapter used by offline evaluation. It does not read
        profiles or make an identity/admission decision.
        """
        import numpy as np

        if sample_rate != 16000 or not pcm or len(pcm) % 2:
            raise ValueError("mono PCM16 at 16 kHz required")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        result = self._load_pipeline()([samples], output_emb=True)
        vector = _normalize(result["embs"][0])
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("invalid embedding")
        return vector


def _normalize(vector):
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm <= 0:
        raise RuntimeError("voiceprint embedding has zero norm")
    return array / norm
