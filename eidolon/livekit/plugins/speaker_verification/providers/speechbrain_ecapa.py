"""Strict numerical adapter over the existing ECAPA extractor; no identity grants."""

import numpy as np


class SpeechBrainEcapaProvider:
    model = "ecapa_voxceleb_16k"

    def __init__(self, *, model_dir, extractor_factory=None):
        self._model_dir = model_dir
        self._factory = extractor_factory
        self._extractor = None

    def extract_embedding(self, pcm, sample_rate):
        if sample_rate != 16000 or not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("invalid_ecapa_pcm")
        if self._extractor is None:
            factory = self._factory
            if factory is None:
                from ...vad.firered.processor import SpeakerEmbExtractor
                factory = SpeakerEmbExtractor
            self._extractor = factory(self._model_dir)
        # pVAD permits zero conditioning as a fallback; identity verification must not.
        if not self._extractor.is_loaded:
            raise RuntimeError("ecapa_model_unavailable")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        vector = np.asarray(self._extractor.get_embedding(samples)).reshape(-1)
        norm = np.linalg.norm(vector)
        if vector.shape != (192,) or not np.all(np.isfinite(vector)) or not .99 <= norm <= 1.01:
            raise RuntimeError("invalid_ecapa_embedding")
        return vector / norm
