"""Bounded PCM extraction shared by registration and conversation verification."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from functools import cached_property, lru_cache

import numpy as np
from eidolon_sdk.biz.voiceprint import ExtractedSample
from .model_paths import default_campplus_model_dir
from .providers.modelscope_campplus import ModelScopeCampPlusSpeakerVerificationProvider


class InsufficientSpeech(ValueError):
    def __init__(self, message: str, *, code: str = "insufficient_speech"):
        super().__init__(message)
        self.code = code


def validate_speaker_consistency(vectors, threshold=0.65):
    """Reject obvious within-utterance speaker changes, not a spoof detector."""
    for index, a in enumerate(vectors):
        for b in vectors[index + 1 :]:
            similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if not np.isfinite(similarity) or similarity < threshold:
                raise InsufficientSpeech("multiple or inconsistent voices; speak alone", code="inconsistent_speakers")


@lru_cache(maxsize=1)
def shared_voice_extractor():
    """One lazy, capacity-bounded model per worker process, not per session."""
    return VoiceEmbeddingExtractor()


class VoiceEmbeddingExtractor:
    def __init__(self, provider=None):
        self.provider = provider or ModelScopeCampPlusSpeakerVerificationProvider(
            model_dir=os.environ.get("EIDOLON_3DSPEAKER_MODEL_DIR") or default_campplus_model_dir()
        )
        self._lock = threading.Lock()
        self._pending = 0
        self._vad = None

    @cached_property
    def weights_hash(self):
        path = self.provider.model_dir / "campplus_cn_common.bin"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    async def extract(self, audio: bytes, sample_rate=16000):
        if sample_rate != 16000 or len(audio) % 2 or not 48000 <= len(audio) <= 960000:
            raise InsufficientSpeech("record 1.5–30 seconds of mono PCM at 16 kHz", code="invalid_audio_length_or_format")
        if self._pending >= 2:
            raise RuntimeError("voice extraction busy")
        self._pending += 1
        try:
            # Shield keeps capacity reserved until the worker really exits even
            # if the request times out; to_thread cancellation does not stop ML.
            task = asyncio.create_task(asyncio.to_thread(self._extract, audio))
            task.add_done_callback(self._finished)
            return await asyncio.wait_for(asyncio.shield(task), timeout=20)
        except BaseException:
            raise

    def _finished(self, task):
        self._pending -= 1
        if not task.cancelled():
            task.exception()  # consume failures after a caller deadline

    def _extract(self, audio):
        with self._lock:
            samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
            clipping = float(np.mean(np.abs(samples) >= 0.995))
            if clipping > 0.02:
                raise InsufficientSpeech("audio is clipped; move away from microphone", code="audio_clipped")
            if self._vad is None:
                from eidolon.livekit.plugins.vad.firered.processor import PvadProcessor

                self._vad = PvadProcessor(force_cpu=True)
            self._vad.reset()
            size = self._vad.window_size_samples
            voiced = []
            for offset in range(0, len(samples) - size + 1, size):
                chunk = samples[offset : offset + size]
                if self._vad(chunk) >= 0.5:
                    voiced.append(chunk)
            speech_ms = len(voiced) * size * 1000 // 16000
            if speech_ms < 1500:
                raise InsufficientSpeech("not enough clear speech; speak a longer sentence")
            speech = np.concatenate(voiced)
            # Disjoint windows cover the entire voiced utterance, including its
            # tail. A short tail is merged into the preceding window. This is a
            # conservative quality gate; simultaneous overlapping voices still
            # require an independently evaluated overlap detector.
            sections = np.array_split(speech, max(1, len(speech) // 32000))
            if len(sections) > 1:
                vectors = [
                    self.provider._extract_embedding_from_audio(
                        (part * 32768).astype("<i2").tobytes(), 16000
                    )
                    for part in sections
                ]
                validate_speaker_consistency(vectors)
            pcm = (speech * 32768).astype("<i2").tobytes()
            vector = self.provider._extract_embedding_from_audio(pcm, 16000)
            return ExtractedSample(
                model_id=self.provider.model,
                weights_hash=self.weights_hash,
                preprocessing_version="firered-0.5-campplus-fbank-v1",
                vector=vector.tolist(),
                speech_ms=speech_ms,
                quality={"clipping": clipping},
            )
