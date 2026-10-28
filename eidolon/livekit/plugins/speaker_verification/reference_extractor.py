"""Continuous PCM embedding with separate signal-quality checks.

Runs only inside the serialized inference executor. It cannot enroll a person,
select a device template, authorize a Companion or retain audio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from eidolon_sdk.biz.voiceprint import ExtractedSample, SpeakerAudioQuality

from .providers.modelscope_campplus import ModelScopeCampPlusSpeakerVerificationProvider


class InvalidSpeakerAudio(ValueError):
    """A machine-readable signal-quality failure, never an identity rejection."""

    def __init__(self, reason, quality=None):
        super().__init__(reason)
        self.quality = quality


class ReferenceSpeakerExtractor:
    def __init__(self, model_dir, *, provider=None, vad_factory=None):
        self.model_dir = Path(model_dir).resolve()
        self._provider = provider
        self._provider_override = provider
        self._vad_factory = vad_factory
        self._vad = None
        self._weights_hash = None

    @property
    def model_identity(self):
        if self._weights_hash is None:
            return None
        return (self._provider.model, self._weights_hash, 'reference-contiguous-pcm16-v1')

    def warm_up(self):
        self._weights_hash = None
        self._vad = None
        manifest = json.loads((self.model_dir / 'manifest.json').read_text())
        alias = manifest.get('model_alias', 'campplus_zh_16k_common')
        if alias == 'campplus_zh_16k_common':
            required = {'campplus_cn_common.bin', 'config.yaml', 'configuration.json'}
        elif alias == 'ecapa_voxceleb_16k':
            required = {'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt',
                        'label_encoder.txt', 'hyperparams.yaml'}
        else:
            raise ValueError('unsupported_speaker_model')
        checked = {}
        for item in manifest['artifacts']:
            path = (self.model_dir / item['path']).resolve()
            if not path.is_relative_to(self.model_dir) or not path.is_file():
                raise ValueError('artifact_integrity')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != item['size_bytes'] or digest != item['sha256']:
                raise ValueError('artifact_integrity')
            checked[item['path']] = digest
        if not required <= checked.keys():
            raise ValueError('artifact_integrity')
        # A successful prior warm-up may have cached an older set of weights.
        # Rechecking files must never relabel that old object with a new digest.
        self._provider = self._provider_override
        if self._provider is None:
            if alias == 'campplus_zh_16k_common':
                self._provider = ModelScopeCampPlusSpeakerVerificationProvider(
                    model_dir=self.model_dir,
                )
            else:
                from .providers.speechbrain_ecapa import SpeechBrainEcapaProvider
                self._provider = SpeechBrainEcapaProvider(model_dir=self.model_dir)
        if self._provider.model != alias:
            raise ValueError('speaker_model_mismatch')
        # ECAPA identity covers all inference assets, including feature configuration.
        weights_hash = (checked['campplus_cn_common.bin'] if alias == 'campplus_zh_16k_common'
                        else hashlib.sha256(json.dumps(
                            {name: checked[name] for name in sorted(required)},
                            sort_keys=True, separators=(',', ':'),
                        ).encode()).hexdigest())
        if self._vad_factory is None:
            from eidolon.livekit.plugins.vad.firered.processor import PvadProcessor
            self._vad = PvadProcessor(force_cpu=True)
        else:
            self._vad = self._vad_factory()
        # Fixed non-personal probe exercises actual inference before reporting ready.
        probe = (np.sin(np.arange(24000) * (2 * np.pi * 220 / 16000)) * 1000).astype('<i2')
        # PvadProcessor constructs its ONNX session lazily too. Readiness must
        # cover the quality path, so first enrollment never pays that startup.
        probability = self._vad(probe[:self._vad.window_size_samples].astype(np.float32) / 32768)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('invalid_quality_output')
        self._vad.reset()
        self._provider.extract_embedding(probe.tobytes(), 16000)
        self._weights_hash = weights_hash

    def extract(self, pcm: bytes, min_speech_ms: int = 1500) -> ExtractedSample:
        if self._vad is None or self._weights_hash is None:
            raise RuntimeError('model_not_ready')
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > 960000:
            raise InvalidSpeakerAudio('invalid_pcm')
        if not 300 <= min_speech_ms <= 30000:
            raise ValueError('invalid_speech_floor')
        quality = SpeakerAudioQuality(
            audio_ms=len(pcm) // 32, required_speech_ms=min_speech_ms,
        )
        if len(pcm) < min_speech_ms * 32:
            raise InvalidSpeakerAudio('audio_too_short', quality)
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        quality.rms = float(np.sqrt(np.mean(samples*samples)))
        if not np.any(samples):
            quality.speech_ms = 0
            raise InvalidSpeakerAudio('digital_silence', quality)
        clipping = float(np.mean(np.abs(samples) >= .995))
        quality.clipping = clipping
        if clipping > .02:
            raise InvalidSpeakerAudio('audio_clipped', quality)
        self._vad.reset()
        size = self._vad.window_size_samples
        speech_samples = sum(
            size for i in range(0, len(samples) - size + 1, size)
            if self._vad(samples[i:i + size]) >= .5
        )
        speech_ms = speech_samples * 1000 // 16000
        quality.speech_ms = speech_ms
        if speech_ms < min_speech_ms:
            raise InvalidSpeakerAudio('speech_too_short', quality)
        # VAD measures quality only. Never delete gaps, split the embedding into
        # 2-second chunks, or compare a quality score with an identity threshold.
        vector = self._provider.extract_embedding(pcm, 16000)
        return ExtractedSample(
            model_id=self._provider.model,
            weights_hash=self._weights_hash,
            preprocessing_version='reference-contiguous-pcm16-v1',
            vector=vector.tolist(),
            speech_ms=speech_ms,
            quality=quality.model_dump(exclude_none=True),
        )
