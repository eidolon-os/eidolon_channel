"""Continuous PCM embedding with separate signal-quality checks.

Runs only inside the serialized inference executor. It cannot enroll a person,
select a device template, authorize a Companion or retain audio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from eidolon_sdk.biz.voiceprint import ExtractedSample

from .providers.modelscope_campplus import ModelScopeCampPlusSpeakerVerificationProvider


class InvalidSpeakerAudio(ValueError):
    """A machine-readable signal-quality failure, never an identity rejection."""


class ReferenceSpeakerExtractor:
    def __init__(self, model_dir, *, provider=None, vad_factory=None):
        self.model_dir = Path(model_dir).resolve()
        self._provider = provider or ModelScopeCampPlusSpeakerVerificationProvider(
            model_dir=self.model_dir,
        )
        self._vad_factory = vad_factory
        self._vad = None
        self._weights_hash = None

    def warm_up(self):
        manifest = json.loads((self.model_dir / 'manifest.json').read_text())
        checked = {}
        for item in manifest['artifacts']:
            path = (self.model_dir / item['path']).resolve()
            if not path.is_relative_to(self.model_dir) or not path.is_file():
                raise ValueError('artifact_integrity')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != item['size_bytes'] or digest != item['sha256']:
                raise ValueError('artifact_integrity')
            checked[item['path']] = digest
        if not {'campplus_cn_common.bin', 'config.yaml', 'configuration.json'} <= checked.keys():
            raise ValueError('artifact_integrity')
        self._weights_hash = checked['campplus_cn_common.bin']
        if self._vad_factory is None:
            from eidolon.livekit.plugins.vad.firered.processor import PvadProcessor
            self._vad = PvadProcessor(force_cpu=True)
        else:
            self._vad = self._vad_factory()
        # Fixed non-personal probe exercises actual inference before reporting ready.
        probe = (np.sin(np.arange(24000) * (2 * np.pi * 220 / 16000)) * 1000).astype('<i2')
        self._provider.extract_embedding(probe.tobytes(), 16000)

    def extract(self, pcm: bytes) -> ExtractedSample:
        if self._vad is None or self._weights_hash is None:
            raise RuntimeError('model_not_ready')
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > 960000:
            raise InvalidSpeakerAudio('invalid_pcm')
        if len(pcm) < 48000:
            raise InvalidSpeakerAudio('audio_too_short')
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        if not np.any(samples):
            raise InvalidSpeakerAudio('digital_silence')
        clipping = float(np.mean(np.abs(samples) >= .995))
        if clipping > .02:
            raise InvalidSpeakerAudio('audio_clipped')
        self._vad.reset()
        size = self._vad.window_size_samples
        speech_samples = sum(
            size for i in range(0, len(samples) - size + 1, size)
            if self._vad(samples[i:i + size]) >= .5
        )
        speech_ms = speech_samples * 1000 // 16000
        if speech_ms < 1500:
            raise InvalidSpeakerAudio('speech_too_short')
        # VAD measures quality only. Never delete gaps, split the embedding into
        # 2-second chunks, or compare a quality score with an identity threshold.
        vector = self._provider.extract_embedding(pcm, 16000)
        return ExtractedSample(
            model_id=self._provider.model,
            weights_hash=self._weights_hash,
            preprocessing_version='reference-contiguous-pcm16-v1',
            vector=vector.tolist(),
            speech_ms=speech_ms,
            quality=dict(clipping=clipping, rms=float(np.sqrt(np.mean(samples*samples)))),
        )
