import hashlib
import json

import numpy as np
import pytest

from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    InvalidSpeakerAudio, ReferenceSpeakerExtractor,
)


class Provider:
    model = 'campplus_zh_16k_common'

    def __init__(self):
        self.received = []

    def extract_embedding(self, pcm, sample_rate):
        assert sample_rate == 16000
        self.received.append(pcm)
        return np.array([1., 0.])


class Vad:
    window_size_samples = 160

    def reset(self):
        pass

    def __call__(self, samples):
        return float(np.any(samples))


def engine(tmp_path):
    artifacts = []
    for name in ('campplus_cn_common.bin', 'config.yaml', 'configuration.json'):
        data = b'fixture'
        (tmp_path / name).write_bytes(data)
        artifacts.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)))
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(artifacts=artifacts)))
    provider = Provider()
    return ReferenceSpeakerExtractor(tmp_path, provider=provider, vad_factory=Vad), provider


def test_quality_vad_never_splices_the_embedding_input(tmp_path):
    extractor, provider = engine(tmp_path)
    extractor.warm_up()
    pcm = b'\x01\x00' * 16000 + b'\x00\x00' * 8000 + b'\x02\x00' * 16000
    sample = extractor.extract(pcm)
    assert provider.received[-1] == pcm
    assert sample.speech_ms == 2000
    assert sample.preprocessing_version == 'reference-contiguous-pcm16-v1'


@pytest.mark.parametrize('pcm,reason', [
    (b'\x00\x00' * 32000, 'digital_silence'),
    (b'\x01\x00' * 8000, 'audio_too_short'),
    (b'\xff\x7f' * 32000, 'audio_clipped'),
    (b'\x01\x00' * 8000 + b'\x00\x00' * 24000, 'speech_too_short'),
    (b'\x01', 'invalid_pcm'),
])
def test_insufficient_audio_never_reaches_embedding_model(tmp_path, pcm, reason):
    extractor, provider = engine(tmp_path)
    extractor.warm_up()
    count = len(provider.received)
    with pytest.raises(InvalidSpeakerAudio, match=reason):
        extractor.extract(pcm)
    assert len(provider.received) == count


def test_corrupt_asset_fails_before_loading_model(tmp_path):
    extractor, provider = engine(tmp_path)
    (tmp_path / 'campplus_cn_common.bin').write_bytes(b'bad')
    with pytest.raises(ValueError, match='artifact_integrity'):
        extractor.warm_up()
    assert not provider.received


def test_lazy_quality_model_failure_prevents_ready(tmp_path):
    extractor, _ = engine(tmp_path)

    class BrokenVad(Vad):
        def __call__(self, samples):
            raise RuntimeError('quality_model_missing')

    extractor._vad_factory = BrokenVad
    with pytest.raises(RuntimeError, match='quality_model_missing'):
        extractor.warm_up()


def test_short_conversation_probe_uses_policy_without_weakening_enrollment(tmp_path):
    extractor, provider = engine(tmp_path)
    extractor.warm_up()
    pcm = b'\x01\x00' * 16000
    with pytest.raises(InvalidSpeakerAudio, match='audio_too_short'):
        extractor.extract(pcm)
    sample = extractor.extract(pcm, min_speech_ms=600)
    assert sample.speech_ms == 1000
    assert provider.received[-1] == pcm
    assert sample.quality['required_speech_ms'] == 600


def test_padding_does_not_turn_insufficient_speech_into_identity_evidence(tmp_path):
    extractor, provider = engine(tmp_path)
    extractor.warm_up()
    pcm = b'\x01\x00' * 8000 + b'\x00\x00' * 24000
    count = len(provider.received)
    with pytest.raises(InvalidSpeakerAudio, match='speech_too_short') as caught:
        extractor.extract(pcm, min_speech_ms=600)
    assert caught.value.quality.audio_ms == 2000
    assert caught.value.quality.speech_ms == 500
    assert caught.value.quality.required_speech_ms == 600
    assert len(provider.received) == count
