import json
import wave

import pytest

from benchmark.speaker_dataset import extract_dataset


def write_wav(path, pcm=b"\x01\x00" * 32000, rate=16000):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


class Extractor:
    def __init__(self):
        self.calls = []

    def extract_embedding(self, pcm, sample_rate):
        self.calls.append((pcm, sample_rate))
        return [1.0, 0.0]


def manifest():
    return dict(schema_version=1, samples=[
        dict(id="e", speaker="a", device="mobile", capture_session="one",
             partition="enrollment", path="e.wav"),
        dict(id="p", speaker="a", device="box3", capture_session="two",
             partition="test", path="p.wav"),
    ], trials=[dict(id="positive", probe_id="p", enrollment_ids=["e"])])


MODEL = dict(id="test", weights_sha256="a" * 64, preprocessing="reference-contiguous-pcm16-v1")


def test_exact_contiguous_pcm_is_used_without_vad_splicing(tmp_path):
    pcm = b"\x01\x00" * 16000 + b"\x00\x00" * 8000 + b"\x02\x00" * 16000
    write_wav(tmp_path / "e.wav", pcm)
    write_wav(tmp_path / "p.wav")
    engine = Extractor()
    result = extract_dataset(manifest(), root=tmp_path, extractor=engine, model=MODEL)
    assert engine.calls[0] == (pcm, 16000)
    assert result["samples"][0]["duration_ms"] == 2500
    assert result["samples"][0]["status"] == "ok"
    assert "path" not in result["samples"][0]


def test_short_probe_remains_in_dataset_without_inference(tmp_path):
    write_wav(tmp_path / "e.wav")
    write_wav(tmp_path / "p.wav", b"\x01\x00" * 8000)
    engine = Extractor()
    result = extract_dataset(manifest(), root=tmp_path, extractor=engine, model=MODEL)
    assert len(engine.calls) == 1
    assert result["samples"][1]["status"] == "insufficient"
    assert result["samples"][1]["reason"] == "audio_too_short"
    assert "vector" not in result["samples"][1]


def test_wrong_sample_rate_is_not_silently_relabelled(tmp_path):
    write_wav(tmp_path / "e.wav", rate=48000)
    with pytest.raises(ValueError, match="16 kHz"):
        extract_dataset(manifest(), root=tmp_path, extractor=Extractor(), model=MODEL)


def test_silence_is_not_sent_to_embedding_model(tmp_path):
    write_wav(tmp_path / "e.wav")
    write_wav(tmp_path / "p.wav", b"\x00\x00" * 32000)
    engine = Extractor()
    result = extract_dataset(manifest(), root=tmp_path, extractor=engine, model=MODEL)
    assert len(engine.calls) == 1
    assert result["samples"][1]["reason"] == "digital_silence"


def test_path_cannot_escape_dataset_root(tmp_path):
    data = manifest()
    data["samples"][0]["path"] = "../outside.wav"
    with pytest.raises(ValueError, match="root"):
        extract_dataset(data, root=tmp_path, extractor=Extractor(), model=MODEL)


def test_model_errors_are_visible_without_private_exception_text(tmp_path):
    class Broken:
        def extract_embedding(self, *args):
            raise RuntimeError("secret audio path or data")

    write_wav(tmp_path / "e.wav")
    write_wav(tmp_path / "p.wav")
    result = extract_dataset(manifest(), root=tmp_path, extractor=Broken(), model=MODEL)
    assert result["samples"][0]["status"] == "unavailable"
    assert "secret" not in json.dumps(result)


def test_public_reference_adapter_preserves_signed_pcm_and_normalizes(tmp_path):
    import numpy as np
    from eidolon.livekit.plugins.speaker_verification.providers.modelscope_campplus import (
        ModelScopeCampPlusSpeakerVerificationProvider,
    )
    engine = ModelScopeCampPlusSpeakerVerificationProvider(model_dir=tmp_path)
    seen = []

    def pipeline(inputs, *, output_emb):
        seen.append(inputs[0].copy())
        assert output_emb is True
        return {"embs": [np.array([3, 4], dtype=np.float32)]}

    engine._pipeline = pipeline
    result = engine.extract_embedding(b"\x00\x80\x00\x00\xff\x7f", 16000)
    assert seen[0].tolist() == [-1, 0, 32767 / 32768]
    assert result.tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        engine.extract_embedding(b"\x00", 16000)
    with pytest.raises(ValueError):
        engine.extract_embedding(b"\x00\x00", 48000)
