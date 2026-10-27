import hashlib
import json
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

from eidolon.livekit.plugins.speaker_verification.providers.speechbrain_ecapa import (
    SpeechBrainEcapaProvider,
)
from eidolon.livekit.plugins.speaker_verification.reference_extractor import ReferenceSpeakerExtractor
from .test_reference_speaker_extractor import Vad


@pytest.mark.parametrize("vector", [np.zeros(192), np.ones(191), np.full(192, np.nan)])
def test_pvad_fallback_or_invalid_vector_never_becomes_identity(vector):
    provider = SpeechBrainEcapaProvider(
        model_dir="unused", extractor_factory=lambda _: SimpleNamespace(
            is_loaded=True, get_embedding=lambda _: vector,
        ),
    )
    with pytest.raises(RuntimeError, match="invalid_ecapa_embedding"):
        provider.extract_embedding(b"\1\0" * 24000, 16000)


def test_missing_model_fails_closed():
    provider = SpeechBrainEcapaProvider(
        model_dir="unused", extractor_factory=lambda _: SimpleNamespace(is_loaded=False),
    )
    with pytest.raises(RuntimeError, match="ecapa_model_unavailable"):
        provider.extract_embedding(b"\1\0" * 24000, 16000)


def test_ecapa_receives_signed_contiguous_pcm():
    seen = []

    def extract(samples):
        seen.append(samples)
        return np.full((1, 192), 1 / np.sqrt(192))

    provider = SpeechBrainEcapaProvider(
        model_dir="unused", extractor_factory=lambda _: SimpleNamespace(
            is_loaded=True, get_embedding=extract,
        ),
    )
    provider.extract_embedding(b"\x00\x80\x00\x00\xff\x7f", 16000)
    assert seen[0].tolist() == [-1, 0, 32767 / 32768]
    with pytest.raises(ValueError, match="invalid_ecapa_pcm"):
        provider.extract_embedding(b"\1\0", 48000)


def make_engine(tmp_path):
    artifacts = []
    for name in ("embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt",
                 "label_encoder.txt", "hyperparams.yaml"):
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        artifacts.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(),
                              size_bytes=len(data)))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        model_alias="ecapa_voxceleb_16k", artifacts=artifacts,
    )))
    provider = SimpleNamespace(model="ecapa_voxceleb_16k",
                               extract_embedding=lambda *_: np.array([1., 0.]))
    return ReferenceSpeakerExtractor(tmp_path, provider=provider, vad_factory=Vad)


def test_config_version_is_part_of_ecapa_identity(tmp_path):
    engine = make_engine(tmp_path)
    engine.warm_up()
    identity = engine.model_identity
    config = tmp_path / "hyperparams.yaml"
    config.write_bytes(b"different features")
    with pytest.raises(ValueError, match="artifact_integrity"):
        engine.warm_up()
    assert engine.model_identity is None
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    item = next(x for x in manifest["artifacts"] if x["path"] == config.name)
    item.update(sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                size_bytes=config.stat().st_size)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    engine.warm_up()
    assert engine.model_identity != identity
    assert engine.extract(b"\1\0" * 32000).model_id == "ecapa_voxceleb_16k"


def test_wrong_provider_cannot_relabel_an_existing_template(tmp_path):
    engine = make_engine(tmp_path)
    engine._provider.model = "campplus_zh_16k_common"
    with pytest.raises(ValueError, match="speaker_model_mismatch"):
        engine.warm_up()


def test_rewarming_reloads_model_before_advertising_new_digest(tmp_path, monkeypatch):
    from eidolon.livekit.plugins.speaker_verification.providers import speechbrain_ecapa

    make_engine(tmp_path)
    loaded = []

    class Provider:
        model = "ecapa_voxceleb_16k"

        def __init__(self, *, model_dir):
            self.weights = (model_dir / "embedding_model.ckpt").read_bytes()
            loaded.append(self.weights)

        def extract_embedding(self, *args):
            return np.array([1., 0.])

    monkeypatch.setattr(speechbrain_ecapa, "SpeechBrainEcapaProvider", Provider)
    engine = ReferenceSpeakerExtractor(tmp_path, vad_factory=Vad)
    engine.warm_up()
    original = engine.model_identity
    weights = tmp_path / "embedding_model.ckpt"
    weights.write_bytes(b"new weights")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    item = next(x for x in manifest["artifacts"] if x["path"] == weights.name)
    item.update(sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                size_bytes=weights.stat().st_size)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    engine.warm_up()
    assert engine.model_identity != original
    assert loaded == [b"embedding_model.ckpt", b"new weights"]


def test_local_loader_never_depends_on_cwd_or_writes_model_assets(tmp_path, monkeypatch):
    from eidolon.livekit.plugins.vad.firered.processor import SpeakerEmbExtractor

    seen = []

    def load(**kwargs):
        seen.append(kwargs)
        assert kwargs["source"] == str(tmp_path)
        assert kwargs["overrides"] == {"pretrained_path": str(tmp_path)}
        assert Path(kwargs["savedir"]).is_dir()
        assert Path(kwargs["savedir"]) != tmp_path
        return object()

    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", SimpleNamespace(
        EncoderClassifier=SimpleNamespace(from_hparams=load),
    ))
    extractor = SpeakerEmbExtractor(tmp_path)
    assert extractor.is_loaded
    assert not Path(seen[0]["savedir"]).exists()
    assert list(tmp_path.iterdir()) == []
