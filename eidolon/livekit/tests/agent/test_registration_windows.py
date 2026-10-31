"""Registration windows (design §3.3): same-person evidence registration gives for free."""

import hashlib
import json
import math

import numpy as np

from eidolon_sdk.biz.voiceprint import CalibrationCrop, EnrollmentSample, ExtractedSample

from eidolon.livekit.plugins.speaker_verification import calibration as cal
from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    ReferenceSpeakerExtractor,
)

IDENTITY = ("m", "a" * 64, "v")


def unit(*values):
    v = np.asarray(values, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------- windows


def test_windows_anchor_to_the_voiced_span_and_skip_lengths_that_do_not_fit():
    windows = cal.calibration_windows(16000 * 5, (8000, 72000))
    assert (8000, 24000) in windows and (56000, 72000) in windows  # 1 s at both ends
    assert (8000, 40000) in windows and (40000, 72000) in windows  # 2 s
    assert (8000, 64000) in windows and (16000, 72000) in windows  # 3.5 s
    assert len(windows) == 6
    short = cal.calibration_windows(16000 * 3, (0, 48000))
    assert all(end - start <= 32000 for start, end in short)


def test_a_window_never_leaves_the_recording_and_duplicates_collapse():
    windows = cal.calibration_windows(16000, (0, 16000))
    assert windows == [(0, 16000)]
    assert cal.calibration_windows(16000, None) == []
    for start, end in cal.calibration_windows(40000, (30000, 39000)):
        assert 0 <= start < end <= 40000


# ---------------------------------------------------------------- genuine scores


def sample(vector, crops=()):
    return EnrollmentSample(model_id=IDENTITY[0], weights_hash=IDENTITY[1],
        preprocessing_version=IDENTITY[2], vector=vector, speech_ms=4000,
        crops=[CalibrationCrop(speech_ms=ms, vector=v) for ms, v in crops])


def test_a_window_never_meets_a_template_built_from_its_own_recording():
    a, b, c = unit(1, 0.2, 0), unit(1, -0.2, 0), unit(1, 0, 0.6)
    samples = [sample(a, [(900, a)]), sample(b, [(900, b)]), sample(c, [(900, c)])]
    trial = sample(unit(1, 0, 0), [(3000, unit(1, 0.1, 0.1))])
    template = ExtractedSample(**samples[0].sample().model_dump() | {"vector": np.mean([a, b, c], axis=0).tolist()})
    result = cal.enrollment_calibration(samples, trial, template)
    assert [g.speech_ms for g in result.genuine] == [900, 900, 900, 3000]
    loo = cal._unit(np.mean([b, c], axis=0))[0]
    assert math.isclose(result.genuine[0].similarity, float(np.dot(a, loo)), abs_tol=1e-5)
    full = cal._unit(template.vector)[0]
    assert result.genuine[0].similarity < float(np.dot(a, full))
    assert math.isclose(result.genuine[3].similarity, float(np.dot(trial.crops[0].vector, full)), abs_tol=1e-5)
    assert len(result.crops) == 4


# ---------------------------------------------------------------- extraction


class Provider:
    model = "campplus_zh_16k_common"

    def __init__(self):
        self.received = []

    def extract_embedding(self, pcm, sample_rate):
        self.received.append(pcm)
        return np.array([1.0, len(pcm) / 1e6])


class Vad:
    window_size_samples = 160

    def reset(self):
        pass

    def __call__(self, samples):
        return float(np.any(samples))


def extractor(tmp_path):
    artifacts = []
    for name in ("campplus_cn_common.bin", "config.yaml", "configuration.json"):
        (tmp_path / name).write_bytes(b"fixture")
        artifacts.append(dict(path=name, sha256=hashlib.sha256(b"fixture").hexdigest(), size_bytes=7))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(artifacts=artifacts)))
    provider = Provider()
    value = ReferenceSpeakerExtractor(tmp_path, provider=provider, vad_factory=Vad)
    value.warm_up()
    provider.received.clear()
    return value, provider


def test_registration_windows_are_extracted_like_probes(tmp_path):
    value, provider = extractor(tmp_path)
    pcm = b"\x00\x00" * 8000 + b"\x01\x00" * 64000 + b"\x00\x00" * 8000  # 4 s voiced inside 5 s
    result = value.extract_enrollment(pcm)
    assert result.speech_ms == 4000
    assert provider.received[0] == pcm
    assert len(result.crops) == 6
    assert {c.speech_ms for c in result.crops} == {1000, 2000, 3500}
    # Each window's embedding came from exactly that window's PCM.
    assert sorted(len(p) for p in provider.received[1:]) == sorted([32000] * 2 + [64000] * 2 + [112000] * 2)
    plain = value.extract(pcm)
    assert not hasattr(plain, "crops")


def test_windows_with_too_little_speech_are_dropped(tmp_path):
    value, _ = extractor(tmp_path)
    pcm = b"\x01\x00" * 25000 + b"\x00\x00" * 31000  # 1.56 s voiced at the start of 3.5 s
    result = value.extract_enrollment(pcm)
    assert all(c.speech_ms >= cal.CROP_MIN_SPEECH_MS for c in result.crops)
