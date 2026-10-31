"""Calibrated identity evidence (design §3.3): the runtime arithmetic."""

import hashlib
import json
import math

import numpy as np
import pytest

from eidolon.livekit.plugins.speaker_verification import evidence as ev


def unit(*v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def write_backend(root, *, weights=None, cohort=None, features=None, sha=None):
    cohort = np.asarray(cohort if cohort is not None else [unit(0, 1, 0), unit(0, 0, 1), unit(0, 1, 1)])
    with open(root / "identity-cohort.npz", "wb") as f:
        np.savez(f, mean=np.zeros(3, np.float32), cohort=cohort.astype(np.float32))
    spec = {
        "version": "v", "identity": ["m", "a" * 64, "p"], "top_n": 2, "max_speech_ms": 20000,
        "features": list(features or ev.FEATURES),
        "weights": list(weights if weights is not None else [1.0] + [0.0] * (len(ev.FEATURES) - 1)),
        "bias": 0.0,
        "cohort": {"path": "identity-cohort.npz",
                   "sha256": sha or hashlib.sha256((root / "identity-cohort.npz").read_bytes()).hexdigest()},
    }
    (root / ev.BACKEND_FILE).write_text(json.dumps(spec))
    return ev.load_backend(root)


def test_the_llr_is_the_affine_qmf_of_the_asnorm_score(tmp_path):
    backend = write_backend(tmp_path)
    enroll = backend.side(unit(1, 0, 0), 15000, "phone")
    probe = backend.side(unit(1, 0.1, 0), 2000, "phone")
    raw = float(enroll.vector @ probe.vector)
    expected = 0.5 * ((raw - enroll.cohort_mean) / enroll.cohort_std
                      + (raw - probe.cohort_mean) / probe.cohort_std)
    assert backend.llr(enroll, probe) == pytest.approx(expected)


def test_trial_features_are_symmetric_except_for_the_condition(tmp_path):
    backend = write_backend(tmp_path)
    a = backend.side(unit(1, 0.2, 0), 15000, "phone")
    b = backend.side(unit(1, 0, 0.3), 1200, "phone")
    assert np.allclose(ev.trial_features(a, b, 20000), ev.trial_features(b, a, 20000))
    names = dict(zip(ev.FEATURES, ev.trial_features(a, b, 20000)))
    assert names["speech_min_s"] == 1.2 and names["speech_max_s"] == 15.0
    assert names["cross_channel"] == 0.0


def test_a_probe_from_another_channel_is_a_different_condition(tmp_path):
    weights = [1.0] + [0.0] * (len(ev.FEATURES) - 3) + [-2.0, 0.5]
    backend = write_backend(tmp_path, weights=weights)
    enroll = backend.side(unit(1, 0, 0), 15000, "phone")
    same = backend.side(unit(1, 0.1, 0), 2000, "phone")
    other = backend.side(unit(1, 0.1, 0), 2000, "box3")
    names = dict(zip(ev.FEATURES, ev.trial_features(enroll, other, 20000)))
    assert names["cross_channel"] == 1.0
    assert names["cross_channel_x_asnorm_score"] == pytest.approx(names["asnorm_score"])
    assert backend.llr(enroll, other) == pytest.approx(
        backend.llr(enroll, same) * 1.5 - 2.0)


def test_durations_are_clipped_at_the_calibrated_maximum(tmp_path):
    backend = write_backend(tmp_path)
    a = backend.side(unit(1, 0, 0), 30000, "phone")
    b = backend.side(unit(1, 0, 0), 900, "phone")
    assert dict(zip(ev.FEATURES, ev.trial_features(a, b, 20000)))["speech_max_s"] == 20.0


def test_open_set_posteriors_include_someone_not_enrolled():
    persons, unknown = ev.posteriors({"a": 0.0, "b": 0.0}, {"a": 0.25, "b": 0.25}, 0.5)
    assert persons == pytest.approx({"a": 0.25, "b": 0.25}) and unknown == pytest.approx(0.5)
    persons, unknown = ev.posteriors({"a": math.log(99), "b": -50}, {"a": 0.25, "b": 0.25}, 0.5)
    assert persons["a"] == pytest.approx(99 * 0.25 / (0.5 + 99 * 0.25), rel=1e-9)
    assert sum(persons.values()) + unknown == pytest.approx(1)
    # Huge LLRs do not overflow.
    persons, unknown = ev.posteriors({"a": 900.0}, {"a": 0.5}, 0.5)
    assert persons["a"] == pytest.approx(1) and unknown >= 0


def test_the_backend_is_exactly_the_cohort_it_names(tmp_path):
    write_backend(tmp_path)
    (tmp_path / "identity-cohort.npz").write_bytes(b"changed")
    with pytest.raises(ev.BackendUnavailable, match="backend_invalid"):
        ev.load_backend(tmp_path)
    with pytest.raises(ev.BackendUnavailable, match="backend_missing"):
        ev.load_backend(tmp_path / "elsewhere")


def test_weights_for_other_features_are_refused(tmp_path):
    with pytest.raises(ev.BackendUnavailable, match="backend_invalid"):
        write_backend(tmp_path, features=list(ev.FEATURES[:-1]))


def test_the_domain_mean_is_removed_before_scoring(tmp_path):
    backend = write_backend(tmp_path)
    shifted = ev.Backend(**{**backend.__dict__, "mean": np.array([0.5, 0.0, 0.0])})
    side = shifted.side(unit(1, 0.1, 0), 2000, "phone")
    assert side.vector == pytest.approx(ev.unit(unit(1, 0.1, 0) - np.array([0.5, 0, 0]))[0])
