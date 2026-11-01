"""Calibrated identity evidence (design §3.3): the runtime arithmetic."""

import hashlib
import json
import math

import numpy as np
import pytest

from eidolon.livekit.plugins.speaker_verification import evidence as ev

CONDITIONS = ("near_same", "far_same", "far_cross")


def unit(*v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def write_backend(root, *, conditions=None, projection=None, mean=None, features=None, sha=None):
    projection = np.asarray(projection if projection is not None else [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    with open(root / "identity-backend.npz", "wb") as f:
        np.savez(f, mean=np.asarray(mean if mean is not None else np.zeros(3), np.float32),
                 projection=projection.astype(np.float32))
    spec = {
        "version": "v", "identity": ["m", "a" * 64, "p"], "max_speech_ms": 20000,
        "features": list(features or ev.FEATURES),
        "conditions": {c: {"max_track_segments": 5, **v} for c, v in (conditions or {
            c: {"weights": [1.0, 0, 0, 0, 0], "bias": 0.0} for c in CONDITIONS}).items()},
        "arrays": {"path": "identity-backend.npz",
                   "sha256": sha or hashlib.sha256((root / "identity-backend.npz").read_bytes()).hexdigest()},
    }
    (root / ev.BACKEND_FILE).write_text(json.dumps(spec))
    return ev.load_backend(root)


def test_same_device_comparisons_score_the_plain_cosine(tmp_path):
    backend = write_backend(tmp_path)
    enroll = backend.side(unit(1, 0, 0), 15000, "mobile")
    probe = backend.side(unit(1, 0.5, 0), 2000, "mobile")
    assert backend.condition(enroll, probe) == "near_same"
    assert backend.llr(enroll, probe) == pytest.approx(float(unit(1, 0, 0) @ unit(1, 0.5, 0)))
    assert backend.max_track_segments == {c: 5 for c in CONDITIONS}


def test_cross_device_comparisons_score_after_channel_compensation(tmp_path):
    # The projection drops the second axis: a difference only there is channel, not speaker.
    backend = write_backend(tmp_path)
    enroll = backend.side(unit(1, 0, 0), 15000, "mobile")
    probe = backend.side(unit(1, 0.5, 0), 2000, "esp-box-3")
    assert backend.condition(enroll, probe) == "far_cross"
    assert backend.llr(enroll, probe) == pytest.approx(1.0)


def test_a_device_against_its_own_observation_template_is_far_same(tmp_path):
    backend = write_backend(tmp_path)
    a = backend.side(unit(1, 0, 0), 3000, "esp-box-3")
    b = backend.side(unit(1, 0.5, 0), 1200, "esp-box-3")
    assert backend.condition(a, b) == "far_same"


def test_each_condition_has_its_own_calibration(tmp_path):
    conditions = {"near_same": {"weights": [2.0, 0, 0, 0, 0], "bias": 1.0},
                  "far_same": {"weights": [3.0, 0, 0, 0, 0], "bias": 0.0},
                  "far_cross": {"weights": [4.0, 0, 0, 0, 0], "bias": -1.0}}
    backend = write_backend(tmp_path, conditions=conditions)
    v = unit(1, 0.2, 0.1)
    enroll = backend.side(unit(1, 0, 0), 15000, "mobile")
    near = backend.llr(enroll, backend.side(v, 2000, "mobile"))
    cross = backend.llr(enroll, backend.side(v, 2000, "esp-box-3"))
    plain = float(unit(1, 0, 0) @ v)
    compensated = float(unit(1, 0) @ unit(v[0], v[2]))
    assert near == pytest.approx(2 * plain + 1) and cross == pytest.approx(4 * compensated - 1)


def test_duration_factors_follow_wespeakers_layout_and_cap(tmp_path):
    backend = write_backend(tmp_path)
    a = backend.side(unit(1, 0.2, 0), 30000, "mobile")
    b = backend.side(unit(1, 0, 0.3), 1200, "mobile")
    names = dict(zip(ev.FEATURES, ev.trial_features(a, b, "near_same", 20000)))
    assert (names["duration_min_s"], names["duration_max_s"]) == (1.2, 20.0)
    assert names["duration_diff_s"] == pytest.approx(18.8)
    assert names["duration_ratio"] == pytest.approx(20.0 / 1.2)
    assert np.allclose(ev.trial_features(a, b, "near_same", 20000), ev.trial_features(b, a, "near_same", 20000))


def test_the_mean_of_live_speech_is_removed_before_scoring(tmp_path):
    backend = write_backend(tmp_path, mean=[0.5, 0.0, 0.0])
    side = backend.side(unit(1, 0.1, 0), 2000, "mobile")
    assert side.vector == pytest.approx(ev.unit(unit(1, 0.1, 0) - np.array([0.5, 0, 0]))[0])


def test_open_set_posteriors_include_someone_not_enrolled():
    persons, unknown = ev.posteriors({"a": 0.0, "b": 0.0}, {"a": 0.25, "b": 0.25}, 0.5)
    assert persons == pytest.approx({"a": 0.25, "b": 0.25}) and unknown == pytest.approx(0.5)
    persons, unknown = ev.posteriors({"a": math.log(99), "b": -50}, {"a": 0.25, "b": 0.25}, 0.5)
    assert persons["a"] == pytest.approx(99 * 0.25 / (0.5 + 99 * 0.25), rel=1e-9)
    assert sum(persons.values()) + unknown == pytest.approx(1)
    persons, unknown = ev.posteriors({"a": 900.0}, {"a": 0.5}, 0.5)
    assert persons["a"] == pytest.approx(1) and unknown >= 0


def test_the_backend_is_exactly_the_arrays_it_names(tmp_path):
    write_backend(tmp_path)
    (tmp_path / "identity-backend.npz").write_bytes(b"changed")
    with pytest.raises(ev.BackendUnavailable, match="backend_invalid"):
        ev.load_backend(tmp_path)
    with pytest.raises(ev.BackendUnavailable, match="backend_missing"):
        ev.load_backend(tmp_path / "elsewhere")


@pytest.mark.parametrize("change", [
    {"features": list(ev.FEATURES[:-1])},
    {"conditions": {c: {"weights": [1.0, 0, 0, 0, 0], "bias": 0.0} for c in CONDITIONS[:2]}},
    {"conditions": {c: {"weights": [1.0, 0, 0], "bias": 0.0} for c in CONDITIONS}},
    {"projection": [[1.0, 0.0], [0.0, 1.0]]},
    {"conditions": {c: {"weights": [1.0, 0, 0, 0, 0], "bias": 0.0, "max_track_segments": 0}
                    for c in CONDITIONS}},
])
def test_a_backend_that_does_not_fit_the_runtime_is_refused(tmp_path, change):
    with pytest.raises(ev.BackendUnavailable, match="backend_invalid"):
        write_backend(tmp_path, **change)
