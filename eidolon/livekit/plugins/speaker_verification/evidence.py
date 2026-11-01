"""Calibrated identity evidence: scores become log-likelihood ratios (design §3.3).

Registration happens on the phone; speech arrives on every device. How well
the embedding separates speakers depends on the comparison (live
multi-device speech, design §3.3 ①a): a phone template against the phone
(``near_same``) or a device's own observation template against that device
(``far_same``) separates speakers well with a plain cosine; a template
against another device (``far_cross``) does not, and is scored after a
channel-compensating LDA projection trained on live multi-device speech.
Compensation only loses within one device, so same-device comparisons skip it.

Every condition has its own affine quality-measure function (QMF) over the
score and WeSpeaker's min / max / difference / ratio of the two sides' voiced
durations (``wespeaker/bin/score_calibration.py``, Apache-2.0,
wenet-e2e/wespeaker). The weights are trained offline with WeSpeaker's own
``train_calibration_model`` on live multi-device recordings
(``evals/speaker-verification/identity-20261101``); only the fixed arithmetic
runs here. No AS-norm: a cohort recorded elsewhere inflated scores on every
real channel it met, and on live speech no cohort helped across devices.

Enrollment and probe are symmetric: an enrollment is a Person's template
with the voiced speech it was built from, a probe is a segment or a track.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from eidolon_sdk.biz.speaker_identity import IDENTITY_CONDITIONS, comparison_condition

FEATURES = ("score", "duration_min_s", "duration_max_s", "duration_diff_s", "duration_ratio")
# The conditions scored in the compensated space.
COMPENSATED = frozenset({"far_cross"})


class BackendUnavailable(ValueError):
    pass


def unit(rows) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


@dataclass(frozen=True)
class Side:
    """One side of a trial: mean removed, plain and channel-compensated."""

    vector: np.ndarray
    compensated: np.ndarray
    speech_ms: int
    channel: str


@dataclass(frozen=True)
class Backend:
    version: str
    identity: tuple[str, str, str]
    mean: np.ndarray
    projection: np.ndarray
    max_speech_ms: int
    # condition -> (weights over FEATURES, bias)
    calibrations: dict[str, tuple[np.ndarray, float]]

    def side(self, vector, speech_ms: int, channel: str) -> Side:
        centred = np.asarray(vector, dtype=np.float64) - self.mean
        return Side(unit(centred)[0], unit(centred @ self.projection)[0], int(speech_ms), channel)

    def condition(self, enroll: Side, probe: Side) -> str:
        return comparison_condition(enroll.channel, probe.channel)

    def features(self, enroll: Side, probe: Side, condition: str) -> np.ndarray:
        return trial_features(enroll, probe, condition, self.max_speech_ms)

    def llr(self, enroll: Side, probe: Side) -> float:
        condition = self.condition(enroll, probe)
        weights, bias = self.calibrations[condition]
        return float(self.features(enroll, probe, condition) @ weights + bias)


def score(enroll: Side, probe: Side, condition: str) -> float:
    if condition in COMPENSATED:
        return float(enroll.compensated @ probe.compensated)
    return float(enroll.vector @ probe.vector)


def _reorder(a: float, b: float) -> list[float]:
    """WeSpeaker's reorder_values: min, max, max - min, max / min."""
    low, high = min(a, b), max(a, b)
    return [low, high, high - low, high / low]


def trial_features(enroll: Side, probe: Side, condition: str, max_speech_ms: int) -> np.ndarray:
    return np.array([
        score(enroll, probe, condition),
        *_reorder(min(enroll.speech_ms, max_speech_ms) / 1000, min(probe.speech_ms, max_speech_ms) / 1000),
    ])


def posteriors(llrs: dict[str, float], priors: dict[str, float], prior_unknown: float):
    """Open-set posterior over the Persons compared and "someone not enrolled".

    P(i) = π_i·e^{LLR_i} / (π_0 + Σ_j π_j·e^{LLR_j}), computed in log space.
    """
    terms = {pid: math.log(priors[pid]) + llr for pid, llr in llrs.items()}
    terms[None] = math.log(prior_unknown)
    top = max(terms.values())
    weights = {k: math.exp(v - top) for k, v in terms.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items() if k is not None}, weights[None] / total


BACKEND_FILE = "identity-backend.json"


def load_backend(model_dir) -> Backend:
    """The calibration and the arrays it was trained with, checked by digest."""
    root = Path(model_dir)
    try:
        spec = json.loads((root / BACKEND_FILE).read_text())
        data = (root / spec["arrays"]["path"]).read_bytes()
    except (OSError, ValueError, KeyError):
        raise BackendUnavailable("backend_missing") from None
    try:
        valid = (hashlib.sha256(data).hexdigest() == spec["arrays"]["sha256"]
                 and spec["features"] == list(FEATURES)
                 and set(spec["conditions"]) == set(IDENTITY_CONDITIONS))
        calibrations = {
            c: (np.asarray(v["weights"], dtype=np.float64), float(v["bias"]))
            for c, v in spec["conditions"].items()
        }
        with np.load(root / spec["arrays"]["path"], allow_pickle=False) as npz:
            mean = np.asarray(npz["mean"], dtype=np.float64)
            projection = np.asarray(npz["projection"], dtype=np.float64)
    except (KeyError, ValueError, TypeError, OSError):
        raise BackendUnavailable("backend_invalid") from None
    if (not valid or mean.ndim != 1 or projection.ndim != 2 or projection.shape[0] != mean.shape[0]
            or any(w.shape != (len(FEATURES),) for w, _ in calibrations.values())):
        raise BackendUnavailable("backend_invalid")
    return Backend(
        version=spec["version"], identity=tuple(spec["identity"]), mean=mean, projection=projection,
        max_speech_ms=int(spec["max_speech_ms"]), calibrations=calibrations,
    )
