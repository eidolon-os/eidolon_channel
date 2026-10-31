"""Calibrated identity evidence: scores become log-likelihood ratios (design §3.3).

The back-end follows WeSpeaker (Apache-2.0, wenet-e2e/wespeaker):

- ``wespeaker/bin/score_norm.py``: subtract a domain mean from every
  embedding, then adaptive score normalisation (AS-norm) with the mean and
  standard deviation of each side's top-N cohort scores;
- ``wespeaker/bin/score_calibration.py``: an affine quality-measure function
  (QMF) over the normalised score and min / max / difference / ratio of each
  quality measure, trained offline with a Cllr objective.

Only the fixed arithmetic runs here; the mean, the cohort and the weights
are delivered with the model artifact. Two deliberate differences: the
extractor returns unit vectors, so the embedding-magnitude measure is not
available; and cohort means can be near zero or negative, so they enter as
min / max / difference without a ratio.

The calibration is condition-aware (Ferrer & McLaren's discriminative
condition-aware back-end): whether the probe comes through a different
channel from the one the template was built on enters as an indicator and
as its product with the score. A single global calibration over-trusts
same-channel strangers and under-trusts cross-channel owners.

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

FEATURES = (
    "asnorm_score",
    "speech_min_s", "speech_max_s", "speech_diff_s", "speech_ratio",
    "cohort_mean_min", "cohort_mean_max", "cohort_mean_diff",
    "cross_channel", "cross_channel_x_asnorm_score",
)


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
    """One side of a trial after mean subtraction, with its cohort statistics."""

    vector: np.ndarray
    speech_ms: int
    cohort_mean: float
    cohort_std: float
    channel: str


@dataclass(frozen=True)
class Backend:
    version: str
    identity: tuple[str, str, str]
    mean: np.ndarray
    cohort: np.ndarray
    top_n: int
    max_speech_ms: int
    weights: np.ndarray
    bias: float

    def side(self, vector, speech_ms: int, channel: str) -> Side:
        prepared = unit(np.asarray(vector, dtype=np.float64) - self.mean)[0]
        scores = np.sort(self.cohort @ prepared)[::-1][: self.top_n]
        return Side(prepared, int(speech_ms), float(scores.mean()), float(max(scores.std(), 1e-6)),
                    channel)

    def features(self, enroll: Side, probe: Side) -> np.ndarray:
        return trial_features(enroll, probe, self.max_speech_ms)

    def llr(self, enroll: Side, probe: Side) -> float:
        return float(self.features(enroll, probe) @ self.weights + self.bias)


def trial_features(enroll: Side, probe: Side, max_speech_ms: int) -> np.ndarray:
    raw = float(enroll.vector @ probe.vector)
    asnorm = 0.5 * ((raw - enroll.cohort_mean) / enroll.cohort_std
                    + (raw - probe.cohort_mean) / probe.cohort_std)
    durations = sorted(min(s.speech_ms, max_speech_ms) / 1000 for s in (enroll, probe))
    means = sorted((enroll.cohort_mean, probe.cohort_mean))
    cross = float(enroll.channel != probe.channel)
    return np.array([
        asnorm,
        durations[0], durations[1], durations[1] - durations[0], durations[1] / max(durations[0], 1e-3),
        means[0], means[1], means[1] - means[0],
        cross, cross * asnorm,
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
    """The calibration and the cohort it was trained with, checked by digest."""
    root = Path(model_dir)
    try:
        spec = json.loads((root / BACKEND_FILE).read_text())
        data = (root / spec["cohort"]["path"]).read_bytes()
    except (OSError, ValueError, KeyError):
        raise BackendUnavailable("backend_missing") from None
    if hashlib.sha256(data).hexdigest() != spec["cohort"]["sha256"] or spec["features"] != list(FEATURES):
        raise BackendUnavailable("backend_invalid")
    try:
        with np.load(root / spec["cohort"]["path"], allow_pickle=False) as npz:
            mean = np.asarray(npz["mean"], dtype=np.float64)
            cohort = unit(np.asarray(npz["cohort"], dtype=np.float64) - mean)
    except (KeyError, ValueError, OSError):
        raise BackendUnavailable("backend_invalid") from None
    return Backend(
        version=spec["version"], identity=tuple(spec["identity"]), mean=mean, cohort=cohort,
        top_n=int(spec["top_n"]), max_speech_ms=int(spec["max_speech_ms"]),
        weights=np.asarray(spec["weights"], dtype=np.float64), bias=float(spec["bias"]),
    )
