"""Per-Person, per-duration thresholds from labels nobody has to give.

Design §3.3. Registration windows are the Person's own voice; the cohort
(public speakers, delivered beside the model) and the household's other
registrations are other people. From those alone each Person gets, per
duration band, the lowest threshold that keeps the estimated false-accept
rate within the policy, and a verdict on whether that threshold still
accepts enough of the Person's own windows to decide directly.

Everything here is deterministic: the same templates, windows and cohort
always give the same table, so the table is recomputed, never stored.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from eidolon_sdk.biz.speaker_identity import (
    CohortReference,
    DurationBand,
    RecognitionCalibration,
    SpeakerRecognitionPolicy,
)
from eidolon_sdk.biz.voiceprint import (
    CalibrationCrop,
    GenuineScore,
    TemplateCalibration,
)

SAMPLE_RATE = 16000
# Raw window lengths; the band is decided by the voiced speech the extractor
# measures in each window, exactly as for a conversation probe.
WINDOW_SECONDS = (1.0, 2.0, 3.5)
CROP_MIN_SPEECH_MS = 600
# Stored windows keep six decimals; a threshold placed above a household
# window clears it by more than that rounding can move a cosine.
ROUNDING_ALLOWANCE = 1e-4


def calibration_windows(total_samples: int, speech: tuple[int, int] | None) -> list[tuple[int, int]]:
    """Windows anchored to the start and the end of the voiced span.

    Two per length, so a short registration still yields windows in every
    band it can reach; identical windows collapse.
    """
    if speech is None:
        return []
    first, last = speech
    windows = []
    for seconds in WINDOW_SECONDS:
        size = int(seconds * SAMPLE_RATE)
        if size > total_samples:
            continue
        start = max(0, min(first, total_samples - size))
        end = min(total_samples, max(last, size))
        for window in ((start, start + size), (end - size, end)):
            if window not in windows:
                windows.append(window)
    return windows


def crop(speech_ms: int, vector) -> CalibrationCrop:
    return CalibrationCrop(speech_ms=speech_ms, vector=[float(x) for x in vector])


def _unit(rows) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def enrollment_calibration(samples, trial, template) -> TemplateCalibration:
    """Genuine scores whose window never contributed to the template it meets.

    A registration sample's windows meet the mean of the other two samples;
    the trial's windows meet the full template (the trial is not part of it).
    """
    vectors = [np.asarray(s.vector, dtype=np.float64) for s in samples]
    genuine = []
    crops = []
    for index, sample in enumerate(samples):
        others = _unit(np.mean([v for i, v in enumerate(vectors) if i != index], axis=0))[0]
        for window in sample.crops:
            crops.append(window)
            genuine.append((window.speech_ms, float(_unit(window.vector)[0] @ others)))
    full = _unit(template.vector)[0]
    for window in trial.crops:
        crops.append(window)
        genuine.append((window.speech_ms, float(_unit(window.vector)[0] @ full)))
    return TemplateCalibration(
        crops=crops[:32],
        genuine=[GenuineScore(speech_ms=ms, similarity=max(-1.0, min(1.0, s)))
                 for ms, s in genuine[:32]],
    )


class CohortUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class Cohort:
    """Impostor embeddings: rows of public speakers, each with its voiced length."""

    vectors: np.ndarray
    speech_ms: np.ndarray
    speakers: np.ndarray
    identity: tuple[str, str, str]

    @classmethod
    def load(cls, model_dir, reference: CohortReference) -> "Cohort":
        path = Path(model_dir) / reference.path
        try:
            data = path.read_bytes()
        except OSError:
            raise CohortUnavailable("cohort_missing") from None
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise CohortUnavailable("cohort_invalid")
        try:
            with np.load(path, allow_pickle=False) as npz:
                vectors = _unit(npz["vectors"]).astype(np.float32)
                speech_ms = npz["speech_ms"].astype(np.int64)
                speakers = npz["speakers"].astype(np.int64)
                identity = tuple(str(npz[k]) for k in
                                 ("model_id", "weights_hash", "preprocessing_version"))
        except (KeyError, ValueError, OSError):
            raise CohortUnavailable("cohort_invalid") from None
        if not (len(vectors) == len(speech_ms) == len(speakers)) or not len(vectors):
            raise CohortUnavailable("cohort_invalid")
        if len(set(speakers.tolist())) != reference.speakers:
            raise CohortUnavailable("cohort_invalid")
        return cls(vectors, speech_ms, speakers, identity)


_COHORTS: dict[tuple[str, str], Cohort] = {}


def load_cohort(model_dir, policy: SpeakerRecognitionPolicy) -> Cohort:
    """One copy per process; the sha in the policy names the exact file."""
    calibration = policy.calibration
    key = (str(Path(model_dir).resolve()), calibration.cohort.sha256)
    if key not in _COHORTS:
        cohort = Cohort.load(model_dir, calibration.cohort)
        if cohort.identity != (policy.model_id, policy.weights_hash, policy.preprocessing_version):
            raise CohortUnavailable("cohort_model_mismatch")
        _COHORTS[key] = cohort
    return _COHORTS[key]


@dataclass(frozen=True)
class BandThreshold:
    band: str
    threshold: float
    # False: this band only extends a track already bound (§3.3), never decides.
    direct: bool
    # Share of the Person's own registration windows the threshold accepts;
    # None when there are too few windows in this band to say.
    genuine_acceptance: float | None
    genuine_scores: int
    estimated_false_accept: float
    cohort_scores: int
    household_max: float | None


def _band_rows(calibration: RecognitionCalibration, speech_ms, band: DurationBand) -> np.ndarray:
    names = np.array([(b.name if (b := calibration.band(int(ms))) else "") for ms in speech_ms])
    return names == band.name


def _lowest_threshold(scores: np.ndarray, max_false_accept: float) -> float:
    """Lowest t with mean(scores >= t) <= max_false_accept; 1 when unknowable."""
    if not len(scores):
        return 1.0
    ordered = np.sort(scores)[::-1]
    allowed = int(math.floor(max_false_accept * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def person_thresholds(
    policy: SpeakerRecognitionPolicy,
    cohort: Cohort,
    templates,
    calibrations,
    household_crops,
    persons: int = 1,
) -> dict[str, BandThreshold]:
    """One Person's table.

    templates: the Person's template vectors (scores take the best one).
    calibrations: TemplateCalibration of those templates (None when absent).
    household_crops: registration windows of every *other* Person.
    persons: how many Persons a stranger's turn is compared with. The policy's
    false-accept target is per turn, and a stranger can be mistaken for any of
    them, so each Person gets an equal share of it.
    """
    calibration = policy.calibration
    max_false_accept = calibration.max_false_accept / max(persons, 1)
    # float64 throughout: runtime scores are float64 cosines, and a threshold
    # placed just above a household score must stay above it.
    matrix = _unit([t for t in templates])
    cohort_scores = (cohort.vectors.astype(np.float64) @ matrix.T).max(axis=1)
    genuine = [g for c in calibrations if c is not None for g in c.genuine]
    measured = bool(genuine)
    if household_crops:
        house_vectors = _unit([c.vector for c in household_crops])
        house_scores = (house_vectors @ matrix.T).max(axis=1)
        house_ms = np.array([c.speech_ms for c in household_crops])
    else:
        house_scores, house_ms = np.zeros(0), np.zeros(0, dtype=np.int64)
    table = {}
    for band in calibration.bands:
        impostors = cohort_scores[_band_rows(calibration, cohort.speech_ms, band)]
        house = house_scores[_band_rows(calibration, house_ms, band)] if len(house_ms) else house_scores
        threshold = max(band.floor, _lowest_threshold(impostors, max_false_accept))
        if len(house):
            threshold = max(threshold, float(house.max()) + ROUNDING_ALLOWANCE)
        if not measured:
            # A Person registered before windows existed decides as before,
            # never below what the cohort requires.
            threshold = max(threshold, policy.threshold)
        own = [g.similarity for g in genuine if (b := calibration.band(g.speech_ms)) and b.name == band.name]
        acceptance = (sum(s >= threshold for s in own) / len(own)
                      if len(own) >= calibration.min_genuine_scores else None)
        direct = (not measured) or (acceptance is not None
                                    and acceptance >= calibration.min_genuine_acceptance)
        table[band.name] = BandThreshold(
            band=band.name,
            threshold=min(threshold, 1.0),
            direct=direct,
            genuine_acceptance=acceptance,
            genuine_scores=len(own),
            estimated_false_accept=float((impostors >= threshold).mean()) if len(impostors) else 1.0,
            cohort_scores=int(len(impostors)),
            household_max=float(house.max()) if len(house) else None,
        )
    return table


def household_tables(policy: SpeakerRecognitionPolicy, cohort: Cohort, persons) -> dict:
    """Tables for every Person in one recognition snapshot."""
    crops = {
        p.person_id: [c for t in p.templates if t.calibration for c in t.calibration.crops]
        for p in persons
    }
    return {
        p.person_id: person_thresholds(
            policy, cohort,
            [t.sample.vector for t in p.templates],
            [t.calibration for t in p.templates],
            [c for other, cs in crops.items() if other != p.person_id for c in cs],
            persons=len(persons),
        )
        for p in persons
    }


@dataclass(frozen=True)
class Decision:
    outcome: str  # identified | below | ambiguous | not_direct
    person_id: str | None
    score: float | None
    threshold: float | None
    margin: float | None


def decide(scores: dict[str, float], tables: dict[str, dict[str, BandThreshold]],
           band: DurationBand) -> Decision:
    """Accept the best Person over its own threshold, leading everyone by the band margin."""
    if not scores:
        return Decision("below", None, None, None, None)
    passing = [p for p, s in scores.items() if s >= tables[p][band.name].threshold]
    if not passing:
        best = max(scores, key=scores.get)
        return Decision("below", None, scores[best], tables[best][band.name].threshold, None)
    best = max(passing, key=scores.get)
    entry = tables[best][band.name]
    others = [s for p, s in scores.items() if p != best]
    margin = scores[best] - max(others) if others else None
    if margin is not None and margin < band.margin:
        return Decision("ambiguous", None, scores[best], entry.threshold, margin)
    if not entry.direct:
        return Decision("not_direct", None, scores[best], entry.threshold, margin)
    return Decision("identified", best, scores[best], entry.threshold, margin)
