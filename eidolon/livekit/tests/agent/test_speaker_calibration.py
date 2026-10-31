"""Per-Person, per-duration calibration (design §3.3): free labels to thresholds."""

import hashlib
import json
import math

import numpy as np
import pytest
from unittest.mock import AsyncMock
from types import SimpleNamespace

from eidolon_sdk.biz.person import PersonProfile
from eidolon_sdk.biz.speaker_identity import (
    CohortReference,
    DurationBand,
    RecognitionCalibration,
    SpeakerCandidate,
    SpeakerCandidates,
    SpeakerRecognitionPolicy,
)
from eidolon_sdk.biz.voiceprint import (
    CalibrationCrop,
    EnrollmentSample,
    ExtractedSample,
    GenuineScore,
    SpeakerComparisonResult,
    SpeakerSimilarity,
    TemplateCalibration,
    VoiceTemplate,
)

from eidolon.livekit.common.audio_evidence import AudioEvidence
from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognizer
from eidolon.livekit.plugins.speaker_verification import calibration as cal
from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    ReferenceSpeakerExtractor,
)

IDENTITY = ("m", "a" * 64, "v")
BANDS = [
    DurationBand(name="short", min_speech_ms=600, floor=0.3, margin=0.1),
    DurationBand(name="medium", min_speech_ms=1200, floor=0.3, margin=0.08),
    DurationBand(name="long", min_speech_ms=2500, floor=0.3, margin=0.06),
]


def unit(*values):
    v = np.asarray(values, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def write_cohort(path, vectors, speech_ms, speakers, identity=IDENTITY):
    with open(path, "wb") as f:
        np.savez(f, vectors=np.asarray(vectors, dtype=np.float32),
                 speech_ms=np.asarray(speech_ms), speakers=np.asarray(speakers),
                 model_id=np.array(identity[0]), weights_hash=np.array(identity[1]),
                 preprocessing_version=np.array(identity[2]))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy(sha="0" * 64, speakers=10, **change):
    calibration = RecognitionCalibration(
        version="c", max_false_accept=0.005, min_genuine_acceptance=0.8, min_genuine_scores=4,
        bands=BANDS, cohort=CohortReference(path="cohort.npz", sha256=sha, speakers=speakers))
    return SpeakerRecognitionPolicy(**({
        "version": "v3", "model_id": IDENTITY[0], "weights_hash": IDENTITY[1],
        "preprocessing_version": IDENTITY[2], "threshold": 0.55, "min_person_margin": 0.1,
        "min_speech_ms": 600, "calibration": calibration} | change))


def spread_cohort(tmp_path, n=400, speakers=10, around=(0.0, 1.0, 0.0)):
    """Impostors spread around an axis far from the Persons below."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(around, 0.35, size=(n, 3))
    speech = np.tile([900, 1800, 3000], n // 3 + 1)[:n]
    sha = write_cohort(tmp_path / "cohort.npz", vectors, speech, np.arange(n) % speakers)
    p = policy(sha=sha, speakers=speakers)
    return p, cal.load_cohort(tmp_path, p)


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


# ---------------------------------------------------------------- thresholds


def test_lowest_threshold_admits_at_most_the_allowed_false_accepts():
    scores = np.arange(1000) / 1000
    t = cal._lowest_threshold(scores, 0.005)
    assert (scores >= t).sum() == 5
    assert (scores >= np.nextafter(t, -1)).sum() > 5 or t == scores[994]
    assert cal._lowest_threshold(np.array([]), 0.005) == 1.0


def calibrated_template(vector, genuine):
    return TemplateCalibration(
        crops=[CalibrationCrop(speech_ms=ms, vector=vector) for ms, _ in genuine],
        genuine=[GenuineScore(speech_ms=ms, similarity=s) for ms, s in genuine])


def test_each_band_gets_the_lowest_threshold_within_the_false_accept_target(tmp_path):
    p, cohort = spread_cohort(tmp_path)
    own = calibrated_template(unit(1, 0, 0), [(900, 0.7)] * 4 + [(1800, 0.8)] * 4 + [(3000, 0.85)] * 4)
    table = cal.person_thresholds(p, cohort, [unit(1, 0, 0)], [own], [])
    for band in BANDS:
        entry = table[band.name]
        assert entry.threshold >= band.floor
        assert entry.estimated_false_accept <= 0.005
        assert entry.cohort_scores > 0
        assert entry.direct and entry.genuine_acceptance == 1.0


def test_a_household_voice_pushes_its_neighbour_threshold_above_it(tmp_path):
    p, cohort = spread_cohort(tmp_path)
    own = calibrated_template(unit(1, 0, 0), [(900, 0.9)] * 4)
    sibling = CalibrationCrop(speech_ms=900, vector=unit(1, 0.45, 0))
    alone = cal.person_thresholds(p, cohort, [unit(1, 0, 0)], [own], [])
    beside = cal.person_thresholds(p, cohort, [unit(1, 0, 0)], [own], [sibling])
    similar = float(np.dot(unit(1, 0, 0), sibling.vector))
    assert alone["short"].threshold < similar < beside["short"].threshold
    assert beside["short"].household_max == pytest.approx(similar, abs=1e-5)
    assert beside["long"].household_max is None  # other bands have no such window


def test_a_band_that_rejects_too_many_own_windows_only_extends_tracks(tmp_path):
    p, cohort = spread_cohort(tmp_path)
    own = calibrated_template(unit(1, 0, 0), [(900, 0.2)] * 3 + [(900, 0.95)] + [(3000, 0.9)] * 4)
    table = cal.person_thresholds(p, cohort, [unit(1, 0, 0)], [own], [])
    assert table["short"].genuine_acceptance == 0.25 and not table["short"].direct
    assert table["long"].direct
    # Too few windows to judge a band: it does not decide either.
    assert table["medium"].genuine_acceptance is None and not table["medium"].direct


def test_a_person_registered_before_windows_existed_decides_as_before(tmp_path):
    p, cohort = spread_cohort(tmp_path)
    table = cal.person_thresholds(p, cohort, [unit(1, 0, 0)], [None], [])
    assert all(e.direct and e.threshold >= p.threshold for e in table.values())


def test_impostors_close_to_the_person_raise_the_threshold(tmp_path):
    near_p, near = spread_cohort(tmp_path, around=(1.0, 0.6, 0.0))
    far_dir = tmp_path / "far"
    far_dir.mkdir()
    far_p, far = spread_cohort(far_dir)
    own = calibrated_template(unit(1, 0, 0), [(3000, 0.95)] * 4)
    assert (cal.person_thresholds(near_p, near, [unit(1, 0, 0)], [own], [])["long"].threshold
            > cal.person_thresholds(far_p, far, [unit(1, 0, 0)], [own], [])["long"].threshold)


# ---------------------------------------------------------------- decision


def entry(threshold, direct=True):
    return {b.name: cal.BandThreshold(b.name, threshold, direct, 1.0, 8, 0.0, 100, None)
            for b in BANDS}


@pytest.mark.parametrize("scores,tables,outcome,person", [
    ({"a": 0.7}, {"a": entry(0.5)}, "identified", "a"),
    ({"a": 0.4}, {"a": entry(0.5)}, "below", None),
    ({"a": 0.7, "b": 0.65}, {"a": entry(0.5), "b": entry(0.5)}, "ambiguous", None),
    ({"a": 0.7, "b": 0.5}, {"a": entry(0.5), "b": entry(0.6)}, "identified", "a"),
    ({"a": 0.7}, {"a": entry(0.5, direct=False)}, "not_direct", None),
    # A Person over its own lower threshold never wins against a higher scorer.
    ({"a": 0.58, "b": 0.6}, {"a": entry(0.5), "b": entry(0.65)}, "ambiguous", None),
    ({}, {}, "below", None),
])
def test_decide(scores, tables, outcome, person):
    decision = cal.decide(scores, tables, BANDS[2])
    assert (decision.outcome, decision.person_id) == (outcome, person)


# ---------------------------------------------------------------- cohort file


def test_the_cohort_is_exactly_the_file_the_policy_names(tmp_path):
    sha = write_cohort(tmp_path / "cohort.npz", [unit(0, 1, 0)] * 2, [900, 3000], [0, 1])
    assert cal.Cohort.load(tmp_path, policy(sha=sha, speakers=2).calibration.cohort).vectors.shape == (2, 3)
    with pytest.raises(cal.CohortUnavailable, match="cohort_invalid"):
        cal.Cohort.load(tmp_path, policy(sha="1" * 64, speakers=2).calibration.cohort)
    with pytest.raises(cal.CohortUnavailable, match="cohort_invalid"):
        cal.Cohort.load(tmp_path, policy(sha=sha, speakers=3).calibration.cohort)
    with pytest.raises(cal.CohortUnavailable, match="cohort_missing"):
        cal.Cohort.load(tmp_path / "elsewhere", policy(sha=sha, speakers=2).calibration.cohort)


def test_a_cohort_from_another_model_is_refused(tmp_path):
    sha = write_cohort(tmp_path / "cohort.npz", [unit(0, 1, 0)], [900], [0], identity=("x", "b" * 64, "v"))
    with pytest.raises(cal.CohortUnavailable, match="cohort_model_mismatch"):
        cal.load_cohort(tmp_path, policy(sha=sha, speakers=1))


def test_a_pickled_cohort_is_not_loaded(tmp_path):
    path = tmp_path / "cohort.npz"
    with open(path, "wb") as f:
        np.savez(f, vectors=np.array([object()], dtype=object), speech_ms=[900], speakers=[0],
                 model_id=np.array("m"), weights_hash=np.array("a" * 64),
                 preprocessing_version=np.array("v"))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(cal.CohortUnavailable, match="cohort_invalid"):
        cal.Cohort.load(tmp_path, policy(sha=sha, speakers=1).calibration.cohort)


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


# ---------------------------------------------------------------- recognizer


PROBE = ExtractedSample(model_id=IDENTITY[0], weights_hash=IDENTITY[1], preprocessing_version=IDENTITY[2],
                        vector=unit(1, 0, 0), speech_ms=3000, quality={"overlap_detected": 0})


def evidence():
    return AudioEvidence(source="mic", generation=1, candidate="c", sample_rate=16000, start=0,
                         end=48000, complete=True, reason=None, pcm=b"\x01\x00" * 48000)


def port(score, genuine):
    template = VoiceTemplate(profile_id="v", revision=1, sample=PROBE.model_copy(),
                             calibration=calibrated_template(unit(1, 0, 0), genuine))
    snapshot = SpeakerCandidates(owner_id="o", revision="b" * 64, persons=[SpeakerCandidate(
        person_id="p", person_revision=1, display_name="P", profile=PersonProfile(), templates=[template])])

    async def compare(probe, templates):
        return SpeakerComparisonResult(model_id=probe.model_id, weights_hash=probe.weights_hash,
            preprocessing_version=probe.preprocessing_version,
            scores=[SpeakerSimilarity(profile_id=t.profile_id, revision=t.revision, status="scored",
                                      similarity=score) for t in templates])

    return SimpleNamespace(candidates=AsyncMock(return_value=snapshot), extract=AsyncMock(return_value=PROBE),
                           compare=AsyncMock(side_effect=compare), record_candidate_evidence=AsyncMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("score,genuine,status,reason", [
    (0.9, [(3000, 0.9)] * 4, "identified", "voice_match"),
    (0.2, [(3000, 0.9)] * 4, "unknown", "no_match"),
    (0.9, [(3000, 0.31)] * 4, "uncertain", "duration_band_not_decisive"),
])
async def test_the_recognizer_decides_with_the_persons_own_band(tmp_path, score, genuine, status, reason):
    p, cohort = spread_cohort(tmp_path)
    result = await SpeakerRecognizer(port(score, genuine), p, cohort).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t")
    assert (result.status, result.reason) == (status, reason)
    assert result.decision_version == "v3" or status == "uncertain"


@pytest.mark.asyncio
async def test_a_calibrated_policy_without_its_cohort_turns_recognition_off(tmp_path):
    p, _ = spread_cohort(tmp_path)
    result = await SpeakerRecognizer(port(0.9, [(3000, 0.9)] * 4), p, None).recognize(
        evidence(), owner_id="o", session_id="s", turn_id="t")
    assert (result.status, result.reason) == ("unavailable", "recognition_policy_missing")


@pytest.mark.asyncio
async def test_tables_are_recomputed_only_when_the_household_changes(tmp_path, monkeypatch):
    p, cohort = spread_cohort(tmp_path)
    calls = []
    real = cal.household_tables
    import eidolon.livekit.agent.session.speaker_recognition as module
    monkeypatch.setattr(module, "household_tables", lambda *a: calls.append(1) or real(*a))
    recognizer = SpeakerRecognizer(port(0.9, [(3000, 0.9)] * 4), p, cohort)
    for turn in ("t1", "t2"):
        await recognizer.recognize(evidence(), owner_id="o", session_id="s", turn_id=turn)
    assert len(calls) == 1
