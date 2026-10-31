"""Prepare versioned enrollment evidence and the Person's free same-person scores."""

import math
from itertools import combinations

from eidolon_sdk.biz.voiceprint import (
    EnrollmentEvaluation,
    EnrollmentEvaluationRequest,
    ExtractedSample,
    enrollment_evidence_digest,
)

from .calibration import enrollment_calibration
from .scoring import embedding_identity, cosine


def evaluate_enrollment(request: EnrollmentEvaluationRequest) -> EnrollmentEvaluation:
    samples = list(request.samples.values())
    if any(s.speech_ms < 1500 for s in [*samples, request.trial]):
        raise ValueError("insufficient_enrollment_speech")

    if any(embedding_identity(s) != embedding_identity(request.trial) for s in samples):
        raise ValueError("incompatible_embedding_versions")

    mean = [
        math.fsum(s.vector[i] for s in samples) / len(samples)
        for i in range(len(request.trial.vector))
    ]
    template = ExtractedSample(**{**samples[0].sample().model_dump(), "vector": mean})
    trial = request.trial.sample()
    return EnrollmentEvaluation(
        evidence_digest=enrollment_evidence_digest(request.samples),
        recording_sha256=request.recording_sha256,
        template=template,
        trial=trial,
        registration_min_similarity=min(cosine(a, b) for a, b in combinations(samples, 2)),
        trial_min_similarity=min(cosine(s, trial) for s in samples),
        calibration=enrollment_calibration(samples, request.trial, template),
    )
