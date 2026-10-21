"""Prepare versioned enrollment evidence; no calibrated identity decisions yet."""

import math
from itertools import combinations

from eidolon_sdk.biz.voiceprint import (
    EnrollmentEvaluation,
    EnrollmentEvaluationRequest,
    ExtractedSample,
    enrollment_evidence_digest,
)


def evaluate_enrollment(request: EnrollmentEvaluationRequest) -> EnrollmentEvaluation:
    samples = list(request.samples.values())

    def identity(x):
        return (x.model_id, x.weights_hash, x.preprocessing_version, len(x.vector))

    if any(identity(s) != identity(request.trial) for s in samples):
        raise ValueError("incompatible_embedding_versions")

    def cosine(a, b):
        return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a.vector, b.vector))))

    mean = [
        math.fsum(s.vector[i] for s in samples) / len(samples)
        for i in range(len(request.trial.vector))
    ]
    template = ExtractedSample(**{**samples[0].model_dump(), "vector": mean})
    return EnrollmentEvaluation(
        evidence_digest=enrollment_evidence_digest(request.samples),
        recording_sha256=request.recording_sha256,
        template=template,
        trial=request.trial,
        registration_min_similarity=min(cosine(a, b) for a, b in combinations(samples, 2)),
        trial_min_similarity=min(cosine(s, request.trial) for s in samples),
    )
