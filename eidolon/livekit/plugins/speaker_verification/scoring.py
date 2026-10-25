"""Shared numerical speaker comparison; no identity or authorization verdict."""

import math
from eidolon_sdk.biz.voiceprint import (
    SpeakerComparisonRequest,
    SpeakerComparisonResult,
    SpeakerSimilarity,
)


def embedding_identity(sample):
    return (sample.model_id, sample.weights_hash, sample.preprocessing_version, len(sample.vector))


def cosine(a, b):
    if embedding_identity(a) != embedding_identity(b):
        raise ValueError("incompatible_embedding_versions")
    return max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a.vector, b.vector, strict=True))))


def compare_speakers(request: SpeakerComparisonRequest) -> SpeakerComparisonResult:
    probe = request.probe
    scores = []
    for template in request.templates:
        compatible = embedding_identity(template.sample) == embedding_identity(probe)
        scores.append(
            SpeakerSimilarity(
                profile_id=template.profile_id,
                revision=template.revision,
                status="scored" if compatible else "incompatible",
                similarity=cosine(template.sample, probe) if compatible else None,
            )
        )
    return SpeakerComparisonResult(
        model_id=probe.model_id,
        weights_hash=probe.weights_hash,
        preprocessing_version=probe.preprocessing_version,
        scores=scores,
    )
