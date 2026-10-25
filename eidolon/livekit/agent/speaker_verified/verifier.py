"""Calibrated permission adapter over current Data and shared Channel extraction.

A calibration describes a model/scoring release, never personal permissions or
a device binding. Missing/incompatible evidence cannot become permission.
"""

import time

from eidolon_sdk.core.http.errors import ServiceUpstreamError
from eidolon_sdk.biz.voiceprint import SpeakerComparisonRequest
from eidolon_sdk.biz.speaker_calibration import SpeakerAdmissionCalibration

from ...common.audio_evidence import AudioEvidence

from ...plugins.speaker_verification.scoring import compare_speakers
from .orchestration import SpeakerDecision


class CalibratedSpeakerVerifier:
    def __init__(
        self, *, data, extractor, calibration: SpeakerAdmissionCalibration, owner_id, companion_id,
        observe_result=None,
    ):
        if not owner_id or not companion_id:
            raise ValueError("SPEAKER_VERIFIED_SCOPE_REQUIRED")
        self.calibration = calibration.model_copy(deep=True)
        self._data, self._extractor = data, extractor
        self._owner_id, self._companion_id = owner_id, companion_id
        self._observe_result = observe_result

    async def __call__(self, evidence: AudioEvidence) -> SpeakerDecision:
        started = time.monotonic()
        if (
            not evidence.complete
            or evidence.reason
            or evidence.sample_rate != 16000
            or not self.calibration.min_audio_ms * 32
            <= len(evidence.pcm)
            <= self.calibration.max_audio_ms * 32
            or len(evidence.pcm) % 2
        ):
            return SpeakerDecision("inconclusive")
        snapshot = await self._data.get_voiceprint_snapshot(self._companion_id)
        if (
            snapshot.owner_id != self._owner_id
            or snapshot.companion_id != self._companion_id
            or not self.calibration.supports(snapshot)
        ):
            return SpeakerDecision("unavailable")
        try:
            ready = await self._extractor.readiness()
            if not ready.admission_ready or ready.calibration_digest != self.calibration.digest:
                return SpeakerDecision("unavailable")
            probe = await self._extractor.extract(evidence.pcm)
        except ServiceUpstreamError as exc:
            return SpeakerDecision("inconclusive" if exc.status_code == 422 else "unavailable")
        if not self.calibration.compatible(probe):
            return SpeakerDecision("unavailable")
        result = compare_speakers(
            SpeakerComparisonRequest(probe=probe, templates=snapshot.templates)
        )
        # A partially incompatible allowlist has not been calibrated either.
        if any(score.status != "scored" for score in result.scores):
            return SpeakerDecision("unavailable")
        best = max(result.scores, key=lambda score: score.similarity)
        matched = best.similarity >= self.calibration.threshold
        if self._observe_result is not None:
            self._observe_result({
                "status": "match" if matched else "non_match",
                "similarity": best.similarity,
                "threshold": self.calibration.threshold,
                "audio_ms": len(evidence.pcm) / 32,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            })
        if not matched:
            return SpeakerDecision("non_match")
        return SpeakerDecision(
            "match",
            best.profile_id,
            best.revision,
            self.calibration.digest,
            policy_revision=snapshot.revision,
        )
