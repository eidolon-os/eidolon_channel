"""Evaluate saved enrollment numbers against a published model release.

This produces evidence for Data; it neither mutates nor grants a Companion policy.
"""

from eidolon_sdk.biz.voiceprint import VoicePolicyApproval, VoicePolicyEvidence
from eidolon_sdk.biz.speaker_calibration import SpeakerAdmissionCalibration


def approve_policy(evidence: VoicePolicyEvidence, calibration: SpeakerAdmissionCalibration):
    if len(evidence.profiles) > calibration.max_allowed_speakers:
        raise ValueError("ALLOWLIST_NOT_CALIBRATED")
    for profile in evidence.profiles:
        evaluation = profile.evaluation
        if not calibration.compatible(evaluation.template) or not calibration.compatible(
            evaluation.trial
        ):
            raise ValueError("ENROLLMENT_VERSION_UNSUPPORTED")
        if (
            evaluation.registration_min_similarity < calibration.registration_min_similarity
            or evaluation.trial_min_similarity < calibration.trial_min_similarity
        ):
            raise ValueError("ENROLLMENT_VALIDATION_REQUIRED")
    return VoicePolicyApproval(
        calibration_digest=calibration.digest, evidence_digest=evidence.digest()
    )
