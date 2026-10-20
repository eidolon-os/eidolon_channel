"""Companion-scoped verification. Data owns policy; Channel owns acoustic evidence."""

from __future__ import annotations
import asyncio
import math
import logging
from dataclasses import dataclass

from eidolon.livekit.common.speaker_verification import SpeakerSignal
from eidolon.livekit.plugins.speaker_verification.extractor import InsufficientSpeech

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AdmissionEvidence:
    signal: SpeakerSignal
    commit_allowed: bool
    commit_reason: str
    companion_id: str
    policy_revision: int
    cached: bool = False


class CompanionVoiceAdmission:
    def __init__(self, authority, extractor, *, threshold=0.65):
        self.authority = authority
        self.extractor = extractor
        # Conservative pilot operating point, not a calibrated FAR guarantee.
        self.threshold = threshold

    async def verify(self, companion_id, audio, sample_rate=16000):
        revision = -1
        try:
            snapshot = await asyncio.wait_for(
                self.authority.get_voiceprint_snapshot(companion_id), 3
            )
            revision = snapshot.revision
            if snapshot.mode == "off":
                return self._result(companion_id, revision, True, "voiceprint_off")
            if not audio:
                return self._result(companion_id, revision, False, "voiceprint_insufficient")
            if not snapshot.templates:
                return self._result(companion_id, revision, False, "voiceprint_no_templates")
            sample = await self.extractor.extract(audio, sample_rate)
            scores = []
            for template in snapshot.templates:
                enrolled = template.sample
                if (
                    enrolled.model_id,
                    enrolled.weights_hash,
                    enrolled.preprocessing_version,
                    len(enrolled.vector),
                ) != (
                    sample.model_id,
                    sample.weights_hash,
                    sample.preprocessing_version,
                    len(sample.vector),
                ):
                    continue
                scores.append(
                    (
                        math.fsum(a * b for a, b in zip(enrolled.vector, sample.vector)),
                        template.profile_id,
                    )
                )
            if not scores:
                return self._result(companion_id, revision, False, "voiceprint_model_mismatch")
            score, profile = max(scores)
            allowed = score >= self.threshold
            return self._result(
                companion_id,
                revision,
                allowed,
                "voiceprint_accepted" if allowed else "voiceprint_rejected",
                profile if allowed else None,
                score,
            )
        except InsufficientSpeech as exc:
            logger.info("voiceprint extraction refused code=%s audio_ms=%d sample_rate=%d",
                        exc.code, len(audio) * 500 // max(sample_rate, 1), sample_rate)
            return self._result(companion_id, revision, False, "voiceprint_insufficient")
        except Exception as exc:
            logger.warning("voiceprint unavailable error_type=%s", type(exc).__name__)
            return self._result(companion_id, revision, False, "voiceprint_unavailable")

    async def authorize(self, companion_id, result):
        if result is None:
            # A missing acoustic result is allowed only by an authoritative off
            # policy, never by provider absence or an uninitialised observer.
            result = await self.verify(companion_id, b"")
        if not result.commit_allowed or getattr(result, "companion_id", None) != companion_id:
            return False
        try:
            return await asyncio.wait_for(
                self.authority.admit_voiceprint(
                    companion_id, result.policy_revision, result.signal.profile_id
                ),
                3,
            )
        except Exception:
            return False

    @staticmethod
    def _result(companion, revision, allowed, reason, profile=None, score=None):
        logger.info("voiceprint decision allowed=%s reason=%s score=%s revision=%s",
                    allowed, reason, score, revision)
        return AdmissionEvidence(
            SpeakerSignal(
                provider="companion_policy",
                model="versioned",
                known=profile is not None,
                profile_id=profile,
                score=score,
            ),
            allowed,
            reason,
            companion,
            revision,
        )
