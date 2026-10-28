"""Bounded Person recognition over existing Data and shared extraction ports.

No file database, worker model, identity cache, admission decision or Realm
selection. The returned identity is valid for exactly one audio evidence span.
"""

import time
import hashlib
from urllib.parse import quote

from eidolon_sdk.biz.candidate_evidence import CANDIDATE_MIN_SPEECH_MS
from eidolon_sdk.biz.speaker_identity import (
    MAX_RECOGNITION_PERSONS,
    SpeakerCandidates,
    SpeakerEvidenceSource,
    SpeakerIdentityObservation,
    SpeakerRecognitionPolicy,
)
from eidolon_sdk.biz.voiceprint import SpeakerComparisonRequest, SpeakerAudioQuality
from .speaker_extraction_client import SpeakerExtractionClient, SpeakerExtractionRejected
from eidolon.livekit.plugins.speaker_verification.scoring import compare_speakers
from eidolon_sdk.core.http.errors import ServiceUpstreamError
from eidolon_sdk.core.http.client import ServiceHTTPClient, authorization_header


class SpeakerRecognitionClient(ServiceHTTPClient):
    def __init__(self, http, *, data_url, data_token, provider_url, provider_token):
        super().__init__(http, data_url, default_headers=authorization_header(data_token))
        self._provider = SpeakerExtractionClient(http, provider_url, token=provider_token)

    async def candidates(self, owner_id):
        response = await self._request(
            "GET",
            "/api/companion-authority/v1/owners/"
            + quote(owner_id, safe="")
            + "/voiceprints/recognition-candidates",
            timeout=1,
        )
        result = SpeakerCandidates.model_validate(response.json())
        if result.owner_id != owner_id:
            raise ValueError("candidate owner mismatch")
        return result

    async def discover_candidate(self, request):
        from eidolon_sdk.biz.person import PersonView
        response = await self._request("POST", "/api/companion-authority/v1/owners/"
            + quote(request.observation.owner_id, safe="") + "/voiceprints/candidate-discovery",
            json=request.model_dump(mode="json"), timeout=1)
        person = PersonView.model_validate(response.json())
        if person.owner_id != request.observation.owner_id:
            raise ValueError("discovery owner mismatch")
        return person

    async def record_candidate_evidence(self, request):
        await self._request("POST", "/api/companion-authority/v1/owners/"
            + quote(request.observation.owner_id, safe="") + "/voiceprints/candidate-evidence",
            json=request.model_dump(mode="json"), timeout=1)

    async def extract(self, pcm, *, min_speech_ms):
        return await self._provider.extract(pcm, purpose="recognition", min_speech_ms=min_speech_ms)

    async def compare(self, probe, templates):
        # Reuse the Provider's pure cosine scorer; no RPC/model per candidate.
        return compare_speakers(SpeakerComparisonRequest(probe=probe, templates=templates))


class SpeakerRecognizer:
    def __init__(self, client, policy: SpeakerRecognitionPolicy | None):
        self.client, self.policy = client, policy
        self._busy = False

    async def recognize(self, evidence, *, owner_id, session_id, turn_id):
        now = time.time()
        base = dict(
            owner_id=owner_id,
            session_id=session_id,
            turn_id=turn_id,
            issued_at=now,
            expires_at=now + 10,
            confidence="none",
        )

        def outcome(status, reason, **extra):
            return SpeakerIdentityObservation(**(base | extra), status=status, reason=reason)

        if evidence is None or not evidence.complete:
            return outcome("uncertain", "audio_evidence_unavailable")
        if self.policy is None:
            return outcome("unavailable", "recognition_policy_missing")
        if evidence.sample_rate != 16000 or len(evidence.pcm) % 2 or len(evidence.pcm) > 960000:
            return outcome("unavailable", "invalid_audio_format")
        if (evidence.start < 0 or evidence.end <= evidence.start
                or (evidence.end - evidence.start) * 2 != len(evidence.pcm)
                or (evidence.session_id and evidence.session_id != session_id)):
            return outcome("unavailable", "invalid_audio_evidence")
        try:
            base["source"] = SpeakerEvidenceSource(
                request_id=turn_id, session_id=session_id, input_id=evidence.source,
                generation=evidence.generation, sample_rate=evidence.sample_rate,
                start_sample=evidence.start, end_sample=evidence.end,
            )
        except ValueError:
            return outcome("unavailable", "invalid_audio_evidence")
        base["audio_quality"] = SpeakerAudioQuality(
            audio_ms=len(evidence.pcm) // 32, required_speech_ms=self.policy.min_speech_ms,
        )
        if len(evidence.pcm) < self.policy.min_speech_ms * 32:
            return outcome("uncertain", "audio_too_short")
        if self._busy:
            return outcome("unavailable", "recognizer_busy")
        self._busy = True
        try:
            snapshot = await self.client.candidates(owner_id)
            if not snapshot.persons and snapshot.discovery_policy is None:
                return outcome("uncertain", "no_enrolled_voices")
            probe = await self.client.extract(evidence.pcm, min_speech_ms=self.policy.min_speech_ms)
            if probe.speech_ms > len(evidence.pcm) // 32:
                return outcome("unavailable", "invalid_extraction_response")
            base["audio_quality"] = SpeakerAudioQuality(
                audio_ms=len(evidence.pcm) // 32, required_speech_ms=self.policy.min_speech_ms,
                speech_ms=probe.speech_ms,
                clipping=probe.quality.get("clipping"), rms=probe.quality.get("rms"),
                overlap_detected=probe.quality.get("overlap_detected"),
            )
            if (base["audio_quality"].overlap_detected is not None
                    and base["audio_quality"].overlap_detected > 0):
                return outcome("uncertain", "overlapping_speech")
            if (probe.model_id, probe.weights_hash, probe.preprocessing_version) != (
                self.policy.model_id,
                self.policy.weights_hash,
                self.policy.preprocessing_version,
            ):
                return outcome("unavailable", "policy_model_mismatch")
            if probe.speech_ms < self.policy.min_speech_ms:
                return outcome("uncertain", "speech_too_short")
            per_person = []
            for person in snapshot.persons:
                scored = await self.client.compare(probe, person.templates)
                if (scored.model_id, scored.weights_hash, scored.preprocessing_version) != (
                    probe.model_id,
                    probe.weights_hash,
                    probe.preprocessing_version,
                ):
                    return outcome("unavailable", "invalid_comparison")
                expected = {(t.profile_id, t.revision) for t in person.templates}
                if (
                    len(scored.scores) != len(expected)
                    or {(s.profile_id, s.revision) for s in scored.scores} != expected
                ):
                    return outcome("unavailable", "invalid_comparison")
                if any(s.status != "scored" for s in scored.scores):
                    return outcome("unavailable", "model_version_mismatch")
                best = max(scored.scores, key=lambda s: s.similarity)
                per_person.append((best.similarity, person, best))
            # Re-read authority after inference: changes never inherit a cached match.
            current = await self.client.candidates(owner_id)
            if current.revision != snapshot.revision:
                return outcome("uncertain", "candidates_changed")
            per_person.sort(key=lambda row: row[0], reverse=True)
            similarity = per_person[0][0] if per_person else None
            version = dict(
                model_id=probe.model_id,
                weights_hash=probe.weights_hash,
                preprocessing_version=probe.preprocessing_version,
                scoring_version="normalized-template-cosine-v1",
                decision_version=self.policy.version,
                confidence=self.policy.confidence,
                cosine_similarity=similarity,
            )
            if not per_person:
                if probe.speech_ms < self.policy.min_unknown_speech_ms:
                    return outcome("uncertain", "short_speech_inconclusive", **version)
                unknown = outcome("unknown", "no_enrolled_voices", **version)
                await self._discover(snapshot, probe, unknown, evidence)
                return unknown
            similarity, person, best = per_person[0]
            if similarity < self.policy.threshold:
                if probe.speech_ms < self.policy.min_unknown_speech_ms:
                    return outcome("uncertain", "short_speech_inconclusive", **version)
                unknown = outcome("unknown", "no_match", **version)
                await self._discover(snapshot, probe, unknown, evidence)
                return unknown
            if (
                len(per_person) > 1
                and similarity - per_person[1][0] < self.policy.min_person_margin
            ):
                return outcome("uncertain", "ambiguous_candidates")
            identified = outcome(
                "identified",
                "voice_match",
                **version,
                person_id=person.person_id,
                person_revision=person.person_revision,
                maturity=person.maturity,
                is_named=person.is_named,
                profile_id=best.profile_id,
                profile_revision=best.revision,
                display_name=person.profile.preferred_name or person.display_name,
                profile=person.profile,
            )
            if person.maturity == "candidate" and probe.speech_ms >= CANDIDATE_MIN_SPEECH_MS:
                from eidolon_sdk.biz.candidate_evidence import CandidateEvidence
                try:
                    await self.client.record_candidate_evidence(CandidateEvidence(
                        snapshot_revision=snapshot.revision,
                        audio_sha256=hashlib.sha256(evidence.pcm).hexdigest(), observation=identified))
                except (ServiceUpstreamError, ValueError, OSError):
                    # Receipt persistence never manufactures an identity or permission.
                    # A rejected/expired receipt leaves the existing match unchanged.
                    pass
            return identified
        except SpeakerExtractionRejected as exc:
            if exc.quality is not None:
                base["audio_quality"] = exc.quality
            return outcome("uncertain" if exc.audio_quality_failure else "unavailable", exc.reason)
        except ServiceUpstreamError as exc:
            return outcome(
                "unavailable",
                "invalid_extraction_response" if exc.status_code == 422 else "dependency_unavailable",
            )
        except (ValueError, OSError):
            return outcome("unavailable", "invalid_dependency_result")
        finally:
            self._busy = False

    async def _discover(self, snapshot, probe, observation, evidence):
        from eidolon_sdk.biz.candidate_evidence import CandidateDiscovery
        policy = snapshot.discovery_policy
        if (policy is None or policy.recognition != self.policy
                or len(snapshot.persons) >= MAX_RECOGNITION_PERSONS):
            return
        if (probe.speech_ms < self.policy.min_unknown_speech_ms
                or (observation.cosine_similarity is not None
                    and observation.cosine_similarity > policy.max_existing_similarity)):
            return
        try:
            request = CandidateDiscovery(policy_version=policy.version,
                snapshot_revision=snapshot.revision,
                audio_sha256=hashlib.sha256(evidence.pcm).hexdigest(), observation=observation, sample=probe)
            await self.client.discover_candidate(request)
        except (ServiceUpstreamError, ValueError, OSError):
            # Missing single-speaker quality, conflict, timeout or full pool
            # cannot turn an unknown observation into an identified speaker.
            return
