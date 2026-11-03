"""Calibrated identity evidence for one utterance over Data's Persons (design §3.3).

No file database, worker model, identity cache, admission decision or Realm
selection. Each compared Person gets a calibrated log-likelihood ratio and an
open-set posterior; each use's decision records who, if anyone, meets that
use's threshold in the condition of the leading comparison (phone template
against the phone, a device's own template against it, or across devices).
The observation is valid for exactly one audio evidence span.

The utterance's segment continues the conversation track of its own voice,
or opens one (design §3.2), and the evidence is the track's: one utterance
across devices is weak, several of the same voice are not.

A turn whose ``personal_memory`` decision names a Person is that turn's
attribution: its verdict goes to Data in the background, never delaying the
observation (design §3.5). It carries no observation template: segments that
meet the bar on their own are the ones the phone template already
recognises, and templates pooled from them added no coverage on far devices
(design §3.2; ``evals/speaker-verification/tracks-20261102/pooled_eval.py``).

The household guard (design §3.3 L7): members' registration windows, scored
against each other's phone templates, are the only live different-speaker
evidence on this household's channel. When one of them would already meet a
use's threshold, that use in ``near_same`` also needs an LLR above the
highest of them plus the policy's margin. It only ever makes a use stricter.
"""

import time
import math
import asyncio
import logging
import hashlib
from urllib.parse import quote

from eidolon_sdk.biz.candidate_evidence import CANDIDATE_MIN_SPEECH_MS
from eidolon_sdk.biz.speaker_identity import (
    MAX_EVIDENCE,
    MAX_RECOGNITION_PERSONS,
    PERSON_USES,
    IdentityDecisions,
    IdentityPolicy,
    PersonEvidence,
    SpeakerCandidates,
    SpeakerEvidenceSource,
    SpeakerIdentityObservation,
    comparison_condition,
)
from eidolon_sdk.biz.voice_attribution import turn_verdict
from eidolon_sdk.biz.voiceprint import SpeakerAudioQuality
from .speaker_extraction_client import SpeakerExtractionClient, SpeakerExtractionRejected
from eidolon.livekit.plugins.speaker_verification.evidence import posteriors
from eidolon.livekit.plugins.speaker_verification.tracks import Tracks
from eidolon_sdk.core.http.errors import ServiceUpstreamError
from eidolon_sdk.core.http.client import ServiceHTTPClient, authorization_header


class SpeakerRecognitionClient(ServiceHTTPClient):
    def __init__(self, http, *, data_url, data_token, provider_url, provider_token):
        super().__init__(http, data_url, default_headers=authorization_header(data_token))
        self._provider = SpeakerExtractionClient(http, provider_url, token=provider_token)

    async def candidates(self, owner_id, *, channel=None, session_id=None):
        params = {k: v for k, v in (("channel", channel), ("session_id", session_id)) if v}
        response = await self._request(
            "GET",
            "/api/companion-authority/v1/owners/"
            + quote(owner_id, safe="")
            + "/voiceprints/recognition-candidates",
            params=params or None,
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

    async def submit_verdict(self, verdict):
        from eidolon_sdk.biz.voice_attribution import VoiceAttribution
        response = await self._request("POST", "/api/companion-authority/v1/owners/"
            + quote(verdict.owner_id, safe="") + "/voiceprints/verdicts",
            json=verdict.model_dump(mode="json"), timeout=1)
        return VoiceAttribution.model_validate(response.json())

    async def record_candidate_evidence(self, request):
        await self._request("POST", "/api/companion-authority/v1/owners/"
            + quote(request.observation.owner_id, safe="") + "/voiceprints/candidate-evidence",
            json=request.model_dump(mode="json"), timeout=1)

    async def extract(self, pcm, *, min_speech_ms):
        return await self._provider.extract(pcm, purpose="recognition", min_speech_ms=min_speech_ms)


class SpeakerRecognizer:
    def __init__(self, client, policy: IdentityPolicy | None, backend=None, *, channel: str):
        # The policy's thresholds were set for exactly this calibration.
        if (policy is None or backend is None or backend.version != policy.backend_version
                or backend.identity != (policy.model_id, policy.weights_hash,
                                        policy.preprocessing_version)):
            policy = backend = None
        self.client, self.policy, self.backend, self.channel = client, policy, backend, channel
        self._busy = False
        self._templates = (None, None, {})
        self._tracks = Tracks()
        self._submissions = set()

    def _track_side(self, track, condition):
        """The track as a probe, over at most the segments ``condition`` was calibrated on."""
        vector, speech_ms, _ = track.probe(self.backend.max_track_segments[condition])
        return self.backend.side(vector, speech_ms, self.channel)

    def _template_sides(self, snapshot):
        # Deterministic in the snapshot, so its revision is the cache key.
        if self._templates[0] != snapshot.revision:
            sides = {
                person.person_id: [
                    (template, self.backend.side(template.sample.scored_vector(),
                                                 template.sample.speech_ms, template.channel))
                    for template in person.templates
                ]
                for person in snapshot.persons
            }
            self._templates = (snapshot.revision, sides, self._household_floors(snapshot, sides))
        return self._templates[1]

    def _household_floors(self, snapshot, sides):
        """Per use, the LLR a near_same leader must exceed in this household (design §3.3 L7)."""
        worst = None
        for person in snapshot.persons:
            for other in snapshot.persons:
                if other.person_id == person.person_id:
                    continue
                for template in other.templates:
                    for crop in (template.calibration.crops if template.calibration else []):
                        probe = self.backend.side(crop.vector, crop.speech_ms, template.channel)
                        for _, side in sides[person.person_id]:
                            if self.backend.condition(side, probe) == "near_same":
                                llr = self.backend.llr(side, probe)
                                worst = llr if worst is None else max(worst, llr)
        floors = {}
        if worst is not None:
            # Posterior of the other member, were they the only one enrolled.
            prior = self.policy.prior_unknown
            alone = 1 / (1 + prior / ((1 - prior) * math.exp(min(worst, 700))))
            thresholds = self.policy.thresholds("near_same")
            floors = {use: worst + self.policy.household_margin_llr for use in PERSON_USES
                      if getattr(thresholds, use) is not None and alone >= getattr(thresholds, use)}
        logging.getLogger(__name__).info(
            "Household guard revision=%s persons=%d worst_member_llr=%s raised=%s",
            snapshot.revision[:12], len(snapshot.persons),
            None if worst is None else round(worst, 3), sorted(floors) or None,
        )
        return floors

    async def recognize(self, evidence, *, owner_id, session_id, turn_id):
        now = time.time()
        base = dict(
            owner_id=owner_id,
            session_id=session_id,
            turn_id=turn_id,
            issued_at=now,
            expires_at=now + 10,
        )

        def outcome(status, reason):
            return SpeakerIdentityObservation(**base, status=status, reason=reason)

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
                start_sample=evidence.start, end_sample=evidence.end, channel=self.channel,
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
            scope = dict(channel=self.channel, session_id=session_id)
            snapshot = await self.client.candidates(owner_id, **scope)
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
            identity = (probe.model_id, probe.weights_hash, probe.preprocessing_version)
            if identity != self.backend.identity:
                return outcome("unavailable", "policy_model_mismatch")
            if probe.speech_ms < self.policy.min_speech_ms:
                return outcome("uncertain", "speech_too_short")
            if any((t.sample.model_id, t.sample.weights_hash, t.sample.preprocessing_version) != identity
                   for person in snapshot.persons for t in person.templates):
                return outcome("unavailable", "model_version_mismatch")
            segment = self.backend.side(probe.vector, probe.speech_ms, self.channel)
            same_device = comparison_condition(self.channel, self.channel)
            track = self._tracks.assign(
                probe.vector, probe.speech_ms,
                lambda t: self.backend.llr(self._track_side(t, same_device), segment),
                self.policy.track_continuation_llr,
            )
            base["track_id"] = track.track_id
            probes, best = {}, {}
            for person in snapshot.persons:
                for template, side in self._template_sides(snapshot)[person.person_id]:
                    condition = self.backend.condition(side, segment)
                    if condition not in probes:
                        probes[condition] = self._track_side(track, condition)
                    llr = self.backend.llr(side, probes[condition])
                    if person.person_id not in best or llr > best[person.person_id][0]:
                        best[person.person_id] = (llr, person, template, condition)
            # Re-read authority after inference: changes never inherit a cached match.
            current = await self.client.candidates(owner_id, **scope)
            if current.revision != snapshot.revision:
                return outcome("uncertain", "candidates_changed")
            observation = self._decide(base, best, probe.speech_ms)
            top = observation.evidence[0] if observation.evidence else None
            logging.getLogger(__name__).info(
                "Speaker evidence turn_id=%s persons=%d top=%s llr=%s posterior=%s unknown=%.4f "
                "decisions=%s channel=%s condition=%s speech_ms=%d track=%s segments=%d policy=%s",
                turn_id, len(best), top.person_id if top else None,
                round(top.llr, 3) if top else None, round(top.posterior, 4) if top else None,
                observation.posterior_unknown, observation.decisions.model_dump(exclude_defaults=True),
                self.channel, observation.condition, probe.speech_ms, track.track_id,
                len(track.segments), self.policy.version,
            )
            await self._record(snapshot, observation, probe, evidence)
            self._attribute(snapshot, observation)
            return observation
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

    def _decide(self, base, best, speech_ms):
        policy = self.policy
        priors = {pid: (1 - policy.prior_unknown) / len(best) for pid in best}
        post, unknown = posteriors({pid: row[0] for pid, row in best.items()}, priors,
                                   policy.prior_unknown)
        ranked = sorted(best, key=post.get, reverse=True)
        evidence = [
            PersonEvidence(
                person_id=pid, person_revision=best[pid][1].person_revision,
                maturity=best[pid][1].maturity, is_named=best[pid][1].is_named,
                profile_id=best[pid][2].profile_id, profile_revision=best[pid][2].revision,
                llr=best[pid][0], posterior=post[pid],
            )
            for pid in ranked[:MAX_EVIDENCE]
        ]
        # The leading comparison's condition decides; with nobody to compare,
        # the probe's own channel does.
        condition = (best[ranked[0]][3] if ranked
                     else comparison_condition(self.channel, self.channel))
        thresholds = policy.thresholds(condition)
        floors = self._templates[2] if condition == "near_same" else {}
        decisions = {}
        if ranked:
            leader = ranked[0]
            for use in PERSON_USES:
                threshold = getattr(thresholds, use)
                if threshold is None or post[leader] < threshold:
                    continue
                if use in floors and best[leader][0] < floors[use]:
                    continue
                if use == "candidate_evidence" and best[leader][1].maturity != "candidate":
                    continue
                decisions[use] = leader
        # A weak short probe is not evidence of someone new.
        decisions["discovery"] = (thresholds.discovery is not None and unknown >= thresholds.discovery
                                  and speech_ms >= policy.min_unknown_speech_ms)
        named = {}
        if "addressing" in decisions:
            person = best[decisions["addressing"]][1]
            named = dict(display_name=person.profile.preferred_name or person.display_name,
                         profile=person.profile)
        return SpeakerIdentityObservation(
            **base, **named, status="compared", reason="compared",
            model_id=self.backend.identity[0], weights_hash=self.backend.identity[1],
            preprocessing_version=self.backend.identity[2],
            backend_version=self.backend.version, policy_version=policy.version,
            evidence=evidence, posterior_unknown=unknown, condition=condition,
            decisions=IdentityDecisions(**decisions),
        )

    def _attribute(self, snapshot, observation):
        """Submit the turn's attribution, if it made one, without delaying the observation."""
        verdict = turn_verdict(observation, snapshot_revision=snapshot.revision)
        if verdict is None:
            return
        task = asyncio.get_running_loop().create_task(self._submit(verdict))
        self._submissions.add(task)
        task.add_done_callback(self._submissions.discard)

    async def _submit(self, verdict):
        log = logging.getLogger(__name__)
        try:
            attribution = await self.client.submit_verdict(verdict)
        except (ServiceUpstreamError, ValueError, OSError) as exc:
            # Refused (snapshot changed, already attributed) or unreachable:
            # the turn simply stays unattributed; nothing here depends on it.
            log.info("Voice verdict not recorded turn_id=%s reason=%s", verdict.refs[0].turn_id,
                     getattr(exc, "status_code", None) or type(exc).__name__)
            return
        log.info("Voice attribution turn_id=%s attribution=%s", verdict.refs[0].turn_id,
                 attribution.attribution_id)

    async def _record(self, snapshot, observation, probe, evidence):
        """Candidate evidence and discovery: Data decides; failures change nothing here."""
        from eidolon_sdk.biz.candidate_evidence import CandidateDiscovery, CandidateEvidence
        audio_sha256 = hashlib.sha256(evidence.pcm).hexdigest()
        try:
            if (observation.decisions.candidate_evidence is not None
                    and probe.speech_ms >= CANDIDATE_MIN_SPEECH_MS):
                await self.client.record_candidate_evidence(CandidateEvidence(
                    snapshot_revision=snapshot.revision, audio_sha256=audio_sha256,
                    observation=observation))
            policy = snapshot.discovery_policy
            if (observation.decisions.discovery and policy is not None
                    and policy.identity_policy_version == self.policy.version
                    and len(snapshot.persons) < MAX_RECOGNITION_PERSONS):
                await self.client.discover_candidate(CandidateDiscovery(
                    policy_version=policy.version, snapshot_revision=snapshot.revision,
                    audio_sha256=audio_sha256, observation=observation, sample=probe))
        except (ServiceUpstreamError, ValueError, OSError):
            # A rejected receipt or proposal (missing single-speaker quality,
            # conflict, full pool) never changes the observation already made.
            return
