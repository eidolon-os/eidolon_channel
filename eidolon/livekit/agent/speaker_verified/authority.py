"""Companion-scoped adapter to the existing System Data authorization authority."""

from eidolon_sdk.biz.system_data import SystemDataRuntimeClient

from .orchestration import SpeakerDecision


class SpeakerPolicyAuthority:
    def __init__(self, client: SystemDataRuntimeClient, *, owner_id: str, companion_id: str):
        if not owner_id or not companion_id:
            raise ValueError("resolved Owner and Companion required")
        self._client, self._owner, self._companion = client, owner_id, companion_id

    async def authorize(self, decision: SpeakerDecision) -> bool:
        if (
            decision.status != "match"
            or not decision.profile_id
            or decision.profile_revision < 1
            or not decision.calibration_version
            or decision.policy_revision < 0
        ):
            return False
        snapshot = await self._client.get_voiceprint_snapshot(self._companion)
        if (
            snapshot.owner_id != self._owner
            or snapshot.companion_id != self._companion
            or snapshot.mode != "allowlist"
            or snapshot.revision != decision.policy_revision
            or snapshot.calibration_digest != decision.calibration_version
        ):
            # Data permits ordinary conversations when off. That is not an
            # identity grant for a session which selected speaker verification.
            return False
        if not any(
            template.profile_id == decision.profile_id
            and template.revision == decision.profile_revision
            for template in snapshot.templates
        ):
            return False
        # Data rechecks revision transactionally. Revocation between snapshot
        # and this request must fail; no read-then-local-allow shortcut.
        return await self._client.admit_voiceprint(
            self._companion,
            snapshot.revision,
            decision.profile_id,
            profile_revision=decision.profile_revision,
            calibration_digest=decision.calibration_version,
        )
