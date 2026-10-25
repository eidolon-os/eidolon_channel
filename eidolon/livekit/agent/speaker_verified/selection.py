"""Resolve the session's voice policy from its actual Companion, never its device."""

from eidolon_sdk.biz.voiceprint import VoicePolicySnapshot


async def resolve_voice_policy(services, room) -> VoicePolicySnapshot | None:
    # Standalone direct-LLM sessions have no Companion. Product sessions always
    # supply these services; failures must not masquerade as an off policy.
    if services is None:
        return None
    identity = await services.resolve_room(room)
    snapshot = await services.runtime.get_voiceprint_snapshot(identity.companion_id)
    if snapshot.owner_id != identity.owner_id or snapshot.companion_id != identity.companion_id:
        raise ValueError("VOICE_POLICY_SCOPE_MISMATCH")
    return snapshot
