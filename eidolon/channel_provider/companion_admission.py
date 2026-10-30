"""Resolve ordinary and device-initiated targets through the existing authorities."""
from .contracts import Forbidden, InvalidTransition


class CompanionAdmission:
    def __init__(self, bodies, runtime):
        self.bodies, self.runtime = bodies, runtime

    async def resolve(self, *, owner_id, device_ref, target_companion_id, require_attachment=False):
        body = await self.bodies.resolve(owner_id=owner_id, device_id=device_ref.device_instance_id)
        if body.owner_id != owner_id or body.device_ref != device_ref:
            raise Forbidden("Companion admission requires the current owned DeviceRef")
        target = target_companion_id or body.answering_companion_id
        if target is None:
            raise InvalidTransition("device has no answering Companion")
        if require_attachment and body.answering_companion_id != target:
            raise InvalidTransition("Team Companion binding changed")
        identity = await self.runtime.resolve_companion(target, device_id=device_ref.device_instance_id)
        if (identity.owner_id != owner_id or identity.companion_id != target
                or identity.device_id != device_ref.device_instance_id):
            raise Forbidden("Companion admission Owner or target mismatch")
        return target
