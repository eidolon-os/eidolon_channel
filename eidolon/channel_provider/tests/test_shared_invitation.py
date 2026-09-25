import base64
import json
from dataclasses import replace

from eidolon_sdk.biz.control.shared_session import SharedSessionInvitation
from eidolon_sdk.device_foundation.v1 import DeviceRef
from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

from eidolon.channel_provider.shared_invitation import invitation_command
from .test_livekit_adapter import _adapter, _spec


async def test_real_adapter_grants_round_trip_through_shared_invitation_contract():
    adapter, client = _adapter()
    first = _spec()
    second = replace(first, device_id=named_device_instance_id("second"))
    grants = await adapter.open_shared(
        (first, second), input_device_id=first.device_id, issued_at_ms=1000
    )
    for device_id, grant in grants.items():
        ref = DeviceRef(device_instance_id=device_id, owner_domain_id="owner-domain-1",
                        owner_domain_generation=1, claim_generation=1, trust_epoch=1)
        command = invitation_command(
            grant, device_ref=ref, session_id="team-1", command_id=f"invite-{device_id}",
            channel_id=adapter.resource_identity(grant.handle),
            kinds=("reliable-data", "realtime-data", "audio"),
            issued_at_ms=1000, deadline_ms=2000,
        )
        parsed = SharedSessionInvitation.model_validate_json(json.dumps(command["payload"]))
        assert base64.b64decode(parsed.channel.opaque_binding) == grant.payload
        assert command["dst"]["id"] == ref.device_instance_id
        assert parsed.device_ref == ref
    assert client.agent_dispatch.created == []
