"""Encode a temporary grant using the existing device control envelope.

No transport or authorization lives here. The caller owns the room and must
retain the exact command for retries, then observe admission independently.
"""

import base64

from eidolon_sdk.biz.control.channel_binding import ChannelBinding
from eidolon_sdk.biz.control.shared_session import SharedSessionInvitation
from eidolon_sdk.device_foundation.v1 import DeviceRef

from .ports import ChannelGrant


SHARED_VISIT_MAX_SECONDS = 120


def invitation_command(
    grant: ChannelGrant, *, device_ref: DeviceRef, session_id: str,
    command_id: str, channel_id: str, kinds: tuple[str, ...],
    issued_at_ms: int, deadline_ms: int,
) -> dict:
    return SharedSessionInvitation(
        session_id=session_id,
        device_ref=device_ref,
        deadline_ms=deadline_ms,
        channel=ChannelBinding(
            channel_id=channel_id, purpose="shared-session",
            kinds=kinds,
            binding_format=grant.binding_format,
            issued_at_ms=issued_at_ms, expires_at_ms=grant.expires_at_ms,
            opaque_binding=base64.b64encode(grant.payload).decode("ascii"),
        ),
    ).command(command_id=command_id)
