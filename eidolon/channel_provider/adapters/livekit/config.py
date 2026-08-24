"""Settings the LiveKit adapter needs, and nothing the service needs.

`interaction_mode` deliberately does not appear here. Turn taking is a property
of the device that is speaking, not of the deployment that hosts it, so a
process-wide setting could only ever be wrong for every device that disagreed
with it. It is read from the device manifest instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiveKitConfig:
    api_url: str
    client_url: str
    api_key: str
    api_secret: str
    room_prefix: str = "eidolon-device"
    agent_name: str = "eidolon"
    grant_ttl_seconds: int = 1800
    sample_rate: int = 16000
    channels: int = 1
