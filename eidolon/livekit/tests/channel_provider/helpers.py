from __future__ import annotations

import json
from typing import Any

from eidolon.livekit.channel_provider.config import LiveKitConfig


def livekit_config(**overrides: Any) -> LiveKitConfig:
    values: dict[str, Any] = {
        "api_url": "http://127.0.0.1:7880",
        "client_url": "wss://livekit.example.test",
        "api_key": "test-api-key",
        "api_secret": "test-api-secret-with-at-least-32-bytes",
        "grant_ttl_seconds": 1800,
        "refresh_before_expiry_seconds": 120,
    }
    values.update(overrides)
    return LiveKitConfig(**values)


def provision_payload(**device_overrides: Any) -> dict[str, Any]:
    device: dict[str, Any] = {
        "device_id": "device-1",
        "owner_id": "owner-1",
        "display_name": "Kitchen Box",
        "device_kind": "waveshare-box3",
        "manifest": {
            "schema_version": 1,
            "title": "Eidolon Box",
            "properties": [],
            "actions": [],
            "events": [],
            "media": [
                {
                    "kind": "audio",
                    "direction": "bidirectional",
                    "codecs": ["audio/opus"],
                }
            ],
        },
        "manifest_revision": "sha256:manifest-1",
    }
    device.update(device_overrides)
    return {
        "operation": "channel.provision-device",
        "operation_id": "enrollment-1",
        "hub_id": "hub-1",
        "device": device,
    }


def revoke_payload(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "operation": "channel.revoke-device",
        "operation_id": "revoke-1",
        "hub_id": "hub-1",
        "device_id": "device-1",
        "reason": "owner-request",
    }
    value.update(overrides)
    return value


def encoded(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
