from __future__ import annotations

import json
from typing import Any

from eidolon_sdk.device_foundation.v1.testing import named_device_instance_id

from eidolon.channel_provider.adapters.livekit import LiveKitConfig
from eidolon.channel_provider.ports import ChannelGrant, ServingRequest, ServingRequestSink
from eidolon.channel_provider.spec import ChannelSpec


def livekit_config(**overrides: Any) -> LiveKitConfig:
    values: dict[str, Any] = {
        "api_url": "http://127.0.0.1:7880",
        "client_url": "wss://livekit.example.test",
        "api_key": "test-api-key",
        "api_secret": "test-api-secret-with-at-least-32-bytes",
        "grant_ttl_seconds": 1800,
    }
    values.update(overrides)
    return LiveKitConfig(**values)


def device_ref(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "device_instance_id": named_device_instance_id("device-1"),
        "owner_domain_id": "owner-domain-1",
        "owner_domain_generation": 1,
        "claim_generation": 1,
        "trust_epoch": 1,
    }
    value.update(overrides)
    return value


def provision_payload(**device_overrides: Any) -> dict[str, Any]:
    ref = device_ref(**device_overrides.pop("device_ref", {}))
    if "device_id" in device_overrides:
        ref["device_instance_id"] = device_overrides.pop("device_id")
    device: dict[str, Any] = {
        "owner_id": "owner_1",
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
        "device_ref": ref,
        "device": device,
    }


def current_payload(**overrides: Any) -> dict[str, Any]:
    ref = device_ref(**overrides.pop("device_ref", {}))
    if "device_id" in overrides:
        ref["device_instance_id"] = overrides.pop("device_id")
    value: dict[str, Any] = {
        "operation": "channel.current-device",
        "device_ref": ref,
    }
    value.update(overrides)
    return value


def revoke_payload(**overrides: Any) -> dict[str, Any]:
    ref = device_ref(**overrides.pop("device_ref", {}))
    if "device_id" in overrides:
        ref["device_instance_id"] = overrides.pop("device_id")
    value: dict[str, Any] = {
        "operation": "channel.revoke-device",
        "operation_id": "revoke-1",
        "device_ref": ref,
        "reason": "owner-request",
    }
    value.update(overrides)
    return value


def session_payload(*, operation: str, **overrides: Any) -> dict[str, Any]:
    ref = device_ref(**overrides.pop("device_ref", {}))
    if "device_id" in overrides:
        ref["device_instance_id"] = overrides.pop("device_id")
    value: dict[str, Any] = {
        "operation": operation,
        "device_ref": ref,
        "conversation_id": "conversation-1",
    }
    value.update(overrides)
    return value


def encoded(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


class FakeAdapter:
    """A transport that records what it was asked to carry."""

    def __init__(
        self,
        *,
        name: str = "fake",
        carries_media: bool = True,
        serves_dataonly: bool = False,
        ttl_seconds: int = 1800,
        binding_format: str | None = None,
        payload: bytes | None = None,
    ):
        self._name = name
        self._carries_media = carries_media
        self._serves_dataonly = serves_dataonly
        self.ttl_seconds = ttl_seconds
        # Overridable so a test can hand the service the exact bytes a golden
        # vector pins and read back what the wire made of them. The service
        # owns the encoding, not the adapter, so that is the only place it can
        # be observed.
        self._binding_format = binding_format
        self._payload = payload
        self.opened: list[ChannelSpec] = []
        self.closed: list[dict[str, Any]] = []
        self.sessions_opened: list[tuple[dict[str, Any], str]] = []
        self.sessions_closed: list[tuple[dict[str, Any], str]] = []
        # Channels currently listened to, keyed the way a real adapter would
        # have to key them, so re-stating one cannot leave two behind.
        self.watched: dict[str, ServingRequestSink] = {}
        self.stopped: list[dict[str, Any]] = []
        self.health_calls = 0
        self.shutdown_calls = 0
        # What this transport will say about the device being on its channel.
        # `None` is the default because "cannot see" is what a transport with no
        # notion of presence must answer, and the tests need that case as much
        # as the other two.
        self.on_channel: bool | None = None
        self.presence_reads: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def carries_media(self) -> bool:
        return self._carries_media

    @property
    def serves_dataonly(self) -> bool:
        return self._serves_dataonly

    async def healthcheck(self) -> None:
        self.health_calls += 1

    async def open(self, spec: ChannelSpec, *, issued_at_ms: int) -> ChannelGrant:
        self.opened.append(spec)
        return ChannelGrant(
            binding_format=self._binding_format
            or f"application/vnd.eidolon.{self._name}-session+json;v=2",
            payload=self._payload
            or json.dumps(
                {"device": spec.device_id, "resource": f"{self._name}:{spec.device_id}"}
            ).encode(),
            expires_at_ms=issued_at_ms + self.ttl_seconds * 1000,
            handle={"resource": f"{self._name}:{spec.device_id}"},
        )

    def resource_identity(self, handle: dict[str, Any]) -> str:
        resource = str(handle.get("resource") or "")
        return f"{self._name}:{resource}" if resource else ""

    async def close(self, handle: dict[str, Any]) -> None:
        self.closed.append(handle)

    async def device_is_on_channel(self, handle: dict[str, Any]) -> bool | None:
        self.presence_reads.append(handle)
        return self.on_channel

    async def open_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        self.sessions_opened.append((handle, conversation_id))

    async def close_session(self, handle: dict[str, Any], conversation_id: str) -> None:
        self.sessions_closed.append((handle, conversation_id))

    async def accept_requests(self, handle: dict[str, Any], *, sink) -> None:
        self.watched[handle["resource"]] = sink

    async def stop_accepting(self, handle: dict[str, Any]) -> None:
        if self.watched.pop(handle.get("resource", ""), None) is not None:
            self.stopped.append(handle)

    async def device_asks(self, device_id: str, request: ServingRequest) -> None:
        """Play the device speaking over its own channel."""
        await self.watched[f"{self._name}:{device_id}"](request)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def audio_manifest(
    *,
    direction: str = "bidirectional",
    interaction_mode: str | None = None,
    video: str | None = None,
) -> dict[str, Any]:
    """A manifest that declares what a device can actually do."""
    media: list[dict[str, Any]] = []
    if direction:
        media.append({"kind": "audio", "direction": direction, "codecs": ["audio/opus"]})
    if video:
        media.append({"kind": "video", "direction": video, "codecs": ["video/h264"]})
    properties: list[dict[str, Any]] = []
    if interaction_mode:
        properties.append(
            {
                "name": "interaction_mode",
                "schema": {"type": "string", "const": interaction_mode},
                "observable": False,
                "writable": False,
            }
        )
    return {
        "schema_version": 1,
        "title": "Eidolon Device",
        "properties": properties,
        "actions": [],
        "events": [],
        "media": media,
    }
