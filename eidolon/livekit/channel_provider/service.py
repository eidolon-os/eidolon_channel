"""Idempotent Channel Provider application service."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import sqlite3
import time
from collections.abc import Callable

from .config import LiveKitConfig
from .contracts import (
    BINDING_FORMAT,
    IdempotencyConflict,
    ProvisionRequest,
    RevokeRequest,
    canonical_json,
)
from .livekit_backend import ChannelBackend
from .store import ChannelProviderStore, StoredProvision


class ChannelProviderService:
    def __init__(
        self,
        *,
        store: ChannelProviderStore,
        backend: ChannelBackend,
        livekit: LiveKitConfig,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._livekit = livekit
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = asyncio.Lock()

    def initialize(self) -> None:
        self._store.initialize()

    async def healthcheck(self) -> None:
        self._store.healthcheck()
        await self._backend.healthcheck()

    async def provision(self, request: ProvisionRequest) -> str:
        async with self._lock:
            stored = self._store.provision(request.operation_id)
            if stored is not None:
                self._require_same_provision(stored, request)
                if stored.status != "active":
                    raise IdempotencyConflict("provision operation was already revoked")
                now = self._now_ms()
                if (
                    stored.response_json
                    and stored.expires_at_ms - now
                    > self._livekit.refresh_before_expiry_seconds * 1000
                ):
                    return stored.response_json
                return await self._refresh(request, stored, now)

            active = self._store.active_device(request.hub_id, request.device.device_id)
            if active is not None:
                raise IdempotencyConflict(
                    "device already has an active Channel Provider operation"
                )

            active_room, control_room, channel_id = self._resource_names(request)
            await self._backend.ensure_rooms(active_room, control_room)
            now = self._now_ms()
            response, expires_at_ms = self._response(
                request=request,
                active_room=active_room,
                control_room=control_room,
                channel_id=channel_id,
                issued_at_ms=now,
            )
            value = StoredProvision(
                operation_id=request.operation_id,
                request_fingerprint=request.fingerprint,
                hub_id=request.hub_id,
                device_id=request.device.device_id,
                owner_id=request.device.owner_id,
                manifest_revision=request.device.manifest_revision,
                active_room=active_room,
                control_room=control_room,
                channel_id=channel_id,
                response_json=response,
                expires_at_ms=expires_at_ms,
                status="active",
            )
            try:
                self._store.create_provision(value)
            except sqlite3.IntegrityError as exc:
                raise IdempotencyConflict("provision operation raced with another authority") from exc
            return response

    async def _refresh(
        self,
        request: ProvisionRequest,
        stored: StoredProvision,
        issued_at_ms: int,
    ) -> str:
        await self._backend.ensure_rooms(stored.active_room, stored.control_room)
        response, expires_at_ms = self._response(
            request=request,
            active_room=stored.active_room,
            control_room=stored.control_room,
            channel_id=stored.channel_id,
            issued_at_ms=issued_at_ms,
        )
        self._store.refresh_provision(
            operation_id=request.operation_id,
            request_fingerprint=request.fingerprint,
            response_json=response,
            expires_at_ms=expires_at_ms,
        )
        return response

    async def revoke(self, request: RevokeRequest) -> str:
        async with self._lock:
            prior = self._store.revocation(request.operation_id)
            if prior is not None:
                if (
                    prior.request_fingerprint != request.fingerprint
                    or prior.device_id != request.device_id
                ):
                    raise IdempotencyConflict(
                        "revocation operation_id was reused with different content"
                    )
                return prior.response_json

            active = self._store.active_device(request.hub_id, request.device_id)
            if active is not None:
                await self._backend.revoke_rooms(active.active_room, active.control_room)
            response = canonical_json(
                {
                    "operation": "channel.revoked-device",
                    "operation_id": request.operation_id,
                    "device_id": request.device_id,
                }
            )
            try:
                self._store.complete_revocation(
                    operation_id=request.operation_id,
                    request_fingerprint=request.fingerprint,
                    hub_id=request.hub_id,
                    device_id=request.device_id,
                    response_json=response,
                )
            except sqlite3.IntegrityError as exc:
                raise IdempotencyConflict("revocation operation raced with another request") from exc
            return response

    def _response(
        self,
        *,
        request: ProvisionRequest,
        active_room: str,
        control_room: str,
        channel_id: str,
        issued_at_ms: int,
    ) -> tuple[str, int]:
        binding = self._backend.build_binding(
            active_room=active_room,
            control_room=control_room,
            device_id=request.device.device_id,
            owner_id=request.device.owner_id,
            issued_at_ms=issued_at_ms,
        )
        response = canonical_json(
            {
                "operation": "channel.provisioned-device",
                "operation_id": request.operation_id,
                "device_id": request.device.device_id,
                "manifest_revision": request.device.manifest_revision,
                "channels": [
                    {
                        "channel_id": channel_id,
                        "purpose": "livekit-device-session",
                        "kinds": ["reliable-data", "realtime-data", "audio"],
                        "binding_format": BINDING_FORMAT,
                        "issued_at_ms": issued_at_ms,
                        "expires_at_ms": binding.expires_at_ms,
                        "opaque_binding": base64.b64encode(binding.payload).decode("ascii"),
                    }
                ],
            }
        )
        return response, binding.expires_at_ms

    def _resource_names(self, request: ProvisionRequest) -> tuple[str, str, str]:
        digest = hashlib.sha256(
            f"{request.hub_id}\0{request.device.device_id}".encode()
        ).hexdigest()[:24]
        stem = f"{self._livekit.room_prefix}-{digest}"
        return f"{stem}-voice", f"{stem}-control", f"livekit-{digest}"

    @staticmethod
    def _require_same_provision(
        stored: StoredProvision,
        request: ProvisionRequest,
    ) -> None:
        if stored.request_fingerprint != request.fingerprint:
            raise IdempotencyConflict(
                "provision operation_id was reused with different content"
            )
        if (
            stored.hub_id != request.hub_id
            or stored.device_id != request.device.device_id
            or stored.owner_id != request.device.owner_id
            or stored.manifest_revision != request.device.manifest_revision
        ):
            raise IdempotencyConflict("stored provision authority does not match request")

    async def close(self) -> None:
        await self._backend.close()
