"""Idempotent Channel Provider application service.

This layer owns idempotency, persistence and the Hub-facing response shape. It
does not know how a channel is realised: it derives what the device declared it
needs, asks the registry which adapter can carry that, and stores whatever
handle the adapter hands back without ever looking inside it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable

from .contracts import IdempotencyConflict, ProvisionRequest, RevokeRequest, canonical_json
from .ports import ChannelGrant
from .selection import AdapterRegistry
from .spec import ChannelSpec, MediaFlow, derive_spec
from .store import ChannelProviderStore, StoredProvision


class ChannelProviderService:
    def __init__(
        self,
        *,
        store: ChannelProviderStore,
        registry: AdapterRegistry,
        agent_name: str,
        refresh_before_expiry_seconds: int,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._registry = registry
        self._agent_name = agent_name
        self._refresh_before_expiry_seconds = refresh_before_expiry_seconds
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = asyncio.Lock()

    def initialize(self) -> None:
        self._store.initialize()

    async def healthcheck(self) -> None:
        self._store.healthcheck()
        await self._registry.healthcheck()

    async def shutdown(self) -> None:
        await self._registry.shutdown()

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
                    > self._refresh_before_expiry_seconds * 1000
                ):
                    return stored.response_json
                return await self._refresh(request, stored, now)

            active = self._store.active_device(request.hub_id, request.device.device_id)
            if active is not None:
                raise IdempotencyConflict(
                    "device already has an active Channel Provider operation"
                )

            spec = derive_spec(request.device, agent_name=self._agent_name)
            adapter = self._registry.select(spec)
            now = self._now_ms()
            grant = await adapter.open(spec, issued_at_ms=now)
            channel_id = self._channel_id(request)
            response = self._response(
                request=request,
                spec=spec,
                grant=grant,
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
                adapter_name=adapter.name,
                handle_json=json.dumps(grant.handle, sort_keys=True, separators=(",", ":")),
                channel_id=channel_id,
                response_json=response,
                expires_at_ms=grant.expires_at_ms,
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
        """Re-open the same channel on the adapter that originally opened it.

        A refresh must not migrate a device between transports mid-life: the
        device is holding a binding of one format and has no way to be told the
        ground moved. Selection therefore runs only on a first provision.
        """
        spec = derive_spec(request.device, agent_name=self._agent_name)
        adapter = self._registry.get(stored.adapter_name)
        grant = await adapter.open(spec, issued_at_ms=issued_at_ms)
        response = self._response(
            request=request,
            spec=spec,
            grant=grant,
            channel_id=stored.channel_id,
            issued_at_ms=issued_at_ms,
        )
        self._store.refresh_provision(
            operation_id=request.operation_id,
            request_fingerprint=request.fingerprint,
            handle_json=json.dumps(grant.handle, sort_keys=True, separators=(",", ":")),
            response_json=response,
            expires_at_ms=grant.expires_at_ms,
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
                adapter = self._registry.get(active.adapter_name)
                await adapter.close(json.loads(active.handle_json))
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
        spec: ChannelSpec,
        grant: ChannelGrant,
        channel_id: str,
        issued_at_ms: int,
    ) -> str:
        return canonical_json(
            {
                "operation": "channel.provisioned-device",
                "operation_id": request.operation_id,
                "device_id": request.device.device_id,
                "manifest_revision": request.device.manifest_revision,
                "channels": [
                    {
                        "channel_id": channel_id,
                        "purpose": "device-session",
                        "kinds": _channel_kinds(spec),
                        "binding_format": grant.binding_format,
                        "issued_at_ms": issued_at_ms,
                        "expires_at_ms": grant.expires_at_ms,
                        "opaque_binding": base64.b64encode(grant.payload).decode("ascii"),
                    }
                ],
            }
        )

    @staticmethod
    def _channel_id(request: ProvisionRequest) -> str:
        digest = hashlib.sha256(
            f"{request.hub_id}\0{request.device.device_id}".encode()
        ).hexdigest()[:24]
        return f"chan-{digest}"

    @staticmethod
    def _require_same_provision(stored: StoredProvision, request: ProvisionRequest) -> None:
        if (
            stored.request_fingerprint != request.fingerprint
            or stored.hub_id != request.hub_id
            or stored.device_id != request.device.device_id
        ):
            raise IdempotencyConflict("provision operation_id was reused with different content")


def _channel_kinds(spec: ChannelSpec) -> list[str]:
    """Advertise what this channel actually carries for this device.

    A camera that never speaks should not be told its channel carries audio.
    """
    kinds = ["reliable-data", "realtime-data"]
    if spec.audio is not MediaFlow.NONE:
        kinds.append("audio")
    if spec.video is not MediaFlow.NONE:
        kinds.append("video")
    return kinds
