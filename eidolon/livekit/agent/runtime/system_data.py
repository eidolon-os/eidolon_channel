"""System Data HTTP adapter for Channel runtime resolution."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from eidolon_sdk.biz.system_data import SystemDataRuntimeClient


class SystemDataRuntimeResolver:
    """Map System Data wire snapshots to Channel runtime domain values."""

    def __init__(
        self,
        client: SystemDataRuntimeClient,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client
        self._http_client = http_client

    async def resolve_owner(self, owner_id: str):
        """Who answers when the entrance named an Owner and no Companion.

        The Owner's default, read from the pointer they set. This used to ask
        for whichever Companion carried ``role='primary'`` — a flag that no
        longer exists, because "which one is the default" became one field on
        the Owner instead of a role competed for across Companion rows.
        """
        snapshot = await self._client.get_owner_default_runtime(owner_id)
        return snapshot.runtime_identity(device_id=None)

    async def resolve_companion(self, companion_id: str, *, device_id: str | None):
        snapshot = await self._client.get_companion_runtime(companion_id)
        return snapshot.runtime_identity(device_id=device_id)

    async def get_companion_face(self, companion_id: str) -> bytes | None:
        return await self._client.get_companion_face(companion_id)

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()


class DeferredSystemDataRuntimeResolver:
    """Lazily open one authenticated System Data client per Channel session.

    Direct-LLM audio does not require runtime identity. Deferring construction
    lets that path start even when optional Voiceprint/Avatar identity is not
    used, while every actual identity/face lookup still goes exclusively
    through the versioned System Data authority.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._resolver: SystemDataRuntimeResolver | None = None
        self._lock = asyncio.Lock()

    async def resolve_owner(self, owner_id: str):
        resolver = await self._get()
        return await resolver.resolve_owner(owner_id)

    async def resolve_companion(self, companion_id: str, *, device_id: str | None):
        resolver = await self._get()
        return await resolver.resolve_companion(companion_id, device_id=device_id)

    async def get_companion_face(self, companion_id: str) -> bytes | None:
        resolver = await self._get()
        return await resolver.get_companion_face(companion_id)

    async def aclose(self) -> None:
        if self._resolver is not None:
            await self._resolver.aclose()

    async def _get(self) -> SystemDataRuntimeResolver:
        if self._resolver is not None:
            return self._resolver
        async with self._lock:
            if self._resolver is None:
                self._resolver = _open_system_data_runtime(self._settings)
            return self._resolver


def build_system_data_runtime(settings: Any) -> DeferredSystemDataRuntimeResolver:
    """Build a session-scoped lazy consumer of System Data Runtime Authority."""
    return DeferredSystemDataRuntimeResolver(settings)


def _open_system_data_runtime(settings: Any) -> SystemDataRuntimeResolver:
    """Open the authenticated HTTP adapter on first authority use."""
    token_env = str(
        getattr(
            settings,
            "data_service_token_env",
            "EIDOLON_DATA_COMPANION_AUTHORITY_TOKEN",
        )
    ).strip()
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"[runtime] {token_env} is required for System Data")
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            float(getattr(settings, "http_timeout_sec", 5.0)),
            connect=float(getattr(settings, "http_connect_timeout_sec", 2.0)),
        ),
        trust_env=False,
    )
    return SystemDataRuntimeResolver(
        SystemDataRuntimeClient(
            http_client,
            str(getattr(settings, "data_api_url", "http://127.0.0.1:8084")),
            service_token=token,
        ),
        http_client=http_client,
    )


__all__ = [
    "DeferredSystemDataRuntimeResolver",
    "SystemDataRuntimeResolver",
    "build_system_data_runtime",
]
