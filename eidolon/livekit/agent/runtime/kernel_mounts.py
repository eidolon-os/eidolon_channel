"""Narrow owner-scoped consumer of Kernel Device Mount V1."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import httpx

from .resolver import DeviceConnectionContext, DeviceTokenResolverError

_FIELDS = {
    "operation",
    "device_id",
    "owner_id",
    "attached_companion_id",
    "revision",
    "created_at",
    "updated_at",
    "request_id",
    "fingerprint",
    "active",
}


class KernelMountError(DeviceTokenResolverError):
    """Base error for Kernel Mount lookup."""


class KernelMountUnavailable(KernelMountError):
    """Kernel transport or service boundary is unavailable."""


class KernelMountNotFound(KernelMountError):
    """No active Device Mount exists in the requested Owner namespace."""


class KernelMountContractError(KernelMountError):
    """Kernel response drifted from the pinned V1 consumed shape."""


def _connection(document: Any, *, owner_id: str, device_id: str) -> DeviceConnectionContext:
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise KernelMountContractError("Kernel Mount response fields do not match V1")
    attached = document["attached_companion_id"]
    revision = document["revision"]
    if (
        document["operation"] != "kernel.device-mount"
        or document["owner_id"] != owner_id
        or document["device_id"] != device_id
        or document["active"] is not True
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
        or (attached is not None and (not isinstance(attached, str) or not attached.strip()))
        or not isinstance(document["request_id"], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(document["fingerprint"])) is None
    ):
        raise KernelMountContractError("Kernel Mount response values do not match V1")
    return DeviceConnectionContext(
        owner_id=owner_id,
        device_id=device_id,
        mount_revision=revision,
        attached_companion_id=attached,
    )


class KernelMountHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._timeout = timeout_sec

    async def resolve(self, *, owner_id: str, device_id: str) -> DeviceConnectionContext:
        if self._http is None:
            async with httpx.AsyncClient(trust_env=False) as http:
                return await self._resolve_with(
                    http,
                    owner_id=owner_id,
                    device_id=device_id,
                )
        return await self._resolve_with(
            self._http,
            owner_id=owner_id,
            device_id=device_id,
        )

    async def _resolve_with(
        self,
        http: httpx.AsyncClient,
        *,
        owner_id: str,
        device_id: str,
    ) -> DeviceConnectionContext:
        path = (
            f"{self._base_url}/device-mounts/resolve/"
            f"{quote(device_id, safe='')}"
        )
        try:
            response = await http.get(
                path,
                headers={"X-Eidolon-Owner": owner_id},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise KernelMountUnavailable(f"Kernel Mount GET failed: {exc}") from exc
        if response.status_code == 404:
            raise KernelMountNotFound(
                f"device {device_id!r} is not mounted for owner {owner_id!r}"
            )
        if response.status_code != 200:
            raise KernelMountUnavailable(
                f"Kernel Mount GET returned HTTP {response.status_code}"
            )
        try:
            document = response.json()
        except ValueError as exc:
            raise KernelMountContractError("Kernel Mount response is not JSON") from exc
        return _connection(document, owner_id=owner_id, device_id=device_id)

    async def close(self) -> None:
        """Injected HTTP clients remain owned by the composition caller."""
