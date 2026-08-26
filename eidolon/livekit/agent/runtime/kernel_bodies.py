"""Narrow owner-scoped consumer of Kernel Body Mesh V1.

One read answers both questions a device session starts with: is this device
this Owner's and mounted, and which Companion answers through it. They used to
be one fact on the mount, which is why re-claiming a device silently forgot its
Eidolon; they are now a mount and an assignment, and the Body endpoint is where
they meet.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from eidolon_sdk.device_foundation.v1 import DeviceRef
from pydantic import ValidationError

from .resolver import DeviceConnectionContext, DeviceTokenResolverError

#: The endpoint document this consumer reads, pinned exactly. A field arriving
#: here that nobody admitted is the producer handing this process something no
#: one decided it should see; a field missing is a producer that moved on.
_FIELDS = {
    "operation",
    "body_endpoint_id",
    "device_id",
    "owner_id",
    "endpoint_id",
    "device_ref",
    "mount_revision",
    "roles",
    "assignment_policy",
    "risk_class",
    "concurrency",
    "source",
    "present",
    "assignment",
}

#: The one endpoint every mounted device has while no Manifest declares any.
#: Mirrored rather than imported: this package deliberately does not depend on
#: the Kernel, and the test beside it compares this consumer to the producer's
#: own schema.
_DERIVED_ENDPOINT_ID = "body"


class KernelBodyError(DeviceTokenResolverError):
    """Base error for Kernel Body Mesh lookup."""


class KernelBodyUnavailable(KernelBodyError):
    """Kernel transport or service boundary is unavailable."""


class KernelBodyNotFound(KernelBodyError):
    """No Body exists for this device in the requested Owner namespace."""


class KernelBodyContractError(KernelBodyError):
    """Kernel response drifted from the pinned V1 consumed shape."""


def body_endpoint_id(device_id: str) -> str:
    return f"{device_id}:{_DERIVED_ENDPOINT_ID}"


def _answering(assignment: Any) -> str | None:
    """Who answers here, taking only what this consumer is entitled to act on.

    ``effective_companion_id`` rather than the spec's ``companion_id``: the
    status is the authority's own answer to "is this actually in force", and it
    is null for a Body whose device is no longer mounted. Reading the spec
    instead would start a session as an Eidolon on hardware that is not there.
    """

    if assignment is None:
        return None
    if not isinstance(assignment, dict):
        raise KernelBodyContractError("Kernel Body assignment is not an object")
    status = assignment.get("status")
    if not isinstance(status, dict):
        raise KernelBodyContractError("Kernel Body assignment carries no status")
    companion_id = status.get("effective_companion_id")
    if companion_id is None:
        return None
    if not isinstance(companion_id, str) or not companion_id.strip():
        raise KernelBodyContractError("Kernel Body assignment names no usable Companion")
    return companion_id


def _connection(document: Any, *, owner_id: str, device_id: str) -> DeviceConnectionContext:
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise KernelBodyContractError("Kernel Body endpoint fields do not match V1")
    try:
        device_ref = DeviceRef.model_validate(document["device_ref"])
    except ValidationError as exc:
        raise KernelBodyContractError("Kernel Body DeviceRef is invalid") from exc
    mount_revision = document["mount_revision"]
    if (
        document["operation"] != "kernel.body-endpoint"
        or document["owner_id"] != owner_id
        or document["device_id"] != device_id
        or device_ref.device_instance_id != device_id
        or document["body_endpoint_id"] != body_endpoint_id(device_id)
        or document["present"] is not True
        or not isinstance(mount_revision, int)
        or isinstance(mount_revision, bool)
        or mount_revision < 1
    ):
        raise KernelBodyContractError("Kernel Body endpoint values do not match V1")
    return DeviceConnectionContext(
        owner_id=owner_id,
        device_id=device_id,
        device_ref=device_ref,
        mount_revision=mount_revision,
        answering_companion_id=_answering(document["assignment"]),
    )


class KernelBodyHttpClient:
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
        endpoint = quote(body_endpoint_id(device_id), safe="")
        path = f"{self._base_url}/body-endpoints/{endpoint}"
        try:
            response = await http.get(
                path,
                headers={"X-Eidolon-Owner": owner_id},
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise KernelBodyUnavailable(f"Kernel Body GET failed: {exc}") from exc
        if response.status_code == 404:
            raise KernelBodyNotFound(
                f"device {device_id!r} is not mounted for owner {owner_id!r}"
            )
        if response.status_code != 200:
            raise KernelBodyUnavailable(f"Kernel Body GET returned HTTP {response.status_code}")
        try:
            document = response.json()
        except ValueError as exc:
            raise KernelBodyContractError("Kernel Body response is not JSON") from exc
        return _connection(document, owner_id=owner_id, device_id=device_id)

    async def close(self) -> None:
        """Injected HTTP clients remain owned by the composition caller."""

