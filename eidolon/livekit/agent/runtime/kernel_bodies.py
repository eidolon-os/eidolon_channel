"""Narrow owner-scoped consumer of Kernel Body Mesh V1.

One read answers both questions a device session starts with: is this device
this Owner's and mounted, and which Companion answers through it. They used to
be one fact on the mount, which is why re-claiming a device silently forgot its
Eidolon; they are now a mount and an assignment, and the Body endpoint is where
they meet.

The document's shape is pinned by importing its definition, not by restating it.
The field set and the name of the endpoint used to be written out here, because
this package deliberately does not depend on the Kernel and the Kernel was where
both lived. They live in the contract package this process already depended on
now, so there is nothing left to mirror — and nothing left to check a mirror
against by reading the producer's source, which is what this file's tests did.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from eidolon_sdk.device_foundation.v1 import (
    DERIVED_ENDPOINT_ID,
    BodyAssignment,
    BodyEndpoint,
)
from pydantic import ValidationError

from .resolver import DeviceConnectionContext, DeviceTokenResolverError


class KernelBodyError(DeviceTokenResolverError):
    """Base error for Kernel Body Mesh lookup."""


class KernelBodyUnavailable(KernelBodyError):
    """Kernel transport or service boundary is unavailable."""


class KernelBodyNotFound(KernelBodyError):
    """No Body exists for this device in the requested Owner namespace."""


class KernelBodyContractError(KernelBodyError):
    """Kernel response drifted from the pinned V1 consumed shape."""


def body_endpoint_id(device_id: str) -> str:
    return f"{device_id}:{DERIVED_ENDPOINT_ID}"


def _answering(assignment: BodyAssignment | None) -> str | None:
    """Who answers here, taking only what this consumer is entitled to act on.

    ``effective_companion_id`` rather than the spec's ``companion_id`` — which
    used to be a paragraph asking the next reader to take it on trust, and the
    trust did not hold: this process once read the spec and started sessions as
    an Eidolon on hardware that was not there. It is a type now. The status is
    the authority's own answer to "is this in force", and it is null for a Body
    whose device is no longer mounted; the assignment outlives the mount on
    purpose, so the spec still names the Companion it will come back to.
    """

    if assignment is None:
        return None
    companion_id = assignment.status.effective_companion_id
    if companion_id is None:
        return None
    if not companion_id.strip():
        # The contract makes this a non-empty string or null; blank-but-present
        # is the one shape it cannot spell. Refused rather than treated as
        # nobody, because a producer emitting it is broken in a way this
        # process should not paper over by starting an anonymous session.
        raise KernelBodyContractError("Kernel Body assignment names no usable Companion")
    return companion_id


def _connection(document: Any, *, owner_id: str, device_id: str) -> DeviceConnectionContext:
    try:
        endpoint = BodyEndpoint.model_validate(document)
    except ValidationError as exc:
        raise KernelBodyContractError("Kernel Body endpoint fields do not match V1") from exc
    # What the shape cannot say: that this is the Body this caller asked about,
    # and that it is answerable at all. ``present`` stays a hard refusal — a
    # device that is no longer mounted is not a session, and the assignment it
    # kept is there so it can come back, not so it can be used while it is gone.
    if (
        endpoint.owner_id != owner_id
        or endpoint.device_id != device_id
        or endpoint.device_ref.device_instance_id != device_id
        or endpoint.body_endpoint_id != body_endpoint_id(device_id)
        or endpoint.present is not True
    ):
        raise KernelBodyContractError("Kernel Body endpoint values do not match V1")
    return DeviceConnectionContext(
        owner_id=owner_id,
        device_id=device_id,
        device_ref=endpoint.device_ref,
        mount_revision=endpoint.mount_revision,
        answering_companion_id=_answering(endpoint.assignment),
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

