"""Compose input ownership and an explicit/default Companion using existing authorities.

Internal Channel application boundary, not an authentication or model-routing API.
Callers must obtain the source from authenticated admission, never from user text.
Kernel owns mounted devices; System Data owns Companion identity. This use case
only reads those authorities and does not grant cross-Companion Agent access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity
from eidolon_sdk.device_foundation.v1 import DeviceRef


class InteractionContextError(Exception):
    """The authorities could not establish a matching source and target."""


@dataclass(frozen=True, slots=True)
class InteractionSource:
    """Identity supplied by trusted ingress, independent of the requested target."""

    owner_id: str
    device_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise InteractionContextError("source requires a nonblank owner_id")
        if self.owner_id != self.owner_id.strip():
            raise InteractionContextError("source requires a canonical owner_id")
        if self.device_id is not None and (
            not isinstance(self.device_id, str)
            or not self.device_id.strip()
            or self.device_id != self.device_id.strip()
        ):
            raise InteractionContextError("source requires a nonblank canonical device_id")


@dataclass(frozen=True, slots=True)
class DeviceConnectionContext:
    """Mounted input connection; valid even when no Companion answers here."""

    owner_id: str
    device_id: str
    device_ref: DeviceRef
    mount_revision: int
    answering_companion_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompanionInteractionContext:
    runtime: ResolvedRuntimeIdentity
    mount_revision: int | None = None


class DeviceMountSource(Protocol):
    async def resolve(self, *, owner_id: str, device_id: str) -> DeviceConnectionContext: ...


class RuntimeIdentitySource(Protocol):
    async def resolve_owner(self, owner_id: str) -> ResolvedRuntimeIdentity: ...

    async def resolve_companion(
        self, companion_id: str, *, device_id: str | None
    ) -> ResolvedRuntimeIdentity: ...


async def resolve_interaction_context(
    *,
    source: InteractionSource,
    runtime: RuntimeIdentitySource,
    mounts: DeviceMountSource | None,
    companion_id: str | None = None,
) -> DeviceConnectionContext | CompanionInteractionContext:
    """Explicit target > device assignment > virtual Owner default.

    An unassigned device has no implicit Owner-default fallback. An explicit
    target does not bypass device admission or mutate its persistent assignment.
    A caller must bind the result to a single session before signing its narrow
    Agent token; changing targets requires a new session.
    """
    if companion_id is not None and (
        not isinstance(companion_id, str)
        or not companion_id.strip()
        or companion_id != companion_id.strip()
    ):
        raise InteractionContextError("explicit companion_id must be a nonblank canonical ID")
    connection = None
    if source.device_id is not None:
        if mounts is None:
            raise InteractionContextError(
                "device participant requires the Kernel Device Mount resolver"
            )
        connection = await mounts.resolve(owner_id=source.owner_id, device_id=source.device_id)
        if (
            connection.owner_id != source.owner_id
            or connection.device_id != source.device_id
            or connection.device_ref.device_instance_id != source.device_id
        ):
            raise InteractionContextError("Kernel Device Mount owner/device mismatch")
        companion_id = companion_id or connection.answering_companion_id
        if companion_id is None:
            return connection

    context = (
        await runtime.resolve_companion(companion_id, device_id=source.device_id)
        if companion_id is not None
        else await runtime.resolve_owner(source.owner_id)
    )
    if (
        context.owner_id != source.owner_id
        or context.device_id != source.device_id
        or (companion_id is not None and context.companion_id != companion_id)
    ):
        raise InteractionContextError("Companion runtime owner/target/device mismatch")
    return CompanionInteractionContext(context, connection.mount_revision if connection else None)
