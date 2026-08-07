"""Compose OS runtime resolution + narrow Agent session authentication.

Phase 32.B: when ``EidolonAgentGrpcLlm`` opens its session (first
``chat()`` call), it invokes the device_token callable to get a fresh
bearer. The resolver here is what that callable does:

  1. Inspect the LiveKit room for a remote participant.
  2. Dispatch by ``participant.metadata.kind``.
  3. Resolve the entrance through Kernel Mount + System Data into the explicit
     owner/companion/runtime identity envelope.
  4. Sign a narrow runtime JWT with Owner, Companion, and optional Device.
  5. Cache the result for the lifetime of this resolver instance —
     subsequent invocations within the session return the same token.

If metadata is missing or an authority denies the context, the resolver raises a clear error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from eidolon_sdk.biz.persona import ResolvedRuntimeIdentity as ResolvedContext
from eidolon_sdk.biz.runtime import sign_runtime_token
from eidolon_sdk.biz.system_data import SystemDataError

_log = logging.getLogger(__name__)

DeviceTokenResolver = Callable[[], Awaitable[str]]
RuntimeTokenResolver = DeviceTokenResolver
RUNTIME_PARTICIPANT_KINDS = frozenset({"device", "companion", "owner", "user"})


class DeviceTokenResolverError(Exception):
    """The authoritative session context could not produce an Agent token."""


class RoomNotConnectedError(DeviceTokenResolverError):
    """Runtime participant resolution requires an active room connection."""


@dataclass(frozen=True, slots=True)
class DeviceConnectionContext:
    """Owner-scoped mounted Device; Companion attachment is optional.

    This context is sufficient for a Channel Provider data connection, but not
    for constructing an Agent/audio runtime token.
    """

    owner_id: str
    device_id: str
    mount_revision: int
    attached_companion_id: str | None = None


@dataclass(frozen=True, slots=True)
class CompanionInteractionContext:
    """Complete Companion runtime selected before entering the audio pipeline."""

    runtime: ResolvedContext
    mount_revision: int | None = None


def _participant_identity_and_metadata(
    room: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Select the runtime participant rather than the first room participant.

    Non-publishing infrastructure participants (for example the Hub control
    bridge) are not voice-session entrants and must never own RoomIO, identity,
    memory, or runtime-token resolution. Explicit participant metadata wins even if
    a system participant joined the room first.  A publishing participant with
    invalid metadata remains visible so the strict configuration error is not
    silently hidden.
    """
    if room is None:
        return None
    try:
        participants = list(getattr(room, "remote_participants", {}).values())
    except Exception:  # noqa: BLE001 — defensive
        return None
    invalid_participant: tuple[str, dict[str, Any]] | None = None
    for participant in participants:
        identity = (getattr(participant, "identity", "") or "").strip()
        if not identity:
            continue
        metadata = _participant_metadata(participant)
        kind = str(metadata.get("kind") or "").strip().lower()
        if kind in RUNTIME_PARTICIPANT_KINDS:
            return identity, metadata
        permissions = getattr(participant, "permissions", None)
        if getattr(permissions, "can_publish", None) is False:
            continue
        if invalid_participant is None:
            invalid_participant = (identity, metadata)
    return invalid_participant


def _participant_metadata(participant: Any) -> dict[str, Any]:
    raw_meta = getattr(participant, "metadata", "") or ""
    if not raw_meta:
        return {}
    try:
        parsed = json.loads(raw_meta)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def wait_for_runtime_participant_identity(
    room: Any,
    *,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.05,
) -> str:
    """Wait until an explicitly typed voice-session participant is in the room."""
    identity, _ = await wait_for_runtime_participant_metadata(
        room,
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )
    return identity


async def wait_for_runtime_participant_metadata(
    room: Any,
    *,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.05,
) -> tuple[str, dict[str, Any]]:
    """Wait for and return the explicitly typed voice-session participant.

    Infrastructure participants may join first.  Returning identity and
    metadata from the same selection pass prevents session construction from
    accidentally reading metadata from the Hub control bridge.
    """

    if room is None or not room.isconnected():
        raise RoomNotConnectedError(
            "wait_for_runtime_participant_metadata() called on an unconnected "
            "room; connect the job (JobContext.connect()) before resolving the "
            "runtime participant"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_sec)
    while True:
        participant = _participant_identity_and_metadata(room)
        if participant is not None:
            identity, metadata = participant
            kind = str(metadata.get("kind") or "").strip().lower()
            if kind in RUNTIME_PARTICIPANT_KINDS:
                return identity, metadata
        if loop.time() >= deadline:
            raise DeviceTokenResolverError(
                "no remote participant with runtime participant metadata "
                "(kind=device/companion/owner/user) became available"
            )
        await asyncio.sleep(max(0.0, poll_interval_sec))


async def _resolve_context(
    *,
    runtime: Any,
    mounts: Any | None,
    identity: str,
    metadata: dict[str, Any],
) -> ResolvedContext:
    """Resolve a participant to its complete Companion runtime context."""
    resolved = await resolve_channel_context(
        runtime=runtime,
        mounts=mounts,
        identity=identity,
        metadata=metadata,
    )
    if isinstance(resolved, DeviceConnectionContext):
        raise DeviceTokenResolverError(f"device {resolved.device_id!r} has no companion target")
    return resolved.runtime


async def resolve_channel_context(
    *,
    runtime: Any,
    mounts: Any | None,
    identity: str,
    metadata: dict[str, Any],
) -> DeviceConnectionContext | CompanionInteractionContext:
    """Resolve ingress before choosing a Channel processing path.

    An unattached Device remains a valid Device connection. A complete
    Companion runtime is requested only when a Kernel attachment or explicit
    same-Owner target exists. Physical Device ingress always requires Kernel.
    """
    kind = str(metadata.get("kind") or "").strip().lower()
    if kind == "device":
        device_id = str(metadata.get("device_id") or identity).strip()
        if not device_id:
            raise DeviceTokenResolverError("device participant missing device_id")
        if mounts is None:
            raise DeviceTokenResolverError(
                "device participant requires the Kernel Device Mount resolver"
            )

        owner_id = str(metadata.get("owner_id") or "").strip()
        if not owner_id:
            raise DeviceTokenResolverError(
                "device participant missing trusted owner_id for Kernel scope"
            )
        connection = await mounts.resolve(owner_id=owner_id, device_id=device_id)
        if connection.owner_id != owner_id or connection.device_id != device_id:
            raise DeviceTokenResolverError("Kernel Device Mount owner/device mismatch")

        companion_id = str(
            metadata.get("companion_id") or connection.attached_companion_id or ""
        ).strip()
        if not companion_id:
            return connection
        context = await runtime.resolve_companion(companion_id, device_id=device_id)
        if (
            context.owner_id != owner_id
            or context.companion_id != companion_id
            or context.device_id != device_id
        ):
            raise DeviceTokenResolverError(
                "Companion runtime does not match mounted Device owner/target"
            )
        return CompanionInteractionContext(context, connection.mount_revision)
    if kind == "companion":
        companion_id = str(metadata.get("companion_id") or identity).strip()
        owner_id = str(metadata.get("owner_id") or "").strip()
        if not companion_id or not owner_id:
            raise DeviceTokenResolverError(
                "companion participant missing companion_id or trusted owner_id"
            )
        context = await runtime.resolve_companion(companion_id, device_id=None)
        if context.owner_id != owner_id or context.companion_id != companion_id:
            raise DeviceTokenResolverError("Companion runtime owner/identity mismatch")
        if context.device_id is not None:
            raise DeviceTokenResolverError("virtual Companion runtime unexpectedly has device")
        return CompanionInteractionContext(context)
    if kind == "owner":
        owner_id = str(metadata.get("owner_id") or identity).strip()
        if not owner_id:
            raise DeviceTokenResolverError("owner participant missing owner_id")
        return CompanionInteractionContext(await runtime.resolve_owner(owner_id))
    if kind == "user":
        owner_id = str(metadata.get("owner_id") or metadata.get("user_id") or identity).strip()
        if not owner_id:
            raise DeviceTokenResolverError("user participant missing owner_id/user_id")
        return CompanionInteractionContext(await runtime.resolve_owner(owner_id))
    raise DeviceTokenResolverError(
        f"participant.metadata.kind must be device/companion/owner/user for "
        f"identity={identity!r}; got {kind!r}."
    )


def make_device_token_resolver(
    *,
    room: Any,
    runtime: Any,
    mounts: Any | None = None,
    context_resolver: Callable[[Any], Awaitable[ResolvedContext]] | None = None,
    jwt_secret: str,
    jwt_algorithm: str = "HS256",
    ttl_seconds: int = 24 * 3600,
) -> DeviceTokenResolver:
    """Build the zero-arg async callable the gRPC LLM invokes lazily.

    The closure caches the resolved token after the first successful
    call — one authority lookup + one signing op per LK session, regardless
    of how many chat() turns happen. If the first call fails, the next
    one retries (we don't cache failures).

    **Identity changes mid-session are NOT re-resolved.** Device binding
    changes require a new LiveKit session so conversation history and memory
    stay inside one companion boundary.
    """
    cache: dict[str, str] = {}
    lock = asyncio.Lock()

    async def _resolve_uncached() -> str:
        peek = _participant_identity_and_metadata(room)
        if peek is None:
            raise DeviceTokenResolverError(
                "no remote participant yet — cannot resolve runtime token. "
                "This shouldn't happen at chat() time; the user should have "
                "spoken already. Check entrypoint ordering."
            )
        identity, metadata = peek

        try:
            if context_resolver is not None:
                context = CompanionInteractionContext(await context_resolver(room))
            else:
                context = await resolve_channel_context(
                    runtime=runtime,
                    mounts=mounts,
                    identity=identity,
                    metadata=metadata,
                )
        except SystemDataError as exc:
            raise DeviceTokenResolverError(
                f"System Data resolve failed for identity={identity!r} "
                f"kind={metadata.get('kind')!r}: {exc}"
            ) from exc
        if isinstance(context, DeviceConnectionContext):
            raise DeviceTokenResolverError(
                f"device {context.device_id!r} is mounted but has no companion target; "
                "keep it on the Device/data path or select an explicit same-Owner Companion"
            )
        ctx = context.runtime

        try:
            token, exp = sign_runtime_token(
                secret=jwt_secret,
                algorithm=jwt_algorithm,
                device_id=ctx.device_id,
                owner_id=ctx.owner_id,
                companion_id=ctx.companion_id,
                ttl_seconds=ttl_seconds,
            )
        except ValueError as exc:
            raise DeviceTokenResolverError(str(exc)) from exc

        cache["token"] = token
        _log.info(
            "resolved runtime token owner=%s companion=%s device=%s exp=%s",
            ctx.owner_id,
            ctx.companion_id,
            ctx.device_id,
            exp.isoformat(),
        )
        return token

    async def _resolve() -> str:
        token = cache.get("token")
        if token is not None:
            return token
        async with lock:
            token = cache.get("token")
            if token is not None:
                return token
            return await _resolve_uncached()

    return _resolve
