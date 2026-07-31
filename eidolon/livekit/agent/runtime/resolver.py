"""Compose runtime identity lookup + JWT signing into one callable.

Phase 32.B: when ``EidolonAgentGrpcLlm`` opens its session (first
``chat()`` call), it invokes the device_token callable to get a fresh
bearer. The resolver here is what that callable does:

  1. Inspect the LiveKit room for a remote participant.
  2. Dispatch by ``participant.metadata.kind``.
  3. Resolve the entrance through Eidolon Data/Admin into the explicit
     owner/companion/runtime identity envelope.
  4. Sign a runtime JWT with owner/companion/memory_realm/genome so
     eidolon_agent accepts it.
  5. Cache the result for the lifetime of this resolver instance —
     subsequent invocations within the session return the same token.

If metadata is missing or admin says no, the resolver raises a clear error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from eidolon_sdk.biz.admin import AdminResolveError, ResolvedContext
from eidolon_sdk.biz.runtime import sign_runtime_token

_log = logging.getLogger(__name__)

DeviceTokenResolver = Callable[[], Awaitable[str]]
RuntimeTokenResolver = DeviceTokenResolver
RUNTIME_PARTICIPANT_KINDS = frozenset({"device", "owner", "user"})


class DeviceTokenResolverError(Exception):
    """Resolver couldn't produce a token — caller decides whether to
    raise to the participant or fall back to a static token."""


class RoomNotConnectedError(DeviceTokenResolverError):
    """Runtime participant resolution requires an active room connection."""


def _participant_identity_and_metadata(
    room: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Select the runtime actor rather than the first room participant.

    Non-publishing infrastructure participants (for example the Hub control
    bridge) are not voice-session actors and must never own RoomIO, identity,
    memory, or runtime-token resolution.  Explicit actor metadata wins even if
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
    invalid_actor: tuple[str, dict[str, Any]] | None = None
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
        if invalid_actor is None:
            invalid_actor = (identity, metadata)
    return invalid_actor


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
    """Wait until an explicitly typed voice-session actor is in the room."""
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
    """Wait for and return the explicitly typed voice-session actor.

    Infrastructure participants may join first.  Returning identity and
    metadata from the same selection pass prevents session construction from
    accidentally reading metadata from the Hub control bridge.
    """

    if room is None or not room.isconnected():
        raise RoomNotConnectedError(
            "wait_for_runtime_participant_metadata() called on an unconnected "
            "room; connect the job (JobContext.connect()) before resolving the "
            "runtime actor"
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
                "no remote participant with runtime actor metadata "
                "(kind=device/owner/user) became available"
            )
        await asyncio.sleep(max(0.0, poll_interval_sec))


async def _resolve_context(
    *,
    admin: Any,
    identity: str,
    metadata: dict[str, Any],
) -> tuple[str, str, ResolvedContext]:
    """Resolve a participant into ``(actor_kind, actor_id, context)``."""
    kind = str(metadata.get("kind") or "").strip().lower()
    if kind == "device":
        device_id = str(metadata.get("device_id") or identity).strip()
        if not device_id:
            raise DeviceTokenResolverError("device participant missing device_id")
        return "device", device_id, await admin.resolve_device(device_id)
    if kind == "owner":
        owner_id = str(metadata.get("owner_id") or identity).strip()
        if not owner_id:
            raise DeviceTokenResolverError("owner participant missing owner_id")
        return "owner", owner_id, await admin.resolve_owner(owner_id)
    if kind == "user":
        owner_id = str(
            metadata.get("owner_id") or metadata.get("user_id") or identity
        ).strip()
        if not owner_id:
            raise DeviceTokenResolverError("user participant missing owner_id/user_id")
        return "owner", owner_id, await admin.resolve_owner(owner_id)
    raise DeviceTokenResolverError(
        f"participant.metadata.kind must be one of 'device' or 'owner' for "
        f"identity={identity!r}; got {kind!r}."
    )


def make_device_token_resolver(
    *,
    room: Any,
    admin: Any,
    jwt_secret: str,
    jwt_algorithm: str = "HS256",
    ttl_seconds: int = 24 * 3600,
) -> DeviceTokenResolver:
    """Build the zero-arg async callable the gRPC LLM invokes lazily.

    The closure caches the resolved token after the first successful
    call — one admin lookup + one signing op per LK session, regardless
    of how many chat() turns happen. If the first call fails, the next
    one retries (we don't cache failures).

    **Identity changes mid-session are NOT re-resolved.** Device binding
    changes require a new LiveKit session so conversation history and memory
    stay inside one companion boundary.
    """
    cache: dict[str, str] = {}

    async def _resolve() -> str:
        if "token" in cache:
            return cache["token"]

        peek = _participant_identity_and_metadata(room)
        if peek is None:
            raise DeviceTokenResolverError(
                "no remote participant yet — cannot resolve runtime token. "
                "This shouldn't happen at chat() time; the user should have "
                "spoken already. Check entrypoint ordering."
            )
        identity, metadata = peek

        try:
            actor_kind, actor_id, ctx = await _resolve_context(
                admin=admin, identity=identity, metadata=metadata
            )
        except AdminResolveError as exc:
            raise DeviceTokenResolverError(
                f"admin resolve failed for identity={identity!r} "
                f"kind={metadata.get('kind')!r}: {exc}"
            ) from exc

        try:
            token, exp = sign_runtime_token(
                secret=jwt_secret,
                algorithm=jwt_algorithm,
                actor_kind=actor_kind,
                actor_id=actor_id,
                device_id=ctx.device_id,
                owner_id=ctx.owner_id,
                companion_id=ctx.companion_id,
                memory_realm_id=ctx.memory_realm_id,
                genome_id=ctx.genome_id,
                schema_version=ctx.schema_version,
                genome_hash=ctx.genome_hash,
                realizer_version=ctx.realizer_version,
                ttl_seconds=ttl_seconds,
            )
        except ValueError as exc:
            raise DeviceTokenResolverError(str(exc)) from exc

        cache["token"] = token
        _log.info(
            "resolved runtime token actor=%s:%s owner=%s companion=%s device=%s genome=%s exp=%s",
            actor_kind,
            actor_id,
            ctx.owner_id,
            ctx.companion_id,
            ctx.device_id,
            ctx.genome_hash,
            exp.isoformat(),
        )
        return token

    return _resolve
