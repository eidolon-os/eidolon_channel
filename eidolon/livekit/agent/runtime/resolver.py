"""Compose admin lookup + JWT signing into one callable.

Phase 32.B: when ``EidolonAgentGrpcLlm`` opens its session (first
``chat()`` call), it invokes the device_token callable to get a fresh
bearer. The resolver here is what that callable does:

  1. Inspect the LiveKit room for a remote participant.
  2. Parse ``participant.metadata`` → ``kind`` (``user`` | ``device``).
  3. GET ``/api/resolve/{kind}/{identity}`` on admin (one HTTP call —
     admin composes user_id → agent → template → memory_url for us).
  4. Sign a device JWT with the resolved (tenant, user, template) so
     agent's ``PairingTokenVerifier`` accepts it.
  5. Cache the result for the lifetime of this resolver instance —
     subsequent invocations within the session return the same token.

If metadata is missing or admin says no, the caller can choose to fall
back to a legacy static token (see ``factory.py``). The resolver itself
does NOT fall back — it raises clear errors so the caller decides.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Awaitable, Callable

from eidolon_sdk.biz.admin import (
    AdminResolveClient,
    AdminResolveError,
    ResolvedContext,
)
from eidolon_sdk.biz.runtime import sign_device_token

_log = logging.getLogger(__name__)

DeviceTokenResolver = Callable[[], Awaitable[str]]


class DeviceTokenResolverError(Exception):
    """Resolver couldn't produce a token — caller decides whether to
    raise to the participant or fall back to a static token."""


def _participant_identity_and_metadata(
    room: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Read the first remote participant's identity + metadata dict.

    Returns ``None`` if no participant is connected yet or metadata is
    unparseable; the caller decides how to handle (fallback / retry /
    raise). Robust to LiveKit Room/RemoteParticipant API differences
    (sync vs async, dict vs list).
    """
    if room is None:
        return None
    try:
        participants = list(getattr(room, "remote_participants", {}).values())
    except Exception:  # noqa: BLE001 — defensive
        return None
    if not participants:
        return None
    p = participants[0]
    identity = (getattr(p, "identity", "") or "").strip()
    if not identity:
        return None
    raw_meta = getattr(p, "metadata", "") or ""
    if not raw_meta:
        return identity, {}
    try:
        parsed = json.loads(raw_meta)
        if isinstance(parsed, dict):
            return identity, parsed
    except (ValueError, TypeError):
        pass
    return identity, {}


async def _resolve_context(
    *,
    admin: AdminResolveClient,
    identity: str,
    metadata: dict[str, Any],
) -> ResolvedContext:
    """Dispatch to /api/resolve/user or /api/resolve/device based on
    ``metadata.kind``.

    Accepted values: ``user`` | ``device``. Anything else (including
    empty / missing) raises :class:`DeviceTokenResolverError` — no
    silent fallback (Phase 33.A5 tightening, 29.H principle).

    Both supported callers tag the participant correctly:
      - hub web path: kind=user (Phase 32.A)
      - hub esp32 path: kind=device (Phase 32.B follow-up)

    If you're seeing this error: either the caller is a new client
    type that needs hub to start tagging it, or a legacy ESP32
    firmware that hasn't been updated since the pre-Phase-32 era.
    Don't reintroduce the device default — fix the caller.
    """
    kind = str(metadata.get("kind") or "").strip().lower()
    if kind == "user":
        return await admin.resolve_user(identity)
    if kind == "device":
        return await admin.resolve_device(identity)
    raise DeviceTokenResolverError(
        f"participant.metadata.kind missing or unknown ({kind!r}) for "
        f"identity={identity!r}. All clients must set kind=user|device; "
        "see hub /api/config for the tagging contract. Legacy ESP32 "
        "firmware predating Phase 32.B must be updated."
    )


def make_device_token_resolver(
    *,
    room: Any,
    admin: AdminResolveClient,
    jwt_secret: str,
    jwt_algorithm: str = "HS256",
    ttl_seconds: int = 24 * 3600,
) -> DeviceTokenResolver:
    """Build the zero-arg async callable the gRPC LLM invokes lazily.

    The closure caches the resolved token after the first successful
    call — one admin lookup + one signing op per LK session, regardless
    of how many chat() turns happen. If the first call fails, the next
    one retries (we don't cache failures).

    **Identity changes mid-session are NOT re-resolved.**

    Once a session's first turn has bound the token to ``(user_id,
    agent_id, tenant_id)``, that binding is sticky for the remainder of
    the LiveKit room — even if the participant updates their metadata
    or the operator switches the user's ``active_agent`` from the admin
    UI. The reasons for "sticky" being the right default:

      - LiveKit sessions are short-lived (a single conversation turn
        sequence). Refreshing the token mid-turn would split an
        audio/text stream between two different agent backends.
      - Admin's ``set_active_agent`` semantics: switching ``active``
        affects *future* sessions, not in-flight ones. This matches
        what operators expect — the admin UI hint in Phase 33.B4 makes
        that contract visible.
      - Hot-rotating mid-session would require revalidating cached
        prompts, memory threads, and audio context — a separate (and
        much larger) feature. The escape hatch today is to revoke the
        user's sessions (POST ``/api/admin/users/{id}/revoke-sessions``
        in agent, see Phase 33.B1) which forces a reconnect.

    If you need the new identity to take effect *now*: end the session
    (close the LK room) and start a new one. The token will be
    re-resolved on the next chat() call's first invocation.
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
            ctx = await _resolve_context(
                admin=admin, identity=identity, metadata=metadata
            )
        except AdminResolveError as exc:
            raise DeviceTokenResolverError(
                f"admin resolve failed for identity={identity!r} "
                f"kind={metadata.get('kind')!r}: {exc}"
            ) from exc

        # Channel mints a fresh runtime device_id distinct from the LK
        # participant.identity. This gives agent's audit logs a "session
        # came in via channel-worker" handle without conflating it with
        # the underlying user/device identity.
        runtime_device_id = (
            ctx.device_id or f"channel-{uuid.uuid4().hex[:8]}"
        )
        try:
            token, exp = sign_device_token(
                secret=jwt_secret,
                algorithm=jwt_algorithm,
                device_id=runtime_device_id,
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                template_id=ctx.template_id,
                ttl_seconds=ttl_seconds,
            )
        except ValueError as exc:
            raise DeviceTokenResolverError(str(exc)) from exc

        cache["token"] = token
        _log.info(
            "resolved device token user=%s agent=%s tenant=%s exp=%s",
            ctx.user_id, ctx.agent_id, ctx.tenant_id, exp.isoformat(),
        )
        return token

    return _resolve
