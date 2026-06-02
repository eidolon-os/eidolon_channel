"""HTTP wrapper over admin's ``/api/resolve`` aggregator.

Phase 32.B: when a participant joins the LiveKit room, channel queries
this client to compose the runtime context (user_id, agent_id,
template_id, tenant_id, memory_mcp_url) in ONE call rather than
chasing /api/users → /api/agents → /api/templates separately. Admin's
``/api/resolve`` was built in Phase 29.G for exactly this purpose.

The client supports both flows:

  - ``resolve_user(user_id)`` for web client (participant.metadata
    ``kind=user``)
  - ``resolve_device(device_id)`` for ESP32 (``kind=device``)

Error mapping mirrors the admin sub-project client conventions
elsewhere: 404 → :class:`AdminResolveNotFound`, 409/412 →
:class:`AdminResolvePrecondition` (user/device exists but isn't ready
to chat — no active_agent, no binding), connection failure →
:class:`AdminResolveUnreachable`, other 4xx/5xx →
:class:`AdminResolveUpstream`. Callers translate to participant-level
errors so the LK session ends with a clear log line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


# ---- exceptions ------------------------------------------------------------


class AdminResolveError(Exception):
    """Base — never raised directly, only subclasses."""


class AdminResolveNotFound(AdminResolveError):
    """admin returned 404 (user_id or device_id doesn't exist)."""


class AdminResolvePrecondition(AdminResolveError):
    """admin returned 409 or 412 — entity exists but isn't ready.

    Examples: user has no active_agent_id set (412), device exists in
    hub but admin's binding row is missing (412). Operator can fix via
    admin UI.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class AdminResolveUpstream(AdminResolveError):
    """admin returned a non-2xx response not specifically classified above."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"admin upstream {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class AdminResolveUnreachable(AdminResolveError):
    """Connection / DNS / timeout — admin process likely down."""


# ---- model -----------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedContext:
    """Subset of admin's ResolvedContext schema that channel actually
    uses. Keep this in sync with
    ``eidolon_admin_server/app/registry/schemas/resolve.py``.

    Extra fields admin returns (soul_preview, template_revision) are
    not used by channel today — we drop them from the dataclass so
    upstream additions don't accidentally become contract-binding."""

    tenant_id: str
    user_id: str
    agent_id: str
    template_id: str | None
    memory_mcp_url: str
    device_id: str | None  # None when resolved via /api/resolve/user

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResolvedContext":
        # Admin wraps the fields in ``{"context": {...}}`` (the
        # ResolveUserResponse / ResolveDeviceResponse envelope —
        # eidolon_admin_server/app/registry/schemas/resolve.py). Unwrap
        # here so callers can hand us either shape without each having
        # to know the envelope. Reaching ``data["context"]`` first means
        # we read the right fields instead of silently building an
        # all-empty ResolvedContext (Phase 32.B regression hunt 2026-06-03:
        # blank user_id flowed through to JWT → agent → memory.search
        # missed every recall).
        if isinstance(data.get("context"), dict):
            data = data["context"]
        return cls(
            tenant_id=str(data.get("tenant_id") or ""),
            user_id=str(data.get("user_id") or ""),
            agent_id=str(data.get("agent_id") or ""),
            template_id=data.get("template_id"),  # may be null
            memory_mcp_url=str(data.get("memory_mcp_url") or ""),
            device_id=data.get("device_id"),
        )


# ---- helpers ---------------------------------------------------------------


def _unwrap_detail(body: str) -> str:
    """FastAPI wraps errors as ``{"detail": "..."}``; strip the envelope."""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body
    if isinstance(parsed, dict) and "detail" in parsed:
        detail = parsed["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail)
    return body


# ---- client ----------------------------------------------------------------


class AdminResolveClient:
    """Async client. The caller owns the underlying httpx client and is
    responsible for closing it (typically at worker shutdown)."""

    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self._http = http
        self._base = base_url.rstrip("/")

    async def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base}{path}"
        try:
            r = await self._http.get(url, timeout=5.0)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AdminResolveUnreachable(f"admin GET {path} failed: {exc}") from exc

        if r.status_code == 404:
            raise AdminResolveNotFound(_unwrap_detail(r.text))
        if r.status_code in (409, 412):
            raise AdminResolvePrecondition(r.status_code, _unwrap_detail(r.text))
        if r.status_code >= 400:
            raise AdminResolveUpstream(r.status_code, _unwrap_detail(r.text))
        return r.json()

    async def resolve_user(self, user_id: str) -> ResolvedContext:
        """GET /api/resolve/user/{user_id} — composes via user.active_agent_id."""
        data = await self._get(f"/api/resolve/user/{quote(user_id, safe='')}")
        return ResolvedContext.from_json(data)

    async def resolve_device(self, device_id: str) -> ResolvedContext:
        """GET /api/resolve/device/{device_id} — composes via device.binding."""
        data = await self._get(f"/api/resolve/device/{quote(device_id, safe='')}")
        return ResolvedContext.from_json(data)
