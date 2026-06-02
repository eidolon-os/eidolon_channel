"""Sign device JWTs that ``eidolon_agent``'s ``PairingTokenVerifier``
will accept.

Phase 32.B: this is THE same payload schema agent's own ``sign_device_token``
emits, kept in this repo because channel doesn't want a hard import
dependency on ``eidolon_agent`` (separate venv, separate deploy unit).
The two sign functions trust each other via a shared HMAC secret —
``PAIRING_JWT_SECRET`` env var, falling back to
``~/eidolon/run/jwt-secret`` (the file agent persists when its own env
is empty).

If you change the payload here, also change it in
``eidolon_agent/app/transport/pairing/token.py`` — and bump tests on
both sides.

**Drift sentinel** (Phase 33.A1): the cross-project contract is pinned
by ``eidolon_admin/server/tests/test_runtime_token_contract.py``. That
test loads THIS file and agent's verifier, signs+verifies, and asserts
every field round-trips. If you break it, CI fails before the runtime
breaks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import jwt


_SHARED_SECRET_FILE = Path("~/eidolon/run/jwt-secret").expanduser()


def resolve_shared_secret(env_value: str = "") -> str:
    """Resolve the HMAC secret from (in order): explicit env_value
    argument, ``~/eidolon/run/jwt-secret`` file, empty string.

    Returning an empty string is intentional — callers (factory) decide
    whether to hard-fail or fall back to the legacy static token path.
    """
    val = env_value.strip()
    if val:
        return val
    if _SHARED_SECRET_FILE.is_file():
        try:
            return _SHARED_SECRET_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def sign_device_token(
    *,
    secret: str,
    algorithm: str = "HS256",
    device_id: str,
    tenant_id: str,
    user_id: str,
    template_id: str | None,
    scopes: Sequence[str] = ("device",),
    ttl_seconds: int = 24 * 3600,
) -> tuple[str, datetime]:
    """Return ``(token, exp_datetime)``.

    Payload mirrors agent-side ``sign_device_token`` exactly. ``ttl``
    default is 1 day — channel re-mints on each LK session start, so a
    short window keeps blast radius bounded if the secret leaks.
    """
    if not secret:
        raise ValueError("sign_device_token: secret is required (empty)")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=ttl_seconds)
    payload = {
        "device_id": device_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "template_id": template_id,
        "scopes": list(scopes),
        "jti": uuid.uuid4().hex,
        "exp": int(exp.timestamp()),
        "iat": int(now.timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token, exp
