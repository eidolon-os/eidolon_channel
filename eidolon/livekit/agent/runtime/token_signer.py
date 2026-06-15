"""Channel compatibility wrapper for SDK runtime device token signing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from eidolon_sdk.runtime import (
    resolve_shared_secret as _sdk_resolve_shared_secret,
    sign_device_token as _sdk_sign_device_token,
)


_SHARED_SECRET_FILE = Path("~/eidolon/run/jwt-secret").expanduser()


def resolve_shared_secret(env_value: str = "") -> str:
    return _sdk_resolve_shared_secret(env_value, secret_file=_SHARED_SECRET_FILE)


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
    return _sdk_sign_device_token(
        secret=secret,
        algorithm=algorithm,
        device_id=device_id,
        tenant_id=tenant_id,
        user_id=user_id,
        template_id=template_id,
        scopes=scopes,
        ttl_seconds=ttl_seconds,
    )
