"""Where this Host's own speech synthesis answers.

Resolved, never configured — for the reason the recognizer's endpoint module
beside this one gives at length: the port is reserved once, by the component
that serves it, and Ops writes it into a Host's registry for the capabilities
that Host declares. A number in this plugin's configuration would be a second
statement of a fact that already has an author.

The host is loopback because the component's contract binds it there: local
synthesis is for the machine it runs on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from eidolon_sdk.biz.contracts import local_tts as contract

#: Named in the sealed Host profile every unit reads. The same variable Admin
#: and the recognizer resolve their ports through.
PORTS_FILE_ENV = "EIDOLON_PORTS_FILE"

#: Where a capability's port roles land in that registry: derived per Host,
#: separate from the curated baseline, in the flat shape they are derived in.
PORT_ROLES_SECTION = "port_roles"

LOOPBACK = "127.0.0.1"


class LocalTtsEndpointError(RuntimeError):
    """This Host does not say where its local synthesis is."""


def resolve_port(
    *,
    registry_path: str | os.PathLike[str] | None = None,
    role: str = contract.LOCAL_TTS_PORT_ROLE,
) -> int:
    """The port this Host reserves for the role, or a refusal that says why.

    A missing entry is not a reason to fall back to a literal. The role exists
    in a Host's registry exactly when that Host declares the capability that
    brings it, so its absence means there is no local synthesis here to reach —
    and guessing a number would turn that into a connection error against
    whatever else happens to be listening.
    """

    configured = registry_path or os.environ.get(PORTS_FILE_ENV, "").strip()
    if not configured:
        raise LocalTtsEndpointError(
            f"${PORTS_FILE_ENV} is not set, so this process cannot ask which port "
            f"this Host reserves for {role!r}. On a product Host the sealed Host "
            "profile names it."
        )
    path = Path(configured)
    try:
        document: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise LocalTtsEndpointError(f"port registry is unreadable: {path}") from exc
    except yaml.YAMLError as exc:
        raise LocalTtsEndpointError(f"port registry is not valid YAML: {path}") from exc

    roles = document.get(PORT_ROLES_SECTION) if isinstance(document, dict) else None
    value = roles.get(role) if isinstance(roles, dict) else None
    if value is None:
        raise LocalTtsEndpointError(
            f"{path} reserves no {role!r}. That section is written for the "
            f"capabilities a Host declares, so this Host does not provide "
            f"{contract.LOCAL_TTS_CAPABILITY!r} and has no local synthesis to "
            "reach. Declare the capability for this Host, or point the provider "
            "at one that is not local."
        )
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise LocalTtsEndpointError(
            f"{path} gives {role!r} as {value!r}, which is not a port"
        ) from exc
    if not 1 <= port <= 65535:
        raise LocalTtsEndpointError(f"{path} gives {role!r} as {port}, out of range")
    return port


def resolve_stream_url(
    *,
    registry_path: str | os.PathLike[str] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> str:
    """The stream URL, composed from the protocol's path and this Host's port.

    `host` defaults late rather than in the signature: a default bound at import
    time is a snapshot of the constant, not the constant, and cannot be pointed
    elsewhere by anything that reasonably might — a test harness reaching a
    board through a forward, for one.
    """

    resolved = port if port is not None else resolve_port(registry_path=registry_path)
    return contract.stream_url(host or LOOPBACK, resolved)


def resolve_ready_url(
    *,
    registry_path: str | os.PathLike[str] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> str:
    """Where the service says whether its engine is warm."""

    resolved = port if port is not None else resolve_port(registry_path=registry_path)
    return f"http://{host or LOOPBACK}:{resolved}{contract.LOCAL_TTS_READY_PATH}"
