"""Who may call which Channel Provider route: declared once, enforced once.

Three callers reach this process over HTTP, and each holds the credential of
its own role and no other:

``hub``
    Hub's control plane. It admits devices, so it is what provisions, reads
    and revokes their channels, and what opens and closes the conversations it
    brokers for an Owner.
``management``
    Admin, acting for the Owner — from its own pages, or for the phone through
    the Local API. Presence, session traces, voiceprints, voices and a device's
    own settings.
``worker``
    This repository's Channel worker, from inside a running conversation: the
    smart-home result it hands a panel, the speaker extraction recognition
    runs, and the voice a session resolves as it starts.

Until this module there was one bearer for every route, and it was Hub's.
Admin held a copy to read presence, and that copy could equally provision and
revoke any device's channel or open and close its conversations; the worker
held the same one. A credential is now a role, the table below says which
roles each route serves, and one middleware refuses everything else — so no
handler authenticates, and a route that nobody declared cannot be served.

A route two callers share lists both, rather than either borrowing the
other's credential.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from aiohttp import web

from .contracts import Forbidden, Unauthenticated
from .problems import problem

logger = logging.getLogger("eidolon.channel_provider.access")

#: Each credential is random material Ops mints; anything shorter was typed.
MINIMUM_CREDENTIAL_BYTES = 32


class Role(StrEnum):
    HUB = "hub"
    MANAGEMENT = "management"
    WORKER = "worker"


#: Where each role's credential arrives. Ops writes exactly these names
#: (``eidolon_ops/install_inputs.py``): the Provider's environment holds all
#: three, and each caller's holds only its own under the same name.
CREDENTIAL_ENV: Mapping[Role, str] = MappingProxyType(
    {
        Role.HUB: "EIDOLON_CHANNEL_PROVIDER_HUB_TOKEN",
        Role.MANAGEMENT: "EIDOLON_CHANNEL_PROVIDER_MANAGEMENT_TOKEN",
        Role.WORKER: "EIDOLON_CHANNEL_PROVIDER_WORKER_TOKEN",
    }
)

_HUB = frozenset({Role.HUB})
_MANAGEMENT = frozenset({Role.MANAGEMENT})
_WORKER = frozenset({Role.WORKER})

#: Answered without a credential. Health carries no Owner state; probes and the
#: service registry read it before anything else is known.
PUBLIC_ROUTES: frozenset[tuple[str, str]] = frozenset({("GET", "/health")})

#: Every authenticated route, keyed by method and aiohttp's canonical path, and
#: the roles it serves. The callers named in each comment are the ones that
#: exist; a role that no caller needs is not granted in case one appears.
ROUTE_ROLES: Mapping[tuple[str, str], frozenset[Role]] = MappingProxyType(
    {
        # Hub — the channels of the devices it admits
        # (hub/channel_reconciliation/provider_client.py).
        ("POST", "/v1/device-channels/current"): _HUB,
        ("POST", "/v1/device-channels/provision"): _HUB,
        ("POST", "/v1/device-channels/revoke"): _HUB,
        # The per-device session contract Hub owned before device-sessions;
        # no caller today, and it acts on a channel, so it stays Hub's.
        ("POST", "/v1/device-channels/sessions/open"): _HUB,
        ("POST", "/v1/device-channels/sessions/close"): _HUB,
        # Hub — conversations it brokers for an Owner.
        ("POST", "/v1/device-conversations/{action}"): _HUB,
        ("POST", "/v1/device-sessions/{action}"): _HUB,
        ("POST", "/v1/role-groups/{action}"): _HUB,
        ("POST", "/v1/shared-sessions/{action}"): _HUB,
        # Management — Admin, for the Owner and for the phone via Local API
        # (eidolon_admin_server/app/management/*, app/mission_control).
        ("GET", "/v1/device-channels/presence"): _MANAGEMENT,
        ("GET", "/v1/device-channels/{device_id}/properties"): _MANAGEMENT,
        ("GET", "/v1/device-channels/{device_id}/offer"): _MANAGEMENT,
        ("POST", "/v1/device-channels/{device_id}/capabilities/{capability}"): _MANAGEMENT,
        ("GET", "/v1/session-traces"): _MANAGEMENT,
        ("GET", "/v1/session-traces/{session_id}"): _MANAGEMENT,
        ("GET", "/v1/voiceprints/readiness"): _MANAGEMENT,
        ("POST", "/v1/voiceprints/evaluate-enrollment"): _MANAGEMENT,
        ("POST", "/v1/voiceprints/compare"): _MANAGEMENT,
        ("POST", "/v1/voiceprints/captures"): _MANAGEMENT,
        ("POST", "/v1/voiceprints/captures/{capture_id}/finish"): _MANAGEMENT,
        ("DELETE", "/v1/voiceprints/captures/{capture_id}"): _MANAGEMENT,
        ("GET", "/v1/speech-voices"): _MANAGEMENT,
        ("POST", "/v1/speech-voices/refresh"): _MANAGEMENT,
        ("POST", "/v1/speech-voices/previews"): _MANAGEMENT,
        ("POST", "/v1/speech-voices/references/check"): _MANAGEMENT,
        ("POST", "/v1/speech-voices/enrollments/design"): _MANAGEMENT,
        ("POST", "/v1/speech-voices/enrollments/clone"): _MANAGEMENT,
        ("GET", "/v1/speech-voices/enrollments"): _MANAGEMENT,
        ("GET", "/v1/speech-voices/enrollments/{voice_id}"): _MANAGEMENT,
        ("DELETE", "/v1/speech-voices/enrollments/{voice_id}"): _MANAGEMENT,
        # Shared. Admin extracts a registration sample and resolves the voice
        # it shows the Owner; the worker extracts a conversation probe
        # (session/speaker_extraction_client.py) and resolves the voice a
        # session starts with when dispatch did not carry it
        # (session/speech_voice.py).
        ("POST", "/v1/voiceprints/extract"): _MANAGEMENT | _WORKER,
        ("POST", "/v1/speech-voices/resolve"): _MANAGEMENT | _WORKER,
        # Worker — the smart-home turn's result, for the device's panel
        # (livekit/agent/smarthome.py).
        ("POST", "/v1/smarthome/{action}"): _WORKER,
    }
)


@dataclass(frozen=True, slots=True, repr=False)
class RoleCredentials:
    """The three credentials this Provider accepts, one per role.

    ``repr=False`` so a credential set that reaches a log or a traceback says
    nothing about its values.
    """

    hub: str
    management: str
    worker: str

    def __post_init__(self) -> None:
        held = {role: getattr(self, role.value) for role in Role}
        for role, value in held.items():
            if len(value.encode()) < MINIMUM_CREDENTIAL_BYTES:
                raise ValueError(
                    f"{CREDENTIAL_ENV[role]} must contain at least "
                    f"{MINIMUM_CREDENTIAL_BYTES} bytes"
                )
        # One value for two roles would make the role a guess: whichever of
        # them matched first would be the answer.
        if len(set(held.values())) != len(held):
            raise ValueError("Channel Provider role credentials must be distinct")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> RoleCredentials:
        return cls(**{role.value: environ.get(CREDENTIAL_ENV[role], "") for role in Role})

    def role_of(self, authorization: str) -> Role | None:
        """Which role presented this ``Authorization`` header, if any."""

        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return None
        presented = authorization[len(prefix) :].encode()
        matched = None
        # Every role is compared, so the time taken does not say which one a
        # near miss was closest to.
        for role in Role:
            if hmac.compare_digest(presented, getattr(self, role.value).encode()):
                matched = role
        return matched


def _route_key(method: str, canonical: str) -> tuple[str, str]:
    # aiohttp serves HEAD beside every GET; it asks what the GET would.
    return ("GET" if method == "HEAD" else method, canonical)


def registered_routes(app: web.Application) -> set[tuple[str, str]]:
    return {
        _route_key(route.method, route.resource.canonical)
        for route in app.router.routes()
        if route.resource is not None
    }


def undeclared_routes(routes: Iterable[tuple[str, str]]) -> list[str]:
    return sorted(
        f"{method} {path}"
        for method, path in routes
        if (method, path) not in ROUTE_ROLES and (method, path) not in PUBLIC_ROUTES
    )


def protect(app: web.Application, credentials: RoleCredentials) -> None:
    """Serve each route only to the roles declared for it. Call after routing.

    Refuses to protect an application with a route this module does not
    declare. The middleware would refuse such a route anyway; stopping here
    says so when the route is added, rather than as a 403 in production.
    """

    missing = undeclared_routes(registered_routes(app))
    if missing:
        raise RuntimeError("routes with no declared caller: " + ", ".join(missing))
    app.middlewares.append(_authorize(credentials))


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def _authorize(credentials: RoleCredentials):
    @web.middleware
    async def authorize(request: web.Request, handler: Handler) -> web.StreamResponse:
        resource = request.match_info.route.resource
        if resource is None:
            # Nothing matched: aiohttp's own 404 or 405, which names no route.
            return await handler(request)
        key = _route_key(request.method, resource.canonical)
        if key in PUBLIC_ROUTES:
            return await handler(request)
        role = credentials.role_of(request.headers.get("Authorization", ""))
        if role is None:
            return problem(Unauthenticated("bearer credential was not accepted"))
        if role not in ROUTE_ROLES.get(key, frozenset()):
            # A known credential at the wrong door is a deployment that handed
            # a caller the wrong role — worth a line, never the value.
            logger.warning("refused %s %s to the %s credential", *key, role)
            return problem(Forbidden(f"the {role} credential does not serve {key[0]} {key[1]}"))
        return await handler(request)

    return authorize
