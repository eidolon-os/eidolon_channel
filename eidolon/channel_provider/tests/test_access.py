"""Each caller's credential reaches its own routes and nothing else.

The policy is tested on its own (every declared route against every role, on
an application whose handlers always succeed), the Provider is tested to serve
exactly the routes the policy declares, and the callers that exist today are
listed with the routes they call — so dropping a role a real caller needs
fails here, not on a Host.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.access import (
    CREDENTIAL_ENV,
    PUBLIC_ROUTES,
    ROUTE_ROLES,
    Role,
    RoleCredentials,
    protect,
    registered_routes,
)
from eidolon.channel_provider.http import create_app

from .helpers import ROLE_CREDENTIALS, bearer


def _concrete(path: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: "open" if m[1] == "action" else f"some-{m[1]}", path)


async def _policy_client() -> TestClient:
    async def succeed(_request: web.Request) -> web.Response:
        return web.Response(status=204)

    app = web.Application()
    for method, path in (*ROUTE_ROLES, *PUBLIC_ROUTES):
        if method == "GET":
            app.router.add_get(path, succeed)  # with the HEAD aiohttp adds, as the Provider has
        else:
            app.router.add_route(method, path, succeed)
    protect(app, ROLE_CREDENTIALS)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.fixture
async def policy():
    client = await _policy_client()
    try:
        yield client
    finally:
        await client.close()


async def test_each_route_serves_exactly_the_roles_declared_for_it(policy) -> None:
    for (method, path), roles in ROUTE_ROLES.items():
        for role in Role:
            response = await policy.request(method, _concrete(path), headers=bearer(role))
            if role in roles:
                assert response.status == 204, (method, path, role)
            else:
                assert response.status == 403, (method, path, role)
                body = await response.json()
                assert body["code"] == "FORBIDDEN"
                assert body["authority"] == "eidolon-channel-provider"


async def test_no_credential_or_an_unknown_one_is_unauthenticated(policy) -> None:
    for method, path in ROUTE_ROLES:
        for headers in (
            {},
            {"Authorization": "Bearer " + "x" * 40},
            # The right value under another scheme is not a bearer credential.
            {"Authorization": "Token " + ROLE_CREDENTIALS.management},
        ):
            response = await policy.request(method, _concrete(path), headers=headers)
            assert response.status == 401, (method, path, headers)
            assert (await response.json())["code"] == "UNAUTHENTICATED"


async def test_health_needs_no_credential(policy) -> None:
    assert (await policy.get("/health")).status == 204
    assert (await policy.head("/health")).status == 204


async def test_head_is_answered_as_the_get_beside_it(policy) -> None:
    assert (await policy.head("/v1/device-channels/presence", headers=bearer(Role.HUB))).status == 403
    assert (
        await policy.head("/v1/device-channels/presence", headers=bearer(Role.MANAGEMENT))
    ).status == 204


#: Every call a caller makes today, from the repository that makes it. The
#: point is the left column: a role a real caller needs cannot be withdrawn
#: from a route without this failing.
CALLS = (
    # Hub — hub/channel_reconciliation/provider_client.py
    (Role.HUB, "POST", "/v1/device-channels/current"),
    (Role.HUB, "POST", "/v1/device-channels/provision"),
    (Role.HUB, "POST", "/v1/device-channels/revoke"),
    (Role.HUB, "POST", "/v1/role-groups/open"),
    (Role.HUB, "POST", "/v1/role-groups/status"),
    (Role.HUB, "POST", "/v1/role-groups/close"),
    (Role.HUB, "POST", "/v1/device-conversations/open"),
    (Role.HUB, "POST", "/v1/device-conversations/status"),
    (Role.HUB, "POST", "/v1/device-conversations/close"),
    (Role.HUB, "POST", "/v1/device-sessions/open"),
    (Role.HUB, "POST", "/v1/device-sessions/status"),
    (Role.HUB, "POST", "/v1/device-sessions/close"),
    (Role.HUB, "POST", "/v1/shared-sessions/open"),
    (Role.HUB, "POST", "/v1/shared-sessions/close"),
    # Admin — app/management/device_presence_router.py, app/mission_control
    (Role.MANAGEMENT, "GET", "/v1/device-channels/presence"),
    # Admin — app/management/device_settings_router.py
    (Role.MANAGEMENT, "GET", "/v1/device-channels/device-1/properties"),
    (Role.MANAGEMENT, "POST", "/v1/device-channels/device-1/capabilities/volume.set"),
    (Role.MANAGEMENT, "GET", "/v1/device-channels/device-1/offer"),
    # Admin — app/management/session_traces_router.py
    (Role.MANAGEMENT, "GET", "/v1/session-traces"),
    (Role.MANAGEMENT, "GET", "/v1/session-traces/session-1"),
    # Admin — app/management/voiceprints_router.py
    (Role.MANAGEMENT, "GET", "/v1/voiceprints/readiness"),
    (Role.MANAGEMENT, "POST", "/v1/voiceprints/extract"),
    (Role.MANAGEMENT, "POST", "/v1/voiceprints/evaluate-enrollment"),
    (Role.MANAGEMENT, "POST", "/v1/voiceprints/compare"),
    (Role.MANAGEMENT, "POST", "/v1/voiceprints/captures"),
    (Role.MANAGEMENT, "POST", "/v1/voiceprints/captures/capture-1/finish"),
    (Role.MANAGEMENT, "DELETE", "/v1/voiceprints/captures/capture-1"),
    # Admin — app/management/speech_voices.py
    (Role.MANAGEMENT, "GET", "/v1/speech-voices"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/refresh"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/resolve"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/previews"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/references/check"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/enrollments/design"),
    (Role.MANAGEMENT, "POST", "/v1/speech-voices/enrollments/clone"),
    (Role.MANAGEMENT, "GET", "/v1/speech-voices/enrollments"),
    (Role.MANAGEMENT, "GET", "/v1/speech-voices/enrollments/voice-1"),
    (Role.MANAGEMENT, "DELETE", "/v1/speech-voices/enrollments/voice-1"),
    # Worker — livekit/agent/smarthome.py
    (Role.WORKER, "POST", "/v1/smarthome/result"),
    # Worker — livekit/agent/session/speaker_extraction_client.py
    (Role.WORKER, "POST", "/v1/voiceprints/extract"),
    # Worker — livekit/agent/session/speech_voice.py
    (Role.WORKER, "POST", "/v1/speech-voices/resolve"),
)


async def test_every_call_a_caller_makes_today_is_served(policy) -> None:
    for role, method, path in CALLS:
        response = await policy.request(method, path, headers=bearer(role))
        assert response.status == 204, (role, method, path)


async def test_management_reaches_no_channel_lifecycle_and_no_conversation(policy) -> None:
    """The reason this module exists: Admin's credential used to be Hub's."""

    for role, method, path in CALLS:
        if role is not Role.HUB:
            continue
        for other in (Role.MANAGEMENT, Role.WORKER):
            response = await policy.request(method, path, headers=bearer(other))
            assert response.status == 403, (other, method, path)


async def test_hub_reaches_nothing_management_or_the_worker_calls(policy) -> None:
    for role, method, path in CALLS:
        if role is Role.HUB:
            continue
        response = await policy.request(method, path, headers=bearer(Role.HUB))
        assert response.status == 403, (method, path)


async def test_the_worker_reaches_only_its_three_routes(policy) -> None:
    served = {
        (method, path)
        for method, path in ROUTE_ROLES
        if Role.WORKER in ROUTE_ROLES[(method, path)]
    }
    assert served == {
        ("POST", "/v1/smarthome/{action}"),
        ("POST", "/v1/voiceprints/extract"),
        ("POST", "/v1/speech-voices/resolve"),
    }


class _Service:
    """Comes up and goes down; every request here is decided before it."""

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


def _full_provider() -> web.Application:
    """The Provider as ``server.py`` composes it, every optional surface on."""

    async def run() -> None:
        return None

    catalog = SimpleNamespace(run=run, adapter=None)
    return create_app(
        service=_Service(),
        credentials=ROLE_CREDENTIALS,
        speaker_engine=None,
        voice_capture=None,
        speech_voices=(catalog, SimpleNamespace()),
    )


def test_the_provider_serves_exactly_the_routes_the_policy_declares() -> None:
    assert registered_routes(_full_provider()) == set(ROUTE_ROLES) | PUBLIC_ROUTES


async def test_the_composed_provider_enforces_the_policy() -> None:
    async with TestClient(TestServer(_full_provider())) as client:
        for role in (Role.MANAGEMENT, Role.WORKER):
            for path in ("/v1/device-channels/provision", "/v1/device-channels/revoke"):
                response = await client.post(path, json={}, headers=bearer(role))
                assert response.status == 403, (role, path)
        response = await client.get("/v1/device-channels/presence", headers=bearer(Role.HUB))
        assert response.status == 403
        response = await client.get("/v1/session-traces", headers=bearer(Role.WORKER))
        assert response.status == 403
        response = await client.post("/v1/device-channels/revoke", json={})
        assert response.status == 401


def test_a_route_nobody_declared_stops_the_provider_from_starting() -> None:
    app = web.Application()

    async def handler(_request: web.Request) -> web.Response:
        return web.Response()

    app.router.add_post("/v1/device-channels/everything", handler)
    with pytest.raises(RuntimeError, match="POST /v1/device-channels/everything"):
        protect(app, ROLE_CREDENTIALS)


def test_credentials_come_from_one_variable_per_role() -> None:
    environ = {
        CREDENTIAL_ENV[Role.HUB]: "h" * 32,
        CREDENTIAL_ENV[Role.MANAGEMENT]: "m" * 32,
        CREDENTIAL_ENV[Role.WORKER]: "w" * 32,
    }
    credentials = RoleCredentials.from_environment(environ)
    assert credentials.role_of("Bearer " + "h" * 32) is Role.HUB
    assert credentials.role_of("Bearer " + "m" * 32) is Role.MANAGEMENT
    assert credentials.role_of("Bearer " + "w" * 32) is Role.WORKER
    assert credentials.role_of("bearer " + "w" * 32) is None
    assert credentials.role_of("Bearer ") is None
    assert "h" * 32 not in repr(credentials)


def test_a_missing_or_short_credential_names_its_variable() -> None:
    environ = {CREDENTIAL_ENV[Role.HUB]: "h" * 32, CREDENTIAL_ENV[Role.WORKER]: "w" * 32}
    with pytest.raises(ValueError, match=CREDENTIAL_ENV[Role.MANAGEMENT]):
        RoleCredentials.from_environment(environ)
    with pytest.raises(ValueError, match=CREDENTIAL_ENV[Role.WORKER]):
        RoleCredentials(hub="h" * 32, management="m" * 32, worker="w" * 31)


def test_two_roles_cannot_share_one_credential() -> None:
    with pytest.raises(ValueError, match="distinct"):
        RoleCredentials(hub="s" * 32, management="s" * 32, worker="w" * 32)


def test_the_retired_single_credential_is_not_read() -> None:
    assert "EIDOLON_CHANNEL_PROVIDER_TOKEN" not in CREDENTIAL_ENV.values()


def test_the_worker_presents_its_own_role_and_no_other() -> None:
    """The worker shares ``channel.env`` with the Provider, so it can see all
    three credentials; what it presents is decided in its code, and only here.
    """

    root = Path(__file__).resolve().parents[2] / "livekit"
    sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
    }
    others = (
        "credentials.hub",
        "credentials.management",
        "Role.HUB",
        "Role.MANAGEMENT",
        CREDENTIAL_ENV[Role.HUB],
        CREDENTIAL_ENV[Role.MANAGEMENT],
        "EIDOLON_CHANNEL_PROVIDER_TOKEN",
    )
    assert sorted(name for name, text in sources.items() if any(o in text for o in others)) == []
    # And its three calls present the worker's.
    assert "credentials.worker" in sources["agent/session/conversation_speaker.py"]
    assert "credentials.worker" in sources["agent/session/speech_voice.py"]
    assert "Role.WORKER" in sources["agent/smarthome.py"]
