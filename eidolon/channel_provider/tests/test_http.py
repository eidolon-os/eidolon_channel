from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.contracts import (
    CLOSE_SESSION,
    OPEN_SESSION,
    BackendUnavailable,
    Forbidden,
    IdempotencyConflict,
    InvalidTransition,
    StaleGeneration,
    Unauthenticated,
)
from eidolon.channel_provider.http import _problem, create_app
from eidolon.channel_provider.selection import AdapterRegistry
from eidolon.channel_provider.service import ChannelProviderService
from eidolon.channel_provider.store import ChannelProviderStore

from .helpers import (
    FakeAdapter,
    provision_payload,
    current_payload,
    revoke_payload,
    session_payload,
)


@pytest.mark.parametrize(
    ("error", "status", "code", "category", "retryable"),
    [
        (StaleGeneration("old generation"), 409, "STALE_GENERATION", "conflict", False),
        (
            IdempotencyConflict("payload changed"),
            409,
            "IDEMPOTENCY_CONFLICT",
            "conflict",
            False,
        ),
        (
            InvalidTransition("state forbids this"),
            409,
            "INVALID_TRANSITION",
            "conflict",
            False,
        ),
        (Unauthenticated("bad credential"), 401, "UNAUTHENTICATED", "auth", False),
        (Forbidden("scope denied"), 403, "FORBIDDEN", "forbidden", False),
        (
            BackendUnavailable("provider down"),
            503,
            "PROVIDER_UNAVAILABLE",
            "unavailable",
            True,
        ),
    ],
)
def test_domain_problems_keep_distinct_transport_mapping(
    error, status, code, category, retryable
) -> None:
    response = _problem(error)
    body = json.loads(response.text)

    assert response.status == body["status"] == status
    assert (body["code"], body["category"], body["retryable"]) == (
        code,
        category,
        retryable,
    )


async def test_http_surface_auth_contract_health_and_revoke(tmp_path) -> None:
    adapter = FakeAdapter(name="livekit")
    service = ChannelProviderService(
        store=ChannelProviderStore(tmp_path / "provider.sqlite3"),
        registry=AdapterRegistry([adapter], preference=("livekit",)),
        agent_name="eidolon",
        now_ms=lambda: 1_700_000_000_000,
    )
    service.initialize()
    client = TestClient(TestServer(create_app(service=service, bearer_token="s" * 32)))
    try:
        await client.start_server()
        health = await client.get("/health")
        assert health.status == 200
        assert await health.json() == {
            "status": "ok",
            "service": "eidolon-channel-provider",
            "contract_version": "v1",
        }
        assert adapter.health_calls == 1

        unauthorized = await client.post("/v1/device-channels/provision", json=provision_payload())
        assert unauthorized.status == 401
        assert unauthorized.content_type == "application/problem+json"
        assert (await unauthorized.json())["code"] == "UNAUTHENTICATED"

        headers = {"Authorization": f"Bearer {'s' * 32}"}
        provisioned = await client.post(
            "/v1/device-channels/provision",
            json=provision_payload(),
            headers=headers,
        )
        assert provisioned.status == 200
        assert (await provisioned.json())["operation"] == "channel.provisioned-device"

        invalid = provision_payload()
        invalid["extra"] = True
        rejected = await client.post("/v1/device-channels/provision", json=invalid, headers=headers)
        assert rejected.status == 422

        opened = await client.post(
            "/v1/device-channels/sessions/open",
            json=session_payload(operation=OPEN_SESSION),
            headers=headers,
        )
        assert opened.status == 200
        assert json.loads(await opened.text())["operation"] == "channel.opened-session"
        assert len(adapter.sessions_opened) == 1

        # Each route accepts only its own operation, so a mis-posted request
        # cannot end a conversation the caller meant to start.
        crossed = await client.post(
            "/v1/device-channels/sessions/close",
            json=session_payload(operation=OPEN_SESSION),
            headers=headers,
        )
        assert crossed.status == 422

        closed = await client.post(
            "/v1/device-channels/sessions/close",
            json=session_payload(operation=CLOSE_SESSION),
            headers=headers,
        )
        assert closed.status == 200
        assert len(adapter.sessions_closed) == 1

        unknown = await client.post(
            "/v1/device-channels/sessions/open",
            json=session_payload(operation=OPEN_SESSION, device_id="nobody"),
            headers=headers,
        )
        assert unknown.status == 404

        unauthenticated_read = await client.post(
            "/v1/device-channels/current", json=current_payload()
        )
        assert unauthenticated_read.status == 401

        read = await client.post(
            "/v1/device-channels/current", json=current_payload(), headers=headers
        )
        assert read.status == 200
        assert json.loads(await read.text())["binding"] is not None

        crossed_read = await client.post(
            "/v1/device-channels/current",
            json=current_payload(operation="channel.revoke-device"),
            headers=headers,
        )
        assert crossed_read.status == 422

        revoked = await client.post(
            "/v1/device-channels/revoke", json=revoke_payload(), headers=headers
        )
        assert revoked.status == 200
        assert json.loads(await revoked.text())["operation"] == "channel.revoked-device"
    finally:
        await client.close()


async def test_http_preserves_domain_error_codes_from_the_adapter(tmp_path) -> None:
    async def response_for(error):
        adapter = FakeAdapter(name="livekit")

        async def fail_open(_spec, *, issued_at_ms):
            raise error

        adapter.open = fail_open
        service = ChannelProviderService(
            store=ChannelProviderStore(tmp_path / f"{error.code}.sqlite3"),
            registry=AdapterRegistry([adapter], preference=("livekit",)),
            agent_name="eidolon",
            now_ms=lambda: 1_700_000_000_000,
        )
        service.initialize()
        client = TestClient(TestServer(create_app(service=service, bearer_token="s" * 32)))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/device-channels/provision",
                json=provision_payload(),
                headers={"Authorization": f"Bearer {'s' * 32}"},
            )
            return response.status, await response.json()
        finally:
            await client.close()

    forbidden_status, forbidden = await response_for(Forbidden("policy denied this principal"))
    assert forbidden_status == 403
    assert forbidden == {
        "type": "https://problems.eidolon.live/channel/forbidden",
        "title": "Forbidden",
        "status": 403,
        "detail": "policy denied this principal",
        "code": "FORBIDDEN",
        "category": "forbidden",
        "retryable": False,
        "authority": "eidolon-channel-provider",
    }

    unavailable_status, problem = await response_for(BackendUnavailable("transport unavailable"))
    assert unavailable_status == 503
    assert (problem["code"], problem["retryable"]) == (
        "PROVIDER_UNAVAILABLE",
        True,
    )
