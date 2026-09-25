import pytest
from aiohttp.test_utils import TestClient, TestServer
from eidolon.channel_provider.http import create_app
from .test_shared_session_commands import fixture

TOKEN = "provider-test-token-at-least-32-bytes"


async def test_http_start_returns_without_closing_then_explicit_end(tmp_path):
    service, adapter, selection, specs = await fixture(tmp_path)
    async with TestClient(TestServer(create_app(service=service, bearer_token=TOKEN))) as client:
        body = {
            "owner_id": "owner_1",
            "selection": selection.model_dump(mode="json"),
            "specifications": list(specs),
        }
        assert (await client.post("/v1/shared-sessions/open", json=body)).status == 401
        assert not adapter.shared_created
        headers = {"Authorization": f"Bearer {TOKEN}"}
        response = await client.post("/v1/shared-sessions/open", json=body, headers=headers)
        assert response.status == 200, await response.text()
        assert (await response.json())["state"] == "transport_ready"
        assert not adapter.closed
        response = await client.post(
            "/v1/shared-sessions/close",
            json={"owner_id": "owner_1", "session_id": selection.session_id},
            headers=headers,
        )
        assert response.status == 200
        assert len(adapter.closed) == 1
        assert (await client.get("/v1/shared-transports", headers=headers)).status == 404


@pytest.mark.parametrize("case", ["owner", "extra", "invalid_spec"])
async def test_rejected_command_allocates_nothing(tmp_path, case):
    service, adapter, selection, specs = await fixture(tmp_path)
    async with TestClient(TestServer(create_app(service=service, bearer_token=TOKEN))) as client:
        body = {
            "owner_id": "owner_1",
            "selection": selection.model_dump(mode="json"),
            "specifications": list(specs),
        }
        if case == "owner":
            body["owner_id"] = "other"
        if case == "extra":
            body["heartbeat"] = True
        if case == "invalid_spec":
            body["specifications"][0]["operation_id"] = "injected"
        response = await client.post(
            "/v1/shared-sessions/open", json=body, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert response.status == (403 if case == "owner" else 422)
        assert not adapter.shared_created
