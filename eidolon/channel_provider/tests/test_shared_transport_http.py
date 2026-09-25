import asyncio
from contextlib import asynccontextmanager

import pytest
from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from eidolon.channel_provider.http import create_app
from eidolon.channel_provider.contracts import BackendUnavailable
from .helpers import provision_payload
from .test_shared_transport_scope import setup

TOKEN = "test-provider-token-at-least-32-bytes"


@asynccontextmanager
async def client_scope(tmp_path):
    service, _, adapter, _, selection, requests = await setup(tmp_path)
    payloads = []
    for request in requests:
        raw = provision_payload(device_id=request.device_id)
        raw["operation_id"] = request.operation_id
        raw["device_ref"] = request.device_ref.model_dump(mode="json")
        payloads.append({k: raw[k] for k in ("device_ref", "device")})
    body = {
        "owner_id": "owner_1",
        "selection": selection.model_dump(mode="json"),
        "specifications": payloads,
    }
    async with TestClient(TestServer(create_app(service=service, bearer_token=TOKEN))) as client:
        yield client, service, adapter, body


async def connect(client):
    return await client.ws_connect(
        "/v1/shared-transports", headers={"Authorization": f"Bearer {TOKEN}"}
    )


async def test_authentication_precedes_upgrade_and_resource_creation(tmp_path):
    async with client_scope(tmp_path) as (client, _, adapter, _):
        with pytest.raises(WSServerHandshakeError) as error:
            await client.ws_connect("/v1/shared-transports")
        assert error.value.status == 401
        assert not adapter.shared_created


async def test_real_service_admits_then_close_cleans_before_reply(tmp_path):
    async with client_scope(tmp_path) as (client, service, adapter, body):
        async with await connect(client) as ws:
            await ws.send_json(body)
            ready = await ws.receive_json(timeout=1)
            assert ready["state"] == "transport_ready"
            assert "token" not in str(ready)
            await ws.send_json({"action": "close"})
            assert (await ws.receive_json(timeout=1))["state"] == "closed"
            assert not service._shared_scopes
            assert adapter.closed[0]["resource"] == "temporary"
            assert not adapter.sessions_opened


@pytest.mark.parametrize("invalid", ["owner", "extra", "malformed"])
async def test_bad_start_cannot_invite(tmp_path, invalid):
    async with client_scope(tmp_path) as (client, _, adapter, body):
        if invalid == "owner":
            body["owner_id"] = "other-owner"
        elif invalid == "extra":
            body["grant"] = "untrusted"
        async with await connect(client) as ws:
            if invalid == "malformed":
                await ws.send_str("{")
            else:
                await ws.send_json(body)
            error = await ws.receive_json(timeout=1)
            assert error["state"] == "error"
            assert error["code"] == ("FORBIDDEN" if invalid == "owner" else "INVALID_CONTRACT")
        assert not adapter.shared_created


async def test_disconnect_during_invitation_cancels_and_cleans(tmp_path):
    async with client_scope(tmp_path) as (client, service, adapter, body):
        sending = asyncio.Event()
        cleaned = asyncio.Event()
        original = adapter.close

        async def deliver(*args, **kw):
            sending.set()
            await asyncio.Event().wait()

        async def close(handle):
            await original(handle)
            cleaned.set()

        adapter.deliver_shared_invitation = deliver
        adapter.close = close
        ws = await connect(client)
        await ws.send_json(body)
        await asyncio.wait_for(sending.wait(), 1)
        await ws.close()
        await asyncio.wait_for(cleaned.wait(), 1)
        assert not service._shared_scopes


async def test_failed_cleanup_does_not_report_closed(tmp_path):
    async with client_scope(tmp_path) as (client, service, adapter, body):
        original = adapter.close

        async def fail(handle):
            raise BackendUnavailable("failed cleanup")

        async with await connect(client) as ws:
            await ws.send_json(body)
            assert (await ws.receive_json(timeout=1))["state"] == "transport_ready"
            adapter.close = fail
            await ws.send_json({"action": "close"})
            assert (await ws.receive_json(timeout=1))["code"] == "PROVIDER_UNAVAILABLE"
            assert service._shared_scopes
        adapter.close = original


@pytest.mark.parametrize("mutation", ["operation", "manifest", "duplicate", "bad_ref"])
async def test_specification_cannot_override_ledger_or_selected_device(tmp_path, mutation):
    async with client_scope(tmp_path) as (client, _, adapter, body):
        specs = body["specifications"]
        if mutation == "operation":
            specs[0]["operation_id"] = "forged"
        elif mutation == "manifest":
            specs[0]["device"]["display_name"] = "changed-without-current-provision"
        elif mutation == "duplicate":
            specs[1] = specs[0]
        else:
            specs[0]["device_ref"]["device_instance_id"] = []
        async with await connect(client) as ws:
            await ws.send_json(body)
            assert (await ws.receive_json(timeout=1))["state"] == "error"
        assert not adapter.shared_created
