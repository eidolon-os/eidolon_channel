from __future__ import annotations

import json

from aiohttp.test_utils import TestClient, TestServer

from eidolon.livekit.channel_provider.contracts import canonical_json
from eidolon.livekit.channel_provider.http import create_app
from eidolon.livekit.channel_provider.livekit_backend import LiveKitBinding
from eidolon.livekit.channel_provider.service import ChannelProviderService
from eidolon.livekit.channel_provider.store import ChannelProviderStore

from .helpers import livekit_config, provision_payload, revoke_payload


class HttpFakeBackend:
    async def ensure_rooms(self, active_room: str, control_room: str) -> None:
        pass

    async def revoke_rooms(self, active_room: str, control_room: str) -> None:
        pass

    def build_binding(self, **values) -> LiveKitBinding:
        return LiveKitBinding(
            payload=canonical_json(
                {
                    "schema_version": 1,
                    "active": {"room_name": values["active_room"]},
                    "control": {"room_name": values["control_room"]},
                }
            ).encode(),
            expires_at_ms=values["issued_at_ms"] + 1_800_000,
        )

    async def close(self) -> None:
        pass


async def test_http_surface_auth_contract_health_and_revoke(tmp_path) -> None:
    config = livekit_config()
    service = ChannelProviderService(
        store=ChannelProviderStore(tmp_path / "provider.sqlite3"),
        backend=HttpFakeBackend(),
        livekit=config,
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

        unauthorized = await client.post(
            "/v1/device-channels/provision", json=provision_payload()
        )
        assert unauthorized.status == 401

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
        rejected = await client.post(
            "/v1/device-channels/provision", json=invalid, headers=headers
        )
        assert rejected.status == 422

        revoked = await client.post(
            "/v1/device-channels/revoke", json=revoke_payload(), headers=headers
        )
        assert revoked.status == 200
        assert json.loads(await revoked.text())["operation"] == "channel.revoked-device"
    finally:
        await client.close()
