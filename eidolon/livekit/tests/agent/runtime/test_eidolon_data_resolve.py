from __future__ import annotations

from types import SimpleNamespace

import pytest

from eidolon.livekit.agent.factory import _build_runtime_resolve_client
from eidolon_data import DataSettings, DataStore

pytestmark = pytest.mark.asyncio


async def test_runtime_resolve_client_prefers_eidolon_data(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "eidolon.sqlite3"
    monkeypatch.setenv("EIDOLON_DATA_SQLITE_PATH", str(db_path))
    store = DataStore.open(DataSettings(sqlite_path=str(db_path)))
    try:
        await store.init_schema()
        await store.owner_service.create_owner(owner_id="owner-a", display_name="Owner A")
        workspace = await store.workspace_provisioning.provision_workspace(
            owner_id="owner-a",
            companion_id="companion-a",
            genome_id="genome-a",
            realm_id="realm-a",
        )
        await store.devices.create_device(
            device_id="esp32-a",
            owner_id="owner-a",
            status="approved",
            bound_companion_id=workspace.companion.companion_id,
        )

        client = _build_runtime_resolve_client(
            SimpleNamespace(
                data_resolve_enabled=True,
                admin_fallback_enabled=False,
                admin_api_url="http://admin.invalid",
            )
        )
        device_ctx = await client.resolve_device("esp32-a")
        assert device_ctx.owner_id == "owner-a"
        assert device_ctx.companion_id == "companion-a"
        assert device_ctx.memory_realm_id == "realm-a"
        assert device_ctx.genome_id == "genome-a"
        assert device_ctx.device_id == "esp32-a"

        owner_ctx = await client.resolve_owner("owner-a")
        assert owner_ctx.owner_id == "owner-a"
        assert owner_ctx.companion_id == "companion-a"
        assert owner_ctx.memory_realm_id == "realm-a"
        assert owner_ctx.genome_id == "genome-a"
        assert owner_ctx.device_id is None
    finally:
        await store.close()
