from __future__ import annotations

from types import SimpleNamespace

import pytest
from eidolon_sdk.biz.registry.models import (
    AgentMetadataRecord,
    DeviceBindingRecord,
    UserRegistryRecord,
)

from eidolon.livekit.agent.factory import _build_runtime_resolve_client
from eidolon_data import DataSettings, DataStore
from eidolon_data.adapters import (
    EidolonDataAgentMetadataRepository,
    EidolonDataDeviceBindingRepository,
    EidolonDataUserRepository,
)

pytestmark = pytest.mark.asyncio


async def test_runtime_resolve_client_prefers_eidolon_data(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "eidolon.sqlite3"
    monkeypatch.setenv("EIDOLON_DATA_SQLITE_PATH", str(db_path))
    store = DataStore.open(DataSettings(sqlite_path=str(db_path)))
    try:
        await EidolonDataUserRepository(store).put(
            UserRegistryRecord(
                user_id="alice",
                tenant_id="default",
                active_agent_id="companion-a",
                display_name="Alice",
                memory_port=18101,
                created_at="2026-06-27T00:00:00+00:00",
            )
        )
        await EidolonDataAgentMetadataRepository(store).put(
            AgentMetadataRecord(
                agent_id="companion-a",
                tenant_id="default",
                user_id="alice",
                template_id="xiaoyi",
                template_revision=1,
                display_name="Xiaoyi",
                created_at="2026-06-27T00:01:00+00:00",
            )
        )
        await EidolonDataDeviceBindingRepository(store).put(
            DeviceBindingRecord(
                device_id="esp32-a",
                agent_id="companion-a",
                bound_at="2026-06-27T00:02:00+00:00",
            )
        )

        client = _build_runtime_resolve_client(
            SimpleNamespace(
                data_resolve_enabled=True,
                admin_fallback_enabled=False,
                admin_api_url="http://admin.invalid",
            )
        )
        user_ctx = await client.resolve_user("alice")
        assert user_ctx.user_id == "alice"
        assert user_ctx.agent_id == "companion-a"
        assert user_ctx.memory_mcp_url == "http://127.0.0.1:18101/mcp"

        device_ctx = await client.resolve_device("esp32-a")
        assert device_ctx.user_id == "alice"
        assert device_ctx.device_id == "esp32-a"
    finally:
        await store.close()
