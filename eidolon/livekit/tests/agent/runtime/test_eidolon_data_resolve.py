from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from eidolon_sdk.biz.system_data import SystemDataRuntimeClient

from eidolon.livekit.agent.runtime.system_data import (
    SystemDataRuntimeResolver,
    build_system_data_runtime,
)

pytestmark = pytest.mark.asyncio


async def test_channel_runtime_has_no_system_data_package_dependency() -> None:
    root = Path(__file__).resolve().parents[5]
    violations = []
    for path in (root / "eidolon").rglob("*.py"):
        if "tests" in path.parts:
            continue
        if "eidolon_data" in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(root)))

    assert violations == []
    assert '"eidolon-data"' not in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "eidolon-data =" not in (root / "pyproject.toml").read_text(encoding="utf-8")


async def test_runtime_resolve_client_consumes_system_data_authority() -> None:
    token = "channel-system-data-contract-token-001"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {token}"
        return httpx.Response(
            200,
            json={
                "contract_version": "1",
                "operation": "companion.runtime-snapshot",
                "owner_id": "owner-a",
                "companion_id": "companion-a",
                "lifecycle_state": "active",
                "runtime_config": {},
                "memory_realm": {"realm_id": "realm-a", "lifecycle_state": "active"},
                "persona_genome": {
                    "genome_id": "genome-a",
                    "version": 1,
                    "lifecycle_state": "committed",
                    "schema_version": "eidolon.persona_genome",
                    "genome_hash": "pg_hash",
                    "realizer_version": "eidolon.persona_realizer",
                    "genome": {},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = SystemDataRuntimeResolver(
            SystemDataRuntimeClient(http, "http://data.test", service_token=token)
        )

        companion_ctx = await client.resolve_companion("companion-a", device_id=None)
        assert companion_ctx.owner_id == "owner-a"
        assert companion_ctx.companion_id == "companion-a"
        assert companion_ctx.memory_realm_id == "realm-a"
        assert companion_ctx.genome_id == "genome-a"
        assert companion_ctx.device_id is None

        device_ctx = await client.resolve_companion("companion-a", device_id="esp32-a")
        assert device_ctx.owner_id == "owner-a"
        assert device_ctx.device_id == "esp32-a"

        owner_ctx = await client.resolve_owner("owner-a")
        assert owner_ctx.owner_id == "owner-a"
        assert owner_ctx.companion_id == "companion-a"
        assert owner_ctx.memory_realm_id == "realm-a"
        assert owner_ctx.genome_id == "genome-a"
        assert owner_ctx.device_id is None


async def test_deferred_runtime_does_not_require_credentials_until_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_env = "EIDOLON_TEST_DATA_AUTHORITY_TOKEN"
    monkeypatch.delenv(token_env, raising=False)
    settings = type(
        "Settings",
        (),
        {
            "data_service_token_env": token_env,
            "data_api_url": "http://data.test",
            "http_timeout_sec": 1.0,
            "http_connect_timeout_sec": 0.5,
        },
    )()

    client = build_system_data_runtime(settings)

    with pytest.raises(RuntimeError, match=token_env):
        await client.resolve_owner("owner-a")
    await client.aclose()
