"""Channel's internal client; Hub is the sole device execution authority."""

import asyncio

import httpx
from eidolon_sdk.core.http import create_async_client
from eidolon_sdk.biz.smarthome import ExecuteRequest, ExecuteResult


class HubSmartHomeClient:
    def __init__(self, base_url: str, token: str, *, client: httpx.AsyncClient | None = None):
        self._base_url, self._token = base_url.rstrip("/"), token
        self._owns_client = client is None
        self._client = (
            client if client is not None else create_async_client(timeout=5, trust_env=False)
        )

    async def aclose(self):
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, action: str, body: dict) -> dict:
        async with asyncio.timeout(5):
            response = await self._client.post(
                self._base_url + "/api/smarthome/v1/" + action,
                headers={"Authorization": "Bearer " + self._token},
                json=body,
            )
            response.raise_for_status()
            return response.json()

    async def snapshot(self, owner_id: str) -> dict:
        return await self._post("snapshot", {"owner_id": owner_id})

    async def changes(self, owner_id: str, *, since: int, timeout_ms: int) -> dict:
        # A long poll waits on purpose; give the transport the wait plus slack.
        async with asyncio.timeout(timeout_ms / 1000 + 10):
            response = await self._client.post(
                self._base_url + "/api/smarthome/v1/changes",
                headers={"Authorization": "Bearer " + self._token},
                json={"owner_id": owner_id, "since": since, "timeout_ms": timeout_ms},
                timeout=timeout_ms / 1000 + 10,
            )
            response.raise_for_status()
            return response.json()

    async def execute(self, owner_id: str, request: ExecuteRequest) -> ExecuteResult:
        return ExecuteResult.model_validate(
            await self._post(
                "execute", {"owner_id": owner_id, "request": request.model_dump(mode="json")}
            )
        )
