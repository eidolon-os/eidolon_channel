"""Channel's internal client; Hub is the sole device execution authority."""

import aiohttp
from eidolon_sdk.biz.smarthome import ExecuteRequest, ExecuteResult


class HubSmartHomeClient:
    def __init__(self, base_url: str, token: str):
        self._base_url, self._token = base_url.rstrip("/"), token

    async def _post(self, action: str, body: dict) -> dict:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.post(
                self._base_url + "/api/smarthome/v1/" + action,
                headers={"Authorization": "Bearer " + self._token},
                json=body,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def snapshot(self, owner_id: str) -> dict:
        return await self._post("snapshot", {"owner_id": owner_id})

    async def execute(self, owner_id: str, request: ExecuteRequest) -> ExecuteResult:
        return ExecuteResult.model_validate(
            await self._post(
                "execute", {"owner_id": owner_id, "request": request.model_dump(mode="json")}
            )
        )
