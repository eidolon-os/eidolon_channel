"""Read the Owner's smart-home registry from System Data's existing authority."""

from __future__ import annotations

from urllib.parse import quote

import aiohttp
from eidolon_sdk.biz.smarthome import Registry


class HttpRegistrySource:
    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token

    async def get(self, owner_id: str) -> Registry:
        path = f"/api/workspace-authority/v1/owners/{quote(owner_id, safe='')}/smarthome/registry"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                self._base_url + path,
                headers={"Authorization": f"Bearer {self._token}"},
            ) as response:
                response.raise_for_status()
                return Registry.model_validate(await response.json())
