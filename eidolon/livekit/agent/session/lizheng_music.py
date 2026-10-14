from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Final

import aiohttp
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from .music_cache import MusicDownloadError, MusicSource

DEFAULT_BASE_URL: Final = "https://api.lizheng.design"
DEFAULT_CHANNEL: Final = "bf117132ace49f34"
HTTP_TIMEOUT_SECONDS: Final = 15.0


class LizhengMusicApiError(RuntimeError):
    def __init__(self, code: int | str, message: str) -> None:
        super().__init__(f"Lizheng music API code={code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class LizhengMusicConfig:
    base_url: str
    channel: str
    mac: str
    aes_key: str

    @classmethod
    def from_environment(cls) -> LizhengMusicConfig:
        base_url = os.getenv("LIZHENG_MUSIC_API_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
        channel = os.getenv("LIZHENG_MUSIC_API_CHANNEL", DEFAULT_CHANNEL).strip()
        mac = os.getenv("LIZHENG_MUSIC_API_MAC", "").strip()
        aes_key = os.getenv("LIZHENG_MUSIC_API_AES_KEY", "").strip()
        if not mac or len(aes_key.encode("utf-8")) != 16:
            raise LizhengMusicApiError(
                "configuration",
                "LIZHENG_MUSIC_API_MAC and a 16-byte LIZHENG_MUSIC_API_AES_KEY are required",
            )
        return cls(base_url, channel, mac, aes_key)


class LizhengMusicApi:
    def __init__(
        self,
        config: LizhengMusicConfig,
        http_session: aiohttp.ClientSession,
    ) -> None:
        self._config = config
        self._http = http_session
        self._credentials: tuple[str, str] | None = None

    async def search(
        self,
        keyword: str,
        *,
        version: str = "",
        language: str = "",
        page_size: int = 10,
    ) -> list[dict[str, object]]:
        async def request(force_auth: bool) -> list[dict[str, object]]:
            sn, ak = await self._credentials_for(force_auth=force_auth)
            params: dict[str, str | int] = {
                "sn": sn,
                "ak": ak,
                "keyword": keyword,
                "pageSize": page_size,
            }
            if version:
                params["version"] = version
            if language:
                params["language"] = language
            return await self._request("GET", "/thrid/mlmusic/app/songs", params=params) or []

        try:
            songs = await request(force_auth=False)
        except LizhengMusicApiError as exc:
            if exc.code != 410:
                raise
            songs = await request(force_auth=True)
        return [song for song in songs if isinstance(song, dict) and song.get("songId")]

    async def recommend(self, *, page_size: int = 20) -> list[dict[str, object]]:
        async def request(force_auth: bool) -> list[dict[str, object]]:
            sn, ak = await self._credentials_for(force_auth=force_auth)
            data = await self._request(
                "GET",
                "/thrid/mlmusic/app/recommend",
                params={"sn": sn, "ak": ak, "pageSize": page_size},
            )
            songs = data.get("songList", []) if isinstance(data, dict) else []
            return songs if isinstance(songs, list) else []

        try:
            songs = await request(force_auth=False)
        except LizhengMusicApiError as exc:
            if exc.code != 410:
                raise
            songs = await request(force_auth=True)
        return [song for song in songs if isinstance(song, dict) and song.get("songId")]

    async def source_for_song(self, song_id: str) -> MusicSource:
        async def request(force_auth: bool) -> MusicSource:
            sn, ak = await self._credentials_for(force_auth=force_auth)
            timestamp = str(int(time.time() * 1000))
            params = {
                "sn": sn,
                "ak": ak,
                "channel": self._config.channel,
                "timestamp": timestamp,
                "sign": hashlib.md5(
                    f"{self._config.channel}{sn}{timestamp}".encode("utf-8")
                ).hexdigest(),
            }
            data = await self._request(
                "GET",
                f"/thrid/mlmusic/app/play-url/{song_id}",
                params=params,
            )
            media = data.get("mediaList", []) if isinstance(data, dict) else []
            if not media or not isinstance(media[0], dict) or not media[0].get("url"):
                raise LizhengMusicApiError(-1, "playback URL is empty")
            return MusicSource(song_id=song_id, url=str(media[0]["url"]))

        try:
            return await request(force_auth=False)
        except LizhengMusicApiError as exc:
            if exc.code != 410:
                raise
            return await request(force_auth=True)

    async def source_for_query(
        self,
        query: str,
        *,
        version: str = "",
        language: str = "",
    ) -> MusicSource:
        songs = await (self.recommend() if query == "random" else self.search(
            query, version=version, language=language
        ))
        if not songs:
            raise MusicDownloadError("no music matched the query")
        song = random.choice(songs)
        return await self.source_for_song(str(song["songId"]))

    async def _credentials_for(self, *, force_auth: bool) -> tuple[str, str]:
        if self._credentials is not None and not force_auth:
            return self._credentials
        payload = await self._request(
            "POST",
            f"/thrid/mlmusic/app/{self._config.channel}/queryDevcie",
            json={"mac": self._config.mac, "type": 5},
        )
        if not isinstance(payload, str):
            raise LizhengMusicApiError(-1, "authentication response is not encrypted text")
        try:
            plain = _decrypt(payload, self._config.aes_key)
            node = plain.get("qq_api_music", {})
            sn, ak = node.get("sn"), node.get("ak")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LizhengMusicApiError(-1, "invalid authentication response") from exc
        if not sn or not ak:
            raise LizhengMusicApiError(-1, "authentication response lacks sn or ak")
        self._credentials = (str(sn), str(ak))
        return self._credentials

    async def _request(self, method: str, path: str, **kwargs: object) -> object:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        async with self._http.request(
            method,
            f"{self._config.base_url}{path}",
            timeout=timeout,
            **kwargs,
        ) as response:
            if response.status < 200 or response.status >= 300:
                if response.status == 410:
                    raise LizhengMusicApiError(410, "credentials expired")
                raise LizhengMusicApiError(response.status, "HTTP request failed")
            body = await response.json()
        if not isinstance(body, dict):
            raise LizhengMusicApiError(-1, "API response is not an object")
        if not body.get("success"):
            raise LizhengMusicApiError(body.get("code", -1), body.get("message", "unknown"))
        return body.get("data")


def _decrypt(encoded: str, aes_key: str) -> dict[str, object]:
    cipher = Cipher(algorithms.AES(aes_key.encode("utf-8")), modes.ECB())
    decryptor = cipher.decryptor()
    padded = decryptor.update(base64.b64decode(encoded)) + decryptor.finalize()
    unpadder = PKCS7(algorithms.AES.block_size).unpadder()
    payload = unpadder.update(padded) + unpadder.finalize()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("decrypted response is not an object")
    return value
