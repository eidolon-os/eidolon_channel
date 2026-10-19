from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import aiohttp
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from ...common.music import MusicProviderError, MusicSearchCandidate, MusicSource

DEFAULT_BASE_URL: Final = "https://api.lizheng.design"
DEFAULT_CHANNEL: Final = "bf117132ace49f34"
HTTP_TIMEOUT_SECONDS: Final = 15.0
logger = logging.getLogger("agent.session.music")


class LizhengMusicApiError(MusicProviderError):
    """A rejected or malformed Lizheng exchange."""


class LizhengErrorSource(StrEnum):
    HTTP_STATUS = "http_status"
    API_RESPONSE = "api_response"
    LOCAL_RESPONSE = "local_response"


class LizhengEndpoint(StrEnum):
    AUTHENTICATION = "authentication"
    SEARCH = "search"
    RECOMMENDATION = "recommendation"
    PLAYBACK_URL = "playback_url"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


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
                source=LizhengErrorSource.LOCAL_RESPONSE,
                endpoint=LizhengEndpoint.CONFIGURATION,
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
            return (
                await self._request(
                    "GET",
                    "/thrid/mlmusic/app/songs",
                    endpoint=LizhengEndpoint.SEARCH,
                    params=params,
                )
                or []
            )

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
                endpoint=LizhengEndpoint.RECOMMENDATION,
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

    async def search_candidates(
        self,
        keyword: str,
        *,
        version: str = "",
        language: str = "",
    ) -> list[MusicSearchCandidate]:
        songs = await self.search(keyword, version=version, language=language)
        return _candidate_list(songs)

    async def recommend_candidates(self) -> list[MusicSearchCandidate]:
        return _candidate_list(await self.recommend())

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
                endpoint=LizhengEndpoint.PLAYBACK_URL,
                params=params,
            )
            media = data.get("mediaList", []) if isinstance(data, dict) else []
            if not media or not isinstance(media[0], dict) or not media[0].get("url"):
                raise LizhengMusicApiError(
                    -1,
                    "playback URL is empty",
                    source=LizhengErrorSource.LOCAL_RESPONSE,
                    endpoint=LizhengEndpoint.PLAYBACK_URL,
                )
            media_song_id = media[0].get("songId")
            if media_song_id is not None and str(media_song_id) != song_id:
                raise LizhengMusicApiError(
                    -1,
                    "playback URL song identity does not match request",
                    source=LizhengErrorSource.LOCAL_RESPONSE,
                    endpoint=LizhengEndpoint.PLAYBACK_URL,
                )
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
        songs = await (
            self.recommend()
            if query == "random"
            else self.search(query, version=version, language=language)
        )
        if not songs:
            raise LizhengMusicApiError(
                "not_found",
                "no music matched the query",
                source=LizhengErrorSource.LOCAL_RESPONSE,
                endpoint=LizhengEndpoint.SEARCH,
            )
        song = random.choice(songs)
        return await self.source_for_song(str(song["songId"]))

    async def _credentials_for(self, *, force_auth: bool) -> tuple[str, str]:
        if self._credentials is not None and not force_auth:
            return self._credentials
        payload = await self._request(
            "POST",
            f"/thrid/mlmusic/app/{self._config.channel}/queryDevcie",
            endpoint=LizhengEndpoint.AUTHENTICATION,
            json={"mac": self._config.mac, "type": 5},
        )
        if not isinstance(payload, str):
            raise LizhengMusicApiError(
                -1,
                "authentication response is not encrypted text",
                source=LizhengErrorSource.LOCAL_RESPONSE,
                endpoint=LizhengEndpoint.AUTHENTICATION,
            )
        try:
            plain = _decrypt(payload, self._config.aes_key)
            node = plain.get("qq_api_music", {})
            sn, ak = node.get("sn"), node.get("ak")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LizhengMusicApiError(
                -1,
                "invalid authentication response",
                source=LizhengErrorSource.LOCAL_RESPONSE,
                endpoint=LizhengEndpoint.AUTHENTICATION,
            ) from exc
        if not sn or not ak:
            raise LizhengMusicApiError(
                -1,
                "authentication response lacks sn or ak",
                source=LizhengErrorSource.LOCAL_RESPONSE,
                endpoint=LizhengEndpoint.AUTHENTICATION,
            )
        self._credentials = (str(sn), str(ak))
        return self._credentials

    async def _request(
        self,
        method: str,
        path: str,
        *,
        endpoint: LizhengEndpoint,
        **kwargs: object,
    ) -> object:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
        request_url = f"{self._config.base_url}{path}"
        async with self._http.request(
            method,
            request_url,
            timeout=timeout,
            **kwargs,
        ) as response:
            try:
                body: object = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                body = {"body": "<non-json response>"}
            logger.debug(
                "lizheng request completed endpoint=%s http_status=%d",
                endpoint.value,
                response.status,
            )
            if response.status < 200 or response.status >= 300:
                if response.status == 410:
                    raise LizhengMusicApiError(
                        410,
                        "credentials expired",
                        source=LizhengErrorSource.HTTP_STATUS,
                        endpoint=endpoint,
                    )
                raise LizhengMusicApiError(
                    response.status,
                    "HTTP request failed",
                    source=LizhengErrorSource.HTTP_STATUS,
                    endpoint=endpoint,
                )
        if not isinstance(body, dict):
            raise LizhengMusicApiError(
                -1,
                "API response is not an object",
                source=LizhengErrorSource.API_RESPONSE,
                endpoint=endpoint,
            )
        if not body.get("success"):
            raise LizhengMusicApiError(
                body.get("code", -1),
                body.get("message", "unknown"),
                source=LizhengErrorSource.API_RESPONSE,
                endpoint=endpoint,
            )
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


def _candidate_list(songs: list[dict[str, object]]) -> list[MusicSearchCandidate]:
    candidates: list[MusicSearchCandidate] = []
    for song in songs:
        song_id = song.get("songId")
        if not isinstance(song_id, str) or not song_id.strip():
            continue
        title = song.get("songName")
        artist = song.get("artist")
        candidates.append(
            MusicSearchCandidate(
                song_id=song_id.strip(),
                title=title.strip() if isinstance(title, str) else "",
                artist=artist.strip() if isinstance(artist, str) else "",
            )
        )
    return candidates
