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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Final

import aiohttp
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7
from yarl import URL

from .music_cache import MusicDownloadError, MusicSource

DEFAULT_BASE_URL: Final = "https://api.lizheng.design"
DEFAULT_CHANNEL: Final = "bf117132ace49f34"
HTTP_TIMEOUT_SECONDS: Final = 15.0
SENSITIVE_FIELDS: Final = frozenset(
    {
        "ak",
        "aeskey",
        "appkey",
        "authorization",
        "credential",
        "credentials",
        "key",
        "mac",
        "password",
        "secret",
        "secretkey",
        "sign",
        "signature",
        "sn",
        "token",
    }
)

logger = logging.getLogger("agent.session.music")


class LizhengMusicApiError(RuntimeError):
    def __init__(
        self,
        code: int | str,
        message: str,
        *,
        source: LizhengErrorSource,
        endpoint: LizhengEndpoint,
    ) -> None:
        super().__init__(f"Lizheng music API code={code}: {message}")
        self.code = code
        self.message = message
        self.source = source
        self.endpoint = endpoint


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
            raise MusicDownloadError("no music matched the query")
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
        params = kwargs.get("params")
        if isinstance(params, dict):
            log_url = str(
                URL(request_url).with_query(
                    [(str(key), str(value)) for key, value in params.items()]
                )
            )
        else:
            log_url = request_url
        logger.info(
            "lizheng http request method=%s endpoint=%s url=%s",
            method,
            endpoint.value,
            _redact_url(log_url),
        )
        async with self._http.request(
            method,
            request_url,
            timeout=timeout,
            **kwargs,
        ) as response:
            safe_url = _redact_url(str(response.url))
            try:
                body: object = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                body = {"body": "<non-json response>"}
            logger.info(
                "lizheng http response endpoint=%s url=%s http_status=%d body=%s",
                endpoint.value,
                safe_url,
                response.status,
                json.dumps(
                    _redact_json(body, redact_data=endpoint is LizhengEndpoint.AUTHENTICATION),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
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


def _redact_url(value: str, *, redact_all_query_values: bool = False) -> str:
    parts = urlsplit(value)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    if parts.username is not None or parts.password is not None:
        hostname = "<redacted>@" + hostname
    query = [
        (
            key,
            "<redacted>" if redact_all_query_values or _field_is_sensitive(key) else item,
        )
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, hostname, parts.path, urlencode(query), ""))


def _redact_json(value: object, *, redact_data: bool = False) -> object:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>"
            if _field_is_sensitive(str(key)) or (redact_data and str(key) == "data")
            else _redact_json(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return _redact_url(value, redact_all_query_values=True)
    return value


def _field_is_sensitive(name: str) -> bool:
    normalized = "".join(character for character in name.lower() if character.isalnum())
    return normalized in SENSITIVE_FIELDS
