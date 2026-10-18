import base64
import json
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from eidolon.livekit.agent.session.lizheng_music import (
    LizhengEndpoint,
    LizhengErrorSource,
    LizhengMusicApi,
    LizhengMusicApiError,
    LizhengMusicConfig,
)


def _encrypted_payload(value: dict[str, object], key: str) -> str:
    padder = PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(json.dumps(value).encode()) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key.encode()), modes.ECB()).encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(encrypted).decode()


@pytest.fixture
async def lizheng_server() -> AsyncIterator[tuple[str, list[dict[str, object]]]]:
    key = "0123456789abcdef"
    requests: list[dict[str, object]] = []
    auth_calls = 0
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        nonlocal auth_calls
        requests.append({"path": request.path, "query": dict(request.query)})
        if request.path.endswith("/queryDevcie"):
            auth_calls += 1
            payload = _encrypted_payload({"qq_api_music": {"sn": "sn-1", "ak": "ak-1"}}, key)
            return web.json_response({"success": True, "data": payload})
        if request.path.endswith("/songs"):
            if request.query.get("version") or request.query.get("language"):
                return web.json_response({"success": True, "data": []})
            if auth_calls == 1:
                return web.json_response({"success": False, "code": 410, "message": "expired"})
            return web.json_response(
                {"success": True, "data": [{"songId": "song-1", "sn": "body-sn", "ak": "body-ak"}]}
            )
        if "/play-url/" in request.path:
            return web.json_response(
                {
                    "success": True,
                    "data": {
                        "mediaList": [
                            {
                                "url": "https://cdn.example/song.mp3?sign=media-sign&token=media-token&transDeliveryCode=delivery-secret",
                                "sn": "body-sn",
                            }
                        ]
                    },
                }
            )
        raise AssertionError(request.path)

    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_search_reauthenticates_after_expired_credentials(
    lizheng_server: tuple[str, list[dict[str, object]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_url, requests = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        with caplog.at_level("INFO", logger="agent.session.music"):
            songs = await LizhengMusicApi(config, http).search("hello")

    assert songs == [{"songId": "song-1", "sn": "body-sn", "ak": "body-ak"}]
    assert [request["path"] for request in requests] == [
        "/thrid/mlmusic/app/channel-1/queryDevcie",
        "/thrid/mlmusic/app/songs",
        "/thrid/mlmusic/app/channel-1/queryDevcie",
        "/thrid/mlmusic/app/songs",
    ]
    assert "lizheng http request" in caplog.text
    assert "lizheng http response" in caplog.text
    assert "endpoint=authentication" in caplog.text
    assert "endpoint=search" in caplog.text
    assert "sn=%3Credacted%3E" in caplog.text
    assert "ak=%3Credacted%3E" in caplog.text
    assert "body-sn" not in caplog.text
    assert "body-ak" not in caplog.text
    assert "iFeqb" not in caplog.text


@pytest.mark.asyncio
async def test_search_candidates_include_only_playable_song_ids(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, _ = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        candidates = await LizhengMusicApi(config, http).search_candidates(
            "晴天", version="现场版", language="国语"
        )

    assert [(song.song_id, song.title, song.artist) for song in candidates] == [
        ("song-1", "", "")
    ]


@pytest.mark.asyncio
async def test_search_candidates_retry_without_filters_when_filtered_search_is_empty(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, requests = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        candidates = await LizhengMusicApi(config, http).search_candidates(
            "hello", version="live", language="en"
        )

    search_requests = [request for request in requests if request["path"].endswith("/songs")]
    assert len(candidates) == 1
    assert search_requests[0]["query"]["version"] == "live"
    assert search_requests[0]["query"]["language"] == "en"
    assert "version" not in search_requests[-1]["query"]
    assert "language" not in search_requests[-1]["query"]


@pytest.mark.asyncio
async def test_recommend_candidates_normalize_display_fields(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, _ = lizheng_server
    app = web.Application()
    key = "0123456789abcdef"

    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/queryDevcie"):
            payload = _encrypted_payload({"qq_api_music": {"sn": "sn-1", "ak": "ak-1"}}, key)
            return web.json_response({"success": True, "data": payload})
        return web.json_response(
            {
                "success": True,
                "data": {
                    "songList": [
                        {"songId": "song-2", "songName": "晴天", "artist": "周杰伦"},
                        {"songId": "", "songName": "invalid"},
                    ]
                },
            }
        )

    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        config = LizhengMusicConfig(
            f"http://127.0.0.1:{port}", "channel-1", "AA:BB", key
        )
        async with aiohttp.ClientSession() as http:
            candidates = await LizhengMusicApi(config, http).recommend_candidates()
    finally:
        await runner.cleanup()

    assert [(song.song_id, song.title, song.artist) for song in candidates] == [
        ("song-2", "晴天", "周杰伦")
    ]


@pytest.mark.asyncio
async def test_http_forbidden_error_preserves_source_and_endpoint(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, _ = lizheng_server
    app = web.Application()

    async def forbidden(_request: web.Request) -> web.Response:
        return web.Response(status=403, text="private proxy detail")

    app.router.add_route("*", "/{tail:.*}", forbidden)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = LizhengMusicConfig(
        f"http://127.0.0.1:{port}", "channel-1", "AA:BB", "0123456789abcdef"
    )
    try:
        async with aiohttp.ClientSession() as http:
            with pytest.raises(LizhengMusicApiError) as caught:
                await LizhengMusicApi(config, http).search("hello")
    finally:
        await runner.cleanup()

    assert caught.value.code == 403
    assert caught.value.source is LizhengErrorSource.HTTP_STATUS
    assert caught.value.endpoint is LizhengEndpoint.AUTHENTICATION
    assert "private proxy detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_business_forbidden_error_preserves_source_and_endpoint(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, _ = lizheng_server
    app = web.Application()

    async def forbidden(_request: web.Request) -> web.Response:
        return web.json_response(
            {"success": False, "code": 403, "message": "private business detail"}
        )

    app.router.add_route("*", "/{tail:.*}", forbidden)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    config = LizhengMusicConfig(
        f"http://127.0.0.1:{port}", "channel-1", "AA:BB", "0123456789abcdef"
    )
    try:
        async with aiohttp.ClientSession() as http:
            with pytest.raises(LizhengMusicApiError) as caught:
                await LizhengMusicApi(config, http).search("hello")
    finally:
        await runner.cleanup()

    assert caught.value.code == 403
    assert caught.value.source is LizhengErrorSource.API_RESPONSE
    assert caught.value.endpoint is LizhengEndpoint.AUTHENTICATION
    assert "private business detail" in str(caught.value)


@pytest.mark.asyncio
async def test_source_for_song_returns_signed_media_url(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, requests = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        source = await LizhengMusicApi(config, http).source_for_song("song-1")

    assert source.song_id == "song-1"
    assert source.url == (
        "https://cdn.example/song.mp3?sign=media-sign&token=media-token"
        "&transDeliveryCode=delivery-secret"
    )
    play_request = requests[-1]
    assert play_request["path"] == "/thrid/mlmusic/app/play-url/song-1"
    query = play_request["query"]
    assert query["channel"] == "channel-1"
    assert query["sn"] == "sn-1"
    assert len(query["sign"]) == 32


@pytest.mark.asyncio
async def test_searched_song_id_resolves_its_own_playback_url(
    lizheng_server: tuple[str, list[dict[str, object]]],
) -> None:
    base_url, requests = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        api = LizhengMusicApi(config, http)
        candidates = await api.search_candidates("hello")
        source = await api.source_for_song(candidates[0].song_id)

    assert candidates[0].song_id == "song-1"
    assert source.song_id == candidates[0].song_id
    assert requests[-1]["path"] == "/thrid/mlmusic/app/play-url/song-1"
    assert source.url.startswith("https://cdn.example/song.mp3?")


@pytest.mark.asyncio
async def test_source_for_song_rejects_mismatched_media_identity() -> None:
    key = "0123456789abcdef"
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith("/queryDevcie"):
            payload = _encrypted_payload({"qq_api_music": {"sn": "sn-1", "ak": "ak-1"}}, key)
            return web.json_response({"success": True, "data": payload})
        return web.json_response(
            {
                "success": True,
                "data": {"mediaList": [{"songId": "different-song", "url": "https://cdn.example/wrong.mp3"}]},
            }
        )

    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        config = LizhengMusicConfig(
            f"http://127.0.0.1:{port}", "channel-1", "AA:BB", key
        )
        async with aiohttp.ClientSession() as http:
            with pytest.raises(LizhengMusicApiError, match="identity"):
                await LizhengMusicApi(config, http).source_for_song("requested-song")
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_media_link_log_preserves_route_and_redacts_all_query_values(
    lizheng_server: tuple[str, list[dict[str, object]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_url, _ = lizheng_server
    config = LizhengMusicConfig(base_url, "channel-1", "AA:BB", "0123456789abcdef")
    async with aiohttp.ClientSession() as http:
        with caplog.at_level("INFO", logger="agent.session.music"):
            source = await LizhengMusicApi(config, http).source_for_song("song-1")

    assert source.song_id == "song-1"
    assert "endpoint=playback_url" in caplog.text
    assert "play-url/song-1" in caplog.text
    assert "https://cdn.example/song.mp3" in caplog.text
    assert "sign=%3Credacted%3E" in caplog.text
    assert "token=%3Credacted%3E" in caplog.text
    assert "transDeliveryCode=%3Credacted%3E" in caplog.text
    assert "media-sign" not in caplog.text
    assert "media-token" not in caplog.text
    assert "delivery-secret" not in caplog.text


def test_environment_requires_aes_key_of_exactly_16_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIZHENG_MUSIC_API_MAC", "AA:BB")
    monkeypatch.setenv("LIZHENG_MUSIC_API_AES_KEY", "short")

    with pytest.raises(LizhengMusicApiError, match="16-byte"):
        LizhengMusicConfig.from_environment()
