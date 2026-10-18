import hashlib
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from eidolon.livekit.agent.session.lizheng_music import LizhengMusicApi, LizhengMusicConfig
from eidolon.livekit.agent.session.music_cache import (
    MusicCache,
    MusicDownloadError,
    MusicSource,
)


@pytest.fixture
async def music_server():
    body = b"ID3" + b"cached-music"
    app = web.Application()

    async def serve(_request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="audio/mpeg")

    async def broken(_request: web.Request) -> web.Response:
        return web.Response(body=b"not-mp3", content_type="audio/mpeg")

    app.router.add_get("/track.mp3", serve)
    app.router.add_get("/broken.mp3", broken)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_downloads_once_and_returns_atomic_cache(tmp_path: Path, music_server: str) -> None:
    calls = 0
    body = b"ID3" + b"cached-music"

    async def resolve(_song_id: str) -> MusicSource:
        nonlocal calls
        calls += 1
        return MusicSource(
            song_id="provider:track",
            url=f"{music_server}/track.mp3",
            expected_size=len(body),
            expected_sha256=hashlib.sha256(body).hexdigest(),
        )

    async with aiohttp.ClientSession() as http:
        cache = MusicCache(tmp_path, source_resolver=resolve, http_session=http)
        first = await cache.get("provider:track")
        second = await cache.get("provider:track")

    assert first == second
    assert first.read_bytes() == body
    assert calls == 1
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_rejects_non_mp3_without_publishing(tmp_path: Path, music_server: str) -> None:
    async def resolve(_song_id: str) -> MusicSource:
        return MusicSource(song_id="broken", url=f"{music_server}/broken.mp3")

    async with aiohttp.ClientSession() as http:
        cache = MusicCache(tmp_path, source_resolver=resolve, http_session=http)
        with pytest.raises(MusicDownloadError, match="not an MP3"):
            await cache.get("broken")

    assert not list(tmp_path.glob("*.mp3"))
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.asyncio
async def test_accepts_mp3_without_id3_tag(tmp_path: Path) -> None:
    body = b"\xff\xfb\x90\x64" + b"mpeg-frame-data"
    app = web.Application()

    async def serve(_request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="audio/mpeg")

    app.router.add_get("/track.mp3", serve)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    async def resolve(_song_id: str) -> MusicSource:
        return MusicSource(song_id="mpeg", url=f"http://127.0.0.1:{port}/track.mp3")

    try:
        async with aiohttp.ClientSession() as http:
            cache = MusicCache(tmp_path, source_resolver=resolve, http_session=http)
            path = await cache.get("mpeg")
    finally:
        await runner.cleanup()

    assert path.read_bytes() == body


@pytest.mark.asyncio
async def test_rejects_untrusted_url_before_network_access(tmp_path: Path) -> None:
    async def resolve(_song_id: str) -> MusicSource:
        return MusicSource(song_id="bad", url="file:///etc/passwd")

    async with aiohttp.ClientSession() as http:
        cache = MusicCache(tmp_path, source_resolver=resolve, http_session=http)
        with pytest.raises(MusicDownloadError, match=r"HTTP\(S\)"):
            await cache.get("bad")


@pytest.mark.asyncio
async def test_search_candidate_song_id_downloads_from_its_resolved_url(tmp_path: Path) -> None:
    key = "0123456789abcdef"
    audio = b"ID3" + b"song-audio"
    requests: list[tuple[str, str]] = []
    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        requests.append((request.path, request.headers.get("Accept", "")))
        if request.path.endswith("/queryDevcie"):
            padder = PKCS7(algorithms.AES.block_size).padder()
            padded = padder.update(
                json.dumps({"qq_api_music": {"sn": "sn-1", "ak": "ak-1"}}).encode()
            ) + padder.finalize()
            encryptor = Cipher(algorithms.AES(key.encode()), modes.ECB()).encryptor()
            encrypted = encryptor.update(padded) + encryptor.finalize()
            import base64

            return web.json_response(
                {"success": True, "data": base64.b64encode(encrypted).decode()}
            )
        if request.path.endswith("/songs"):
            return web.json_response(
                {"success": True, "data": [{"songId": "song-selected", "songName": "Selected"}]}
            )
        if request.path.endswith("/play-url/song-selected"):
            return web.json_response(
                {"success": True, "data": {"mediaList": [{"url": f"http://127.0.0.1:{port}/audio/song-selected.mp3"}]}}
            )
        if request.path == "/audio/song-selected.mp3":
            return web.Response(body=audio, content_type="audio/mpeg")
        raise AssertionError(request.path)

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
            api = LizhengMusicApi(config, http)
            candidates = await api.search_candidates("Selected")
            cache = MusicCache(
                tmp_path,
                source_resolver=api.source_for_song,
                http_session=http,
            )
            path = await cache.get(candidates[0].song_id)
    finally:
        await runner.cleanup()

    assert candidates[0].song_id == "song-selected"
    assert path.read_bytes() == audio
    assert any(
        path == "/thrid/mlmusic/app/play-url/song-selected"
        for path, _accept in requests
    )
    assert ("/audio/song-selected.mp3", "audio/mpeg") in requests
