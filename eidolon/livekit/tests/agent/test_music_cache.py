import hashlib
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

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
async def test_rejects_untrusted_url_before_network_access(tmp_path: Path) -> None:
    async def resolve(_song_id: str) -> MusicSource:
        return MusicSource(song_id="bad", url="file:///etc/passwd")

    async with aiohttp.ClientSession() as http:
        cache = MusicCache(tmp_path, source_resolver=resolve, http_session=http)
        with pytest.raises(MusicDownloadError, match=r"HTTP\(S\)"):
            await cache.get("bad")
