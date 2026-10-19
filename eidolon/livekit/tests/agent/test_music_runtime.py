"""Music lifecycle regressions using a fake catalogue and the real playback owner."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eidolon.livekit.agent.eidolon_agent_rpc.channel_tools import ChannelToolCall
from eidolon.livekit.agent.output.music import MusicPlaybackService
from eidolon.livekit.agent.runtime.music import create_music_runtime, music_cache_directory
from eidolon.livekit.agent.session.music import SessionMusicRuntime
from eidolon.livekit.agent.session.music_commands import MusicCommandService
from .test_music_commands import FakeCache
from .test_music_playback import FakeDecoder, FakeSession


@pytest.mark.asyncio
async def test_custom_ref_can_stop_and_cancel_playback(tmp_path):
    source = tmp_path / "song.mp3"
    source.write_bytes(b"ID3fake")
    decoder = FakeDecoder()
    async with MusicPlaybackService(
        FakeSession(), music_dir=tmp_path, decoder=decoder, cache=FakeCache(source)
    ) as playback:
        runtime = SessionMusicRuntime(None, music_dir=tmp_path)
        runtime.playback = playback
        runtime.commands = MusicCommandService(playback)
        runtime._api = SimpleNamespace()
        call = ChannelToolCall(
            "turn", "call", "music.play", {"song_id": "song", "playback_ref": "chosen"}
        )
        result = await runtime.execute_channel_tool(call)
        assert result.ok and result.content["playback_ref"] == "chosen"
        stopped = await runtime.execute_channel_tool(
            ChannelToolCall("t2", "c2", "music.stop", {"playback_ref": "chosen"})
        )
        assert stopped.ok and stopped.content["status"] == "interrupted"
        await runtime.execute_channel_tool(call)
        await runtime.cancel_channel_tool(call)
        assert (await playback.get_status()).status.value == "interrupted"


@pytest.mark.asyncio
async def test_runtime_closes_from_a_different_task(tmp_path):
    service = MusicPlaybackService(FakeSession(), music_dir=tmp_path, decoder=FakeDecoder())
    await service.__aenter__()
    await asyncio.create_task(service.aclose())
    await service.aclose()


def test_cache_follows_host_root(monkeypatch, tmp_path):
    monkeypatch.setenv("EIDOLON_CACHE_ROOT", str(tmp_path))
    assert music_cache_directory() == tmp_path / "channel/music"


@pytest.mark.asyncio
async def test_disabled_music_never_initializes_http_or_storage(monkeypatch):
    monkeypatch.delenv("LIZHENG_MUSIC_API_MAC", raising=False)
    monkeypatch.delenv("LIZHENG_MUSIC_API_AES_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled music must not initialize")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert await create_music_runtime(FakeSession()) is None


@pytest.mark.asyncio
async def test_cache_failure_disables_music_and_closes_http(monkeypatch, tmp_path):
    from eidolon.livekit.agent.runtime import music

    monkeypatch.setenv("LIZHENG_MUSIC_API_MAC", "test")
    monkeypatch.setenv("LIZHENG_MUSIC_API_AES_KEY", "0123456789abcdef")
    monkeypatch.setenv("EIDOLON_CACHE_ROOT", str(tmp_path))
    http = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(music.aiohttp, "ClientSession", lambda **kwargs: http)

    def denied(*args, **kwargs):
        raise PermissionError("unwritable")

    monkeypatch.setattr(Path, "mkdir", denied)
    assert await create_music_runtime(FakeSession()) is None
    http.close.assert_awaited_once()
