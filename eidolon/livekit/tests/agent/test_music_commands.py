from pathlib import Path

import pytest

from eidolon.livekit.agent.session.music import MusicPlaybackService
from eidolon.livekit.agent.session.music_commands import (
    MusicCommandService,
    MusicCommandStatus,
)
from .test_music_playback import FakeDecoder, FakeSession


class FakeCache:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def get(self, _song_id: str) -> Path:
        return self.path


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    root = tmp_path / "music"
    root.mkdir()
    (root / "track.mp3").write_bytes(b"ID3fake")
    return root


@pytest.mark.asyncio
async def test_play_returns_accepted_receipt(music_dir: Path) -> None:
    session = FakeSession()
    async with MusicPlaybackService(
        session,
        music_dir=music_dir,
        decoder=FakeDecoder(),
        cache=FakeCache(music_dir / "track.mp3"),
    ) as playback:
        commands = MusicCommandService(playback)
        receipt = await commands.play(
            request_id="request-1",
            playback_ref="play-1",
            song_id="provider:track",
        )

    assert receipt.request_id == "request-1"
    assert receipt.playback_ref == "play-1"
    assert receipt.status is MusicCommandStatus.ACCEPTED


@pytest.mark.asyncio
async def test_status_without_playback_returns_not_found(music_dir: Path) -> None:
    async with MusicPlaybackService(
        FakeSession(),
        music_dir=music_dir,
        decoder=FakeDecoder(),
    ) as playback:
        receipt = await MusicCommandService(playback).status(request_id="request-1")

    assert receipt.status is MusicCommandStatus.NOT_FOUND
    assert receipt.playback_ref is None


@pytest.mark.asyncio
async def test_stop_maps_interrupted_status(music_dir: Path) -> None:
    session = FakeSession()
    decoder = FakeDecoder()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=decoder) as playback:
        await playback.play("play-1", file_name="track.mp3")
        await decoder.started.wait()
        receipt = await MusicCommandService(playback).stop(
            request_id="request-2",
            playback_ref="play-1",
        )

    assert receipt.status is MusicCommandStatus.INTERRUPTED
    assert receipt.playback_ref == "play-1"
