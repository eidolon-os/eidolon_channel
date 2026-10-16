from pathlib import Path

import pytest

from eidolon.livekit.agent.session.music import MusicPlaybackService
from eidolon.livekit.agent.session.lizheng_music import LizhengMusicApiError
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


class FailingCache:
    def __init__(self, error: RuntimeError) -> None:
        self.error = error

    async def get(self, _song_id: str) -> Path:
        raise self.error


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
async def test_play_logs_provider_failure_stage_without_error_details(
    music_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_detail = "signed-url?ak=secret&token=private"
    async with MusicPlaybackService(
        FakeSession(),
        music_dir=music_dir,
        decoder=FakeDecoder(),
        cache=FailingCache(LizhengMusicApiError(401, secret_detail)),
    ) as playback:
        with caplog.at_level("WARNING", logger="agent.session.music"):
            receipt = await MusicCommandService(playback).play(
                request_id="request-42",
                playback_ref="play-42",
                song_id="song-42",
            )

    assert receipt.status is MusicCommandStatus.FAILED
    assert (
        "request_id=request-42 stage=provider_api error_type=LizhengMusicApiError "
        "provider_code=401" in caplog.text
    )
    assert secret_detail not in caplog.text
    assert "secret" not in caplog.text


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
