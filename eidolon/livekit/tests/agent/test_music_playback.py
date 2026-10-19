from collections.abc import AsyncIterator
from pathlib import Path
import shutil

import anyio
import pytest
from livekit import rtc

from eidolon.livekit.agent.output.music import (
    FfmpegPcmDecoder,
    MusicPlaybackService,
    MusicStatus,
    SAMPLE_RATE,
)


class FakeDecoder:
    def __init__(self, *, fail: bool = False, finite: bool = False) -> None:
        self.fail = fail
        self.finite = finite
        self.started = anyio.Event()
        self.cancelled = anyio.Event()

    async def frames(self, _path) -> AsyncIterator[rtc.AudioFrame]:
        self.started.set()
        try:
            yield rtc.AudioFrame(bytes(640), SAMPLE_RATE, 1, 320)
            if self.fail:
                raise ValueError("invalid mp3")
            if not self.finite:
                await anyio.sleep_forever()
        finally:
            self.cancelled.set()


class FakeSpeech:
    def __init__(self, audio: AsyncIterator[rtc.AudioFrame]) -> None:
        self.interrupted = False
        self.frames: list[rtc.AudioFrame] = []
        self._audio = audio
        self._cancel_scope: anyio.CancelScope | None = None

    async def wait_for_playout(self) -> None:
        with anyio.CancelScope() as cancel_scope:
            self._cancel_scope = cancel_scope
            async for frame in self._audio:
                self.frames.append(frame)

    def interrupt(self, *, force: bool) -> None:
        assert force
        self.interrupted = True
        if self._cancel_scope is not None:
            self._cancel_scope.cancel()


class FakeSession:
    def __init__(self) -> None:
        self.speeches: list[FakeSpeech] = []
        self.speech_started: list[anyio.Event] = []

    def say(self, text: str, *, audio, add_to_chat_ctx: bool) -> FakeSpeech:
        assert text == ""
        assert add_to_chat_ctx is False
        speech = FakeSpeech(audio)
        self.speeches.append(speech)
        self.speech_started.append(anyio.Event())
        self.speech_started[-1].set()
        return speech


@pytest.fixture
def music_dir(tmp_path):
    root = tmp_path / "music"
    root.mkdir()
    (root / "track.mp3").write_bytes(b"ID3fake")
    return root


@pytest.mark.asyncio
async def test_play_streams_frames_through_session_and_completes(music_dir):
    decoder = FakeDecoder(finite=True)
    session = FakeSession()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=decoder) as service:
        accepted = await service.play("play-1", file_name="track.mp3")
        assert accepted.status is MusicStatus.ACCEPTED
        await decoder.started.wait()
        while (status := await service.get_status()).status is not MusicStatus.COMPLETED:
            await anyio.sleep(0)
        assert status == type(accepted)("play-1", MusicStatus.COMPLETED)
        assert session.speeches[0].frames


@pytest.mark.asyncio
async def test_real_ffmpeg_frames_reach_session(tmp_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the real decoder smoke test")
    source = tmp_path / "tone.mp3"
    process = await anyio.open_process(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-codec:a",
            "libmp3lame",
            "-y",
            str(source),
        ],
        stdout=-3,
        stderr=-3,
    )
    assert await process.wait() == 0
    frames = [frame async for frame in FfmpegPcmDecoder().frames(source)]

    assert frames
    assert all(frame.sample_rate == 16000 and frame.num_channels == 1 for frame in frames)


@pytest.mark.asyncio
async def test_stop_cancels_decoder_and_speech(music_dir):
    decoder = FakeDecoder()
    session = FakeSession()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=decoder) as service:
        await service.play("play-1", file_name="track.mp3")
        await decoder.started.wait()
        speech = session.speeches[0]
        result = await service.stop("play-1")

        assert result.status is MusicStatus.STOPPED
        assert speech.interrupted
        assert decoder.cancelled.is_set()


@pytest.mark.asyncio
async def test_aclose_cancels_active_playback(music_dir):
    decoder = FakeDecoder()
    session = FakeSession()
    service = MusicPlaybackService(session, music_dir=music_dir, decoder=decoder)
    await service.__aenter__()
    try:
        await service.play("play-1", file_name="track.mp3")
        await decoder.started.wait()
        await service.aclose()
        assert session.speeches[0].interrupted
        assert decoder.cancelled.is_set()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_replace_cancels_previous_source(music_dir):
    decoder = FakeDecoder()
    session = FakeSession()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=decoder) as service:
        await service.play("play-1", file_name="track.mp3")
        await decoder.started.wait()
        accepted = await service.play("play-2", file_name="track.mp3")

        assert accepted.playback_ref == "play-2"
        assert accepted.status is MusicStatus.ACCEPTED
        assert len(session.speeches) == 2
        assert decoder.cancelled.is_set()
        await service.stop()


@pytest.mark.asyncio
async def test_decoder_failure_sets_failed_status(music_dir):
    decoder = FakeDecoder(fail=True)
    async with MusicPlaybackService(FakeSession(), music_dir=music_dir, decoder=decoder) as service:
        await service.play("play-1", file_name="track.mp3")
        await decoder.started.wait()
        while (status := await service.get_status()).status is not MusicStatus.FAILED:
            await anyio.sleep(0)
        assert status.status is MusicStatus.FAILED
        assert status.error == "audio output failed"


@pytest.mark.asyncio
async def test_rejects_path_outside_music_dir(music_dir, tmp_path):
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"ID3fake")
    async with MusicPlaybackService(
        FakeSession(), music_dir=music_dir, decoder=FakeDecoder()
    ) as service:
        with pytest.raises(ValueError, match="inside music_dir"):
            await service.play("play-1", file_name=str(outside))


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "close", "replace", "cancel"])
async def test_preparation_is_owned_and_cancellable(music_dir, action):
    import asyncio

    class SlowCache:
        def __init__(self):
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def get(self, song_id):
            if song_id == "ready":
                return music_dir / "track.mp3"
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

    cache = SlowCache()
    session = FakeSession()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=FakeDecoder(), cache=cache) as service:
        pending = asyncio.create_task(service.play_song("old", song_id="slow"))
        await cache.entered.wait()
        assert (await service.get_status()).status is MusicStatus.PREPARING
        if action == "stop":
            assert (await service.stop("old")).status is MusicStatus.STOPPED
        elif action == "close":
            await service.aclose()
        elif action == "replace":
            await service.play_song("new", song_id="ready")
        else:
            pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert cache.cancelled.is_set()
        assert len(session.speeches) == (1 if action == "replace" else 0)
        if action == "replace":
            assert (await service.get_status()).playback_ref == "new"
            assert (await service.get_status()).status is MusicStatus.PLAYING


@pytest.mark.asyncio
async def test_duplicate_reference_does_not_restart_and_wrong_stop_cannot_stop_new_output(music_dir):
    session = FakeSession()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=FakeDecoder()) as service:
        await service.play("same", file_name="track.mp3")
        await service.play("same", file_name="track.mp3")
        assert len(session.speeches) == 1
        with pytest.raises(ValueError, match="different source"):
            other = music_dir / "other.mp3"
            other.write_bytes(b"ID3")
            await service.play("same", file_name="other.mp3")
        with pytest.raises(LookupError):
            await service.stop("stale")
        assert not session.speeches[0].interrupted


@pytest.mark.asyncio
async def test_framework_interrupt_updates_music_status(music_dir):
    session = FakeSession()
    decoder = FakeDecoder()
    async with MusicPlaybackService(session, music_dir=music_dir, decoder=decoder) as service:
        await service.play("ref", file_name="track.mp3")
        await decoder.started.wait()
        session.speeches[0].interrupt(force=True)
        with anyio.fail_after(1):
            while (await service.get_status()).status is not MusicStatus.INTERRUPTED:
                await anyio.sleep(0)
