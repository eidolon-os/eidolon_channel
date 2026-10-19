from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import anyio
from livekit import rtc

from .music_cache import MusicCache

logger = logging.getLogger("agent.session.music")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000
MAX_DECODED_BYTES = SAMPLE_RATE * 2 * 60 * 60


class MusicStatus(StrEnum):
    PREPARING = "preparing"
    ACCEPTED = "accepted"
    PLAYING = "playing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MusicPlayback:
    playback_ref: str
    status: MusicStatus
    error: str | None = None


class SpeechHandle(Protocol):
    interrupted: bool

    def interrupt(self, *, force: bool) -> None: ...

    async def wait_for_playout(self) -> None: ...


class MusicSession(Protocol):
    def say(
        self,
        text: str,
        *,
        audio: AsyncIterator[rtc.AudioFrame],
        add_to_chat_ctx: bool,
    ) -> SpeechHandle: ...


class FfmpegPcmDecoder:
    def __init__(self, executable: str = "ffmpeg") -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"ffmpeg executable not found: {executable}")
        self._executable = resolved

    async def frames(self, path: Path) -> AsyncIterator[rtc.AudioFrame]:
        process = await anyio.open_process(
            [
                self._executable,
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "pipe:1",
            ],
            stdout=-1,
            stderr=-3,
        )
        pending = bytearray()
        decoded_bytes = 0
        try:
            if process.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe was not created")
            while True:
                try:
                    chunk = await process.stdout.receive(FRAME_BYTES)
                except anyio.EndOfStream:
                    break
                pending.extend(chunk)
                while len(pending) >= FRAME_BYTES:
                    pcm = bytes(pending[:FRAME_BYTES])
                    del pending[:FRAME_BYTES]
                    decoded_bytes += len(pcm)
                    if decoded_bytes > MAX_DECODED_BYTES:
                        raise ValueError("decoded audio exceeds two-hour limit")
                    yield rtc.AudioFrame(pcm, SAMPLE_RATE, 1, FRAME_BYTES // 2)
            if pending:
                decoded_bytes += len(pending)
                if decoded_bytes > MAX_DECODED_BYTES:
                    raise ValueError("decoded audio exceeds two-hour limit")
                pending.extend(bytes(FRAME_BYTES - len(pending)))
                yield rtc.AudioFrame(bytes(pending), SAMPLE_RATE, 1, FRAME_BYTES // 2)
            await process.wait()
            if process.returncode != 0:
                raise ValueError(f"ffmpeg could not decode audio (exit {process.returncode})")
        finally:
            if process.returncode is None:
                process.terminate()
                with anyio.move_on_after(1.0) as timeout_scope:
                    await process.wait()
                if timeout_scope.cancelled_caught:
                    process.kill()
                    await process.wait()


class MusicPlaybackService:
    def __init__(
        self,
        session: MusicSession,
        *,
        music_dir: Path,
        decoder: FfmpegPcmDecoder | None = None,
        cache: MusicCache | None = None,
        max_file_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self._session = session
        self._music_dir = music_dir.resolve(strict=True)
        self._decoder = decoder or FfmpegPcmDecoder()
        self._cache = cache
        self._max_file_bytes = max_file_bytes
        self._lock = anyio.Lock()
        self._generation = 0
        self._playback_ref: str | None = None
        self._status: MusicStatus | None = None
        self._error: str | None = None
        self._open = False
        self._task: asyncio.Task[None] | None = None
        self._speech: SpeechHandle | None = None
        self._source_key: tuple[str, str] | None = None

    async def __aenter__(self) -> MusicPlaybackService:
        self._open = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            self._open = False
            await self._stop_locked(MusicStatus.INTERRUPTED)

    async def play(self, playback_ref: str, *, file_name: str) -> MusicPlayback:
        path = self._resolve_file(file_name)
        return await self._start(playback_ref, source_key=("file", str(path)), path=path)

    async def play_song(self, playback_ref: str, *, song_id: str) -> MusicPlayback:
        if self._cache is None:
            raise RuntimeError("MusicPlaybackService has no music cache")
        return await self._start(playback_ref, source_key=("song", song_id))

    async def _start(
        self, playback_ref: str, *, source_key: tuple[str, str], path: Path | None = None,
    ) -> MusicPlayback:
        async with self._lock:
            if not self._open:
                raise RuntimeError("MusicPlaybackService must be used as an async context manager")
            if playback_ref == self._playback_ref:
                if source_key != self._source_key:
                    raise ValueError("playback_ref reused for a different source")
                return self._snapshot()
            await self._stop_locked(MusicStatus.INTERRUPTED)
            self._generation += 1
            generation = self._generation
            self._playback_ref = playback_ref
            self._source_key = source_key
            self._status = MusicStatus.PREPARING
            self._error = None
            ready: asyncio.Future[MusicPlayback] = asyncio.get_running_loop().create_future()
            self._task = asyncio.create_task(
                self._prepare_and_play(generation, playback_ref, source_key, path, ready),
                name=f"music-playback-{playback_ref}",
            )
        # Download is owned by the same task as playback. Stop/replacement/close
        # can cancel preparation without waiting for network or taking this lock.
        try:
            return await ready
        except asyncio.CancelledError:
            async with self._lock:
                if generation == self._generation:
                    await self._stop_locked(MusicStatus.INTERRUPTED)
            raise

    async def _prepare_and_play(
        self, generation: int, playback_ref: str, source_key: tuple[str, str],
        path: Path | None, ready: asyncio.Future[MusicPlayback],
    ) -> None:
        try:
            if path is None:
                assert self._cache is not None
                path = await self._cache.get(source_key[1])
            if generation != self._generation:
                return
            self._status = MusicStatus.ACCEPTED
            # The command is accepted, not evidence that the endpoint is audible.
            if not ready.done():
                ready.set_result(MusicPlayback(playback_ref, MusicStatus.ACCEPTED))
            await self._play(generation, path)
        except asyncio.CancelledError:
            if not ready.done():
                ready.cancel()
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            if generation == self._generation:
                self._status = MusicStatus.FAILED
                self._error = "music preparation failed"
            if not ready.done():
                ready.set_exception(exc)
        finally:
            if not ready.done():
                ready.cancel()

    async def stop(self, playback_ref: str | None = None) -> MusicPlayback:
        async with self._lock:
            if playback_ref is not None and playback_ref != self._playback_ref:
                raise LookupError("playback_ref is not current")
            await self._stop_locked(MusicStatus.STOPPED)
            return self._snapshot()

    async def get_status(self) -> MusicPlayback | None:
        async with self._lock:
            return self._snapshot() if self._playback_ref is not None else None

    def _resolve_file(self, file_name: str) -> Path:
        candidate = (self._music_dir / file_name).resolve(strict=True)
        if not candidate.is_relative_to(self._music_dir):
            raise ValueError("music file must stay inside music_dir")
        if not candidate.is_file() or candidate.suffix.lower() != ".mp3":
            raise ValueError("music file must be an MP3 file")
        if candidate.stat().st_size > self._max_file_bytes:
            raise ValueError("music file exceeds configured size limit")
        return candidate

    async def _play(self, generation: int, path: Path) -> None:
        async def audio() -> AsyncIterator[rtc.AudioFrame]:
            async for frame in self._decoder.frames(path):
                if generation != self._generation:
                    return
                if self._status is MusicStatus.ACCEPTED:
                    self._status = MusicStatus.PLAYING
                yield frame

        try:
            speech = self._session.say("", audio=audio(), add_to_chat_ctx=False)
            self._speech = speech
            await speech.wait_for_playout()
            if generation != self._generation:
                return
            self._status = MusicStatus.INTERRUPTED if speech.interrupted else MusicStatus.COMPLETED
            if speech.interrupted:
                self._error = "speech interrupted"
        except anyio.get_cancelled_exc_class():
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "music playback failed playback_ref=%s error_type=%s error=%s",
                self._playback_ref,
                type(exc).__name__,
                "audio output failed",
            )
            if generation == self._generation:
                self._status = MusicStatus.FAILED
                self._error = "audio output failed"

    async def _stop_locked(self, status: MusicStatus) -> None:
        task, speech = self._task, self._speech
        if task is None:
            return
        if not task.done():
            self._generation += 1
            self._status = status
            if speech is not None:
                speech.interrupt(force=True)
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._task = None
        self._speech = None

    def _snapshot(self) -> MusicPlayback:
        assert self._playback_ref is not None and self._status is not None
        return MusicPlayback(self._playback_ref, self._status, self._error)
