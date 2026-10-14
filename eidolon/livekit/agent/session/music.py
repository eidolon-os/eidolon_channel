from __future__ import annotations

import logging
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import anyio
import aiohttp
from livekit import rtc
from anyio.abc import CancelScope, TaskGroup, TaskStatus

from .lizheng_music import LizhengMusicApi, LizhengMusicApiError, LizhengMusicConfig
from ..eidolon_agent_rpc.channel_tools import ChannelToolCall, ChannelToolResult
from .music_cache import MusicCache

logger = logging.getLogger("agent.session.music")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1000
MAX_DECODED_BYTES = SAMPLE_RATE * 2 * 60 * 60


class MusicStatus(StrEnum):
    ACCEPTED = "accepted"
    PLAYING = "playing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
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
        self._task_group: TaskGroup | None = None
        self._cancel_scope: CancelScope | None = None
        self._finished: anyio.Event | None = None
        self._speech: SpeechHandle | None = None

    async def __aenter__(self) -> MusicPlaybackService:
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        async with self._lock:
            await self._stop_locked(MusicStatus.INTERRUPTED)
        if self._task_group is not None:
            self._task_group.cancel_scope.cancel()
            await self._task_group.__aexit__(None, None, None)
            self._task_group = None

    async def play(self, playback_ref: str, *, file_name: str) -> MusicPlayback:
        path = self._resolve_file(file_name)
        return await self._start(playback_ref, path)

    async def play_song(self, playback_ref: str, *, song_id: str) -> MusicPlayback:
        if self._cache is None:
            raise RuntimeError("MusicPlaybackService has no music cache")
        path = await self._cache.get(song_id)
        return await self._start(playback_ref, path)

    async def _start(self, playback_ref: str, path: Path) -> MusicPlayback:
        async with self._lock:
            await self._stop_locked(MusicStatus.INTERRUPTED)
            if self._task_group is None:
                raise RuntimeError("MusicPlaybackService must be used as an async context manager")
            self._generation += 1
            generation = self._generation
            self._playback_ref = playback_ref
            self._status = MusicStatus.ACCEPTED
            self._error = None
            await self._task_group.start(
                self._run,
                generation,
                path,
                name=f"music-playback-{playback_ref}",
            )
            return MusicPlayback(playback_ref, MusicStatus.ACCEPTED)

    async def stop(self, playback_ref: str | None = None) -> MusicPlayback:
        async with self._lock:
            if self._playback_ref is not None and playback_ref not in (None, self._playback_ref):
                return self._snapshot()
            await self._stop_locked(MusicStatus.INTERRUPTED)
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

    async def _run(
        self,
        generation: int,
        path: Path,
        *,
        task_status: TaskStatus[CancelScope] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        with anyio.CancelScope() as cancel_scope:
            self._cancel_scope = cancel_scope
            self._finished = anyio.Event()
            task_status.started(cancel_scope)
            try:
                await self._play(generation, path)
            finally:
                if generation == self._generation:
                    self._speech = None
                    self._cancel_scope = None
                self._finished.set()

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
            self._status = (
                MusicStatus.INTERRUPTED if speech.interrupted else MusicStatus.COMPLETED
            )
            if speech.interrupted:
                self._error = "speech interrupted"
        except anyio.get_cancelled_exc_class():
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("music playback failed ref=%s: %s", self._playback_ref, exc)
            if generation == self._generation:
                self._status = MusicStatus.FAILED
                self._error = str(exc)
    async def _stop_locked(self, status: MusicStatus) -> None:
        cancel_scope, speech, finished = self._cancel_scope, self._speech, self._finished
        if cancel_scope is None or finished is None:
            return
        self._generation += 1
        self._status = status
        if speech is not None:
            speech.interrupt(force=True)
        cancel_scope.cancel()
        await finished.wait()
        self._finished = None
        self._speech = None
        self._cancel_scope = None

    def _snapshot(self) -> MusicPlayback:
        assert self._playback_ref is not None and self._status is not None
        return MusicPlayback(self._playback_ref, self._status, self._error)


class SessionMusicRuntime:
    def __init__(self, session: MusicSession, *, music_dir: Path) -> None:
        self._session = session
        self._music_dir = music_dir
        self._http: aiohttp.ClientSession | None = None
        self.playback: MusicPlaybackService | None = None
        from .music_commands import MusicCommandService

        self.commands: MusicCommandService | None = None

    @classmethod
    async def create(cls, session: MusicSession, *, music_dir: Path) -> SessionMusicRuntime | None:
        mac = os.getenv("LIZHENG_MUSIC_API_MAC", "").strip()
        aes_key = os.getenv("LIZHENG_MUSIC_API_AES_KEY", "").strip()
        if not mac and not aes_key:
            return None
        try:
            config = LizhengMusicConfig.from_environment()
        except LizhengMusicApiError as exc:
            logger.warning("online music disabled: %s", exc)
            return None
        music_dir.mkdir(parents=True, exist_ok=True)
        http = aiohttp.ClientSession()
        try:
            api = LizhengMusicApi(config, http)
            from .music_commands import MusicCommandService

            runtime = cls(session, music_dir=music_dir)
            runtime._http = http
            cache = MusicCache(
                music_dir,
                source_resolver=api.source_for_song,
                http_session=http,
            )
            runtime.playback = MusicPlaybackService(session, music_dir=music_dir, cache=cache)
            await runtime.playback.__aenter__()
            runtime.commands = MusicCommandService(runtime.playback)
            return runtime
        except BaseException:
            await http.close()
            raise

    async def execute_channel_tool(self, call: ChannelToolCall) -> ChannelToolResult:
        commands = self.commands
        if commands is None:
            return _channel_tool_error(call, "MUSIC_DISABLED", "Online music is disabled")
        match call.name:
            case "music.play":
                song_id = call.arguments.get("song_id")
                playback_ref = call.arguments.get("playback_ref", f"music:{call.call_id}")
                if not isinstance(song_id, str) or not song_id.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "song_id is required")
                if not isinstance(playback_ref, str) or not playback_ref.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "playback_ref is invalid")
                receipt = await commands.play(
                    request_id=call.call_id,
                    playback_ref=f"music:{call.call_id}",
                    song_id=song_id,
                )
            case "music.stop":
                playback_ref = call.arguments.get("playback_ref")
                if playback_ref is not None and not isinstance(playback_ref, str):
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "playback_ref is invalid")
                receipt = await commands.stop(
                    request_id=call.call_id,
                    playback_ref=playback_ref,
                )
            case "music.status":
                receipt = await commands.status(request_id=call.call_id)
            case _:
                return _channel_tool_error(call, "TOOL_NOT_FOUND", "unsupported Channel tool")
        content: dict[str, object] = {"status": str(receipt.status)}
        if receipt.playback_ref is not None:
            content["playback_ref"] = receipt.playback_ref
        if receipt.error is not None:
            content["error"] = receipt.error
        success = receipt.status.value in {"accepted", "completed", "interrupted"}
        return ChannelToolResult(
            turn_id=call.turn_id,
            call_id=call.call_id,
            name=call.name,
            ok=success,
            content=content,
            error_code="MUSIC_COMMAND_FAILED" if not success else "",
            error_message=(receipt.error or "") if not success else "",
        )

    async def cancel_channel_tool(self, call: ChannelToolCall) -> None:
        if call.name == "music.play" and self.playback is not None:
            current = await self.playback.get_status()
            if current is not None and current.playback_ref == f"music:{call.call_id}":
                await self.playback.stop(current.playback_ref)

    async def aclose(self) -> None:
        if self.playback is not None:
            await self.playback.aclose()
            self.playback = None
        if self._http is not None:
            await self._http.close()
            self._http = None


def _channel_tool_error(
    call: ChannelToolCall,
    code: str,
    message: str,
) -> ChannelToolResult:
    return ChannelToolResult(
        turn_id=call.turn_id,
        call_id=call.call_id,
        name=call.name,
        ok=False,
        content={},
        error_code=code,
        error_message=message,
    )
