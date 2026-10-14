from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from .music import MusicPlayback, MusicPlaybackService, MusicStatus


class MusicCommandStatus(StrEnum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class MusicCommandReceipt:
    request_id: str
    playback_ref: str | None
    status: MusicCommandStatus
    error: str | None = None


class MusicCommandService:
    def __init__(self, playback: MusicPlaybackService) -> None:
        self._playback = playback

    async def play(self, *, request_id: str, playback_ref: str, song_id: str) -> MusicCommandReceipt:
        try:
            result = await self._playback.play_song(playback_ref, song_id=song_id)
        except (OSError, RuntimeError, ValueError) as exc:
            return MusicCommandReceipt(request_id, playback_ref, MusicCommandStatus.FAILED, str(exc))
        return MusicCommandReceipt(
            request_id,
            result.playback_ref,
            MusicCommandStatus.ACCEPTED,
        )

    async def stop(self, *, request_id: str, playback_ref: str | None = None) -> MusicCommandReceipt:
        if await self._playback.get_status() is None:
            return MusicCommandReceipt(request_id, None, MusicCommandStatus.NOT_FOUND)
        result = await self._playback.stop(playback_ref)
        return MusicCommandReceipt(
            request_id,
            result.playback_ref,
            _command_status(result),
            result.error,
        )

    async def status(self, *, request_id: str) -> MusicCommandReceipt:
        result = await self._playback.get_status()
        if result is None:
            return MusicCommandReceipt(request_id, None, MusicCommandStatus.NOT_FOUND)
        return MusicCommandReceipt(
            request_id,
            result.playback_ref,
            _command_status(result),
            result.error,
        )


def _command_status(result: MusicPlayback) -> MusicCommandStatus:
    match result.status:
        case MusicStatus.ACCEPTED | MusicStatus.PLAYING:
            return MusicCommandStatus.ACCEPTED
        case MusicStatus.COMPLETED:
            return MusicCommandStatus.COMPLETED
        case MusicStatus.INTERRUPTED:
            return MusicCommandStatus.INTERRUPTED
        case MusicStatus.FAILED:
            return MusicCommandStatus.FAILED
        case unreachable:
            assert_never(unreachable)
