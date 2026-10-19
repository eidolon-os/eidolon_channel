from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from ...common.music import MusicProviderError
from ..output.music import MusicPlayback, MusicPlaybackService, MusicStatus
from ..output.music_cache import MusicDownloadError

logger = logging.getLogger("agent.session.music")


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

    async def play(
        self,
        *,
        request_id: str,
        playback_ref: str,
        song_id: str,
    ) -> MusicCommandReceipt:
        try:
            result = await self._playback.play_song(
                playback_ref,
                song_id=song_id,
            )
            logger.info(
                "music command completed request_id=%s playback_ref=%s song_id=%s status=%s",
                request_id,
                result.playback_ref,
                song_id,
                result.status.value,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            provider_code = "-"
            provider_source = "-"
            provider_endpoint = "-"
            match exc:
                case MusicProviderError():
                    stage = "provider_api"
                    provider_source = str(exc.source)
                    provider_endpoint = str(exc.endpoint)
                    match exc.code:
                        case int() as code:
                            provider_code = str(code)
                        case str():
                            pass
                case MusicDownloadError():
                    stage = "source_download"
                case OSError():
                    stage = "local_io"
                case _:
                    stage = "music_setup"
            logger.warning(
                "music command failed request_id=%s stage=%s error_type=%s "
                "provider_code=%s provider_source=%s provider_endpoint=%s",
                request_id,
                stage,
                type(exc).__name__,
                provider_code,
                provider_source,
                provider_endpoint,
            )
            return MusicCommandReceipt(
                request_id, playback_ref, MusicCommandStatus.FAILED, "music playback setup failed"
            )
        return MusicCommandReceipt(
            request_id,
            result.playback_ref,
            MusicCommandStatus.ACCEPTED,
        )

    async def stop(
        self, *, request_id: str, playback_ref: str | None = None
    ) -> MusicCommandReceipt:
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
