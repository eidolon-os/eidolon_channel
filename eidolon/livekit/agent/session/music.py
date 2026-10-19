"""Per-session music tool orchestration; provider construction is a startup concern."""

from __future__ import annotations
import logging
import aiohttp
from pathlib import Path
from ...common.music import MusicCatalogue, MusicProviderError
from ..output.music import MusicPlaybackService, MusicSession
from ..output.music_cache import MusicCache
from ..eidolon_agent_rpc.channel_tools import ChannelToolCall, ChannelToolResult

logger = logging.getLogger("agent.session.music")


class SessionMusicRuntime:
    def __init__(self, session: MusicSession, *, music_dir: Path) -> None:
        self._session = session
        self._music_dir = music_dir
        self._http: aiohttp.ClientSession | None = None
        self._api: MusicCatalogue | None = None
        self.playback: MusicPlaybackService | None = None
        from .music_commands import MusicCommandService

        self.commands: MusicCommandService | None = None

    @classmethod
    async def create(
        cls,
        session: MusicSession,
        *,
        music_dir: Path,
        api: MusicCatalogue,
        http: aiohttp.ClientSession,
    ) -> SessionMusicRuntime:
        from .music_commands import MusicCommandService

        music_dir.mkdir(parents=True, exist_ok=True)
        runtime = cls(session, music_dir=music_dir)
        runtime._http = http
        runtime._api = api
        cache = MusicCache(music_dir, source_resolver=api.source_for_song, http_session=http)
        runtime.playback = MusicPlaybackService(session, music_dir=music_dir, cache=cache)
        await runtime.playback.__aenter__()
        runtime.commands = MusicCommandService(runtime.playback)
        return runtime

    async def execute_channel_tool(self, call: ChannelToolCall) -> ChannelToolResult:
        commands = self.commands
        api = self._api
        if commands is None or api is None:
            return _channel_tool_error(call, "MUSIC_DISABLED", "Online music is disabled")
        match call.name:
            case "music.search":
                keyword = call.arguments.get("keyword")
                version = call.arguments.get("version", "")
                language = call.arguments.get("language", "")
                if not isinstance(keyword, str) or not keyword.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "keyword is required")
                if not isinstance(version, str) or not isinstance(language, str):
                    return _channel_tool_error(
                        call, "INVALID_ARGUMENT", "search filters are invalid"
                    )
                try:
                    songs = await (
                        api.recommend_candidates()
                        if keyword.strip().lower() == "random"
                        else api.search_candidates(
                            keyword.strip(), version=version, language=language
                        )
                    )
                except (MusicProviderError, OSError, RuntimeError, ValueError):
                    logger.warning(
                        "music search failed turn_id=%s call_id=%s keyword=%s",
                        call.turn_id,
                        call.call_id,
                        keyword,
                    )
                    return _channel_tool_error(call, "MUSIC_SEARCH_FAILED", "music search failed")
                candidates = [
                    {
                        "song_id": song.song_id,
                        "title": song.title,
                        "artist": song.artist,
                    }
                    for song in songs
                ]
                return ChannelToolResult(
                    turn_id=call.turn_id,
                    call_id=call.call_id,
                    name=call.name,
                    ok=True,
                    content={"status": "completed", "songs": candidates},
                )
            case "music.play":
                song_id = call.arguments.get("song_id")
                playback_ref = call.arguments.get(
                    "playback_ref", f"music:{call.turn_id}:{call.call_id}"
                )
                if not isinstance(song_id, str) or not song_id.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "song_id is required")
                if not isinstance(playback_ref, str) or not playback_ref.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "playback_ref is invalid")
                receipt = await commands.play(
                    request_id=call.call_id,
                    playback_ref=playback_ref,
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
            if current is not None and current.playback_ref == call.arguments.get(
                "playback_ref", f"music:{call.turn_id}:{call.call_id}"
            ):
                await self.playback.stop(current.playback_ref)

    async def aclose(self) -> None:
        if self.playback is not None:
            await self.playback.aclose()
            self.playback = None
        if self._http is not None:
            await self._http.close()
            self._http = None
        self._api = None


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
