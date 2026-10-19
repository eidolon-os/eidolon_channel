"""Per-session music tool orchestration; provider construction is a startup concern."""

from __future__ import annotations
import logging
import aiohttp
from pathlib import Path
from ...common.music import MusicCatalogue, MusicProviderError
from ...plugins.music.search import MusicSearchService, MusicSearchResult
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
        self._search: MusicSearchService | None = None
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
                fields = {key: call.arguments.get(key, "") for key in (
                    "keyword", "title", "artist", "version", "language"
                )}
                if not all(isinstance(value, str) for value in fields.values()):
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "search fields must be strings")
                fields = {key: value.strip() for key, value in fields.items()}
                if not any(fields[key] for key in ("keyword", "title", "artist")):
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "a search query is required")
                try:
                    if fields["keyword"].lower() == "random" and not (fields["title"] or fields["artist"]):
                        result = MusicSearchResult(tuple(await api.recommend_candidates()), ("random",), "", "")
                    else:
                        if self._search is None:
                            self._search = MusicSearchService(api)
                        result = await self._search.search(**fields)
                except (MusicProviderError, OSError, RuntimeError, ValueError):
                    logger.warning("music search failed turn_id=%s call_id=%s", call.turn_id, call.call_id)
                    return _channel_tool_error(call, "MUSIC_SEARCH_FAILED", "music search failed")
                return ChannelToolResult(
                    turn_id=call.turn_id, call_id=call.call_id, name=call.name,
                    ok=True, content=result.content(), outcome_state="completed",
                    reference_context={"candidates": result.content()["songs"]},
                )
            case "music.play":
                song_id = call.arguments.get("song_id")
                # Instance identity belongs to the capability, not model arguments.
                # A new invocation gets a new instance; a replay of that invocation
                # keeps its identity. Agent dispatcher owns command deduplication.
                playback_ref = f"music:{call.turn_id}:{call.call_id}"
                if not isinstance(song_id, str) or not song_id.strip():
                    return _channel_tool_error(call, "INVALID_ARGUMENT", "song_id is required")
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
        # Invocation success and the resource lifecycle are independent facts.
        # An empty status read is successful; stopping an absent current player is
        # idempotent, but a mismatched explicit reference must not claim success.
        if call.name == "music.status":
            success = True
            outcome_state = "completed"
        elif call.name == "music.stop":
            success = receipt.status.value != "failed" and not (
                receipt.status.value == "not_found" and call.arguments.get("playback_ref")
            )
            outcome_state = "completed" if success else "failed"
        else:
            success = receipt.status.value in {"preparing", "accepted", "playing", "completed"}
            outcome_state = (
                "failed" if not success else
                "completed" if receipt.status.value == "completed" else "accepted"
            )
        return ChannelToolResult(
            turn_id=call.turn_id,
            call_id=call.call_id,
            name=call.name,
            ok=success,
            content=content,
            error_code="MUSIC_COMMAND_FAILED" if not success else "",
            error_message=(receipt.error or "") if not success else "",
            outcome_state=outcome_state,
            reference_context={"playback_ref": receipt.playback_ref} if receipt.playback_ref else {},
        )

    async def cancel_channel_tool(self, call: ChannelToolCall) -> None:
        if call.name == "music.play" and self.playback is not None:
            current = await self.playback.get_status()
            if current is not None and current.playback_ref == f"music:{call.turn_id}:{call.call_id}":
                await self.playback.stop(current.playback_ref)

    async def aclose(self) -> None:
        if self.playback is not None:
            await self.playback.aclose()
            self.playback = None
        if self._http is not None:
            await self._http.close()
            self._http = None
        self._api = None
        self._search = None


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
