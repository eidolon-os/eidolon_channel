"""Bounded catalogue search policy owned by the music capability, not the OS."""
from __future__ import annotations

import asyncio
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass

from ...common.music import MusicCatalogue, MusicSearchCandidate


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class MusicSearchResult:
    songs: tuple[MusicSearchCandidate, ...]
    queries: tuple[str, ...]
    title: str
    artist: str

    def content(self) -> dict[str, object]:
        def matched(song: MusicSearchCandidate) -> bool:
            return bool(self.title or self.artist) and (
                not self.title or _normalized(song.title) == _normalized(self.title)
            ) and (not self.artist or _normalized(song.artist) == _normalized(self.artist))

        ordered = sorted(self.songs, key=lambda song: not matched(song))
        return {
            "status": "completed", "scope": "returned_candidates", "exhaustive": False,
            "queries": list(self.queries),
            "songs": [
                {"song_id": song.song_id, "title": song.title, "artist": song.artist,
                 "match": "exact" if matched(song) else "unverified"}
                for song in ordered
            ],
        }


class MusicSearchService:
    """No dialogue decisions or automatic substitution; results remain candidates."""

    def __init__(self, catalogue: MusicCatalogue) -> None:
        self._catalogue = catalogue
        self._cache: OrderedDict[tuple[str, ...], tuple[float, MusicSearchResult]] = OrderedDict()

    async def search(
        self, *, keyword: str = "", title: str = "", artist: str = "",
        version: str = "", language: str = "",
    ) -> MusicSearchResult:
        key = (keyword, title, artist, version, language)
        cached = self._cache.get(key)
        if cached is not None and time.monotonic() - cached[0] < 60:
            self._cache.move_to_end(key)
            return cached[1]
        queries = list(dict.fromkeys(q for q in (
            " ".join(v for v in (artist, title) if v) or keyword, title, artist,
        ) if q))
        songs: dict[str, MusicSearchCandidate] = {}
        executed: list[str] = []
        # Keep the whole search below the 15s tool budget, including fallback.
        async with asyncio.timeout(12):
            for query in queries:
                executed.append(query)
                candidates = await self._catalogue.search_candidates(
                    query, version=version, language=language,
                )
                for song in candidates:
                    songs.setdefault(song.song_id, song)
                if candidates and not (title or artist):
                    break
                if any(
                    (not title or _normalized(s.title) == _normalized(title))
                    and (not artist or _normalized(s.artist) == _normalized(artist))
                    for s in candidates
                ):
                    break
        result = MusicSearchResult(tuple(songs.values())[:30], tuple(executed), title, artist)
        self._cache[key] = (time.monotonic(), result)
        self._cache.move_to_end(key)
        while len(self._cache) > 64:
            self._cache.popitem(last=False)
        return result
