"""Provider-neutral music catalogue and source contracts."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MusicSource:
    song_id: str
    url: str
    expected_size: int | None = None
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class MusicSearchCandidate:
    song_id: str
    title: str
    artist: str


class MusicProviderError(RuntimeError):
    def __init__(self, code: int | str, message: str, *, source: str, endpoint: str):
        super().__init__(f"Music provider code={code}: {message}")
        self.code, self.source, self.endpoint = code, source, endpoint
        self.message = message


class MusicCatalogue(Protocol):
    async def search_candidates(
        self, keyword: str, *, version: str = "", language: str = ""
    ) -> list[MusicSearchCandidate]: ...
    async def recommend_candidates(self) -> list[MusicSearchCandidate]: ...
    async def source_for_song(self, song_id: str) -> MusicSource: ...
