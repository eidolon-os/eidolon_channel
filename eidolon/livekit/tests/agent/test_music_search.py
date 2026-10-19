from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from eidolon.livekit.common.music import MusicSearchCandidate
from eidolon.livekit.plugins.music.search import MusicSearchService


@pytest.mark.asyncio
async def test_search_fallback_preserves_constraints_and_does_not_substitute():
    alternative = MusicSearchCandidate("other", "Other work", "Artist")
    exact = MusicSearchCandidate("chosen", "Requested work", "Artist")
    api = SimpleNamespace(search_candidates=AsyncMock(side_effect=[[], [alternative], [exact]]))
    service = MusicSearchService(api)
    result = await service.search(title="Requested work", artist="Artist", version="live", language="en")
    content = result.content()
    assert api.search_candidates.await_count == 3
    assert all(c.kwargs == {"version": "live", "language": "en"} for c in api.search_candidates.await_args_list)
    assert content["songs"][0] == {"song_id": "chosen", "title": "Requested work", "artist": "Artist", "match": "exact"}
    assert content["songs"][1]["match"] == "unverified"
    assert content["exhaustive"] is False
    assert await service.search(title="Requested work", artist="Artist", version="live", language="en") == result
    assert api.search_candidates.await_count == 3


@pytest.mark.asyncio
async def test_exact_match_ends_search_and_empty_is_not_global_absence():
    api = SimpleNamespace(search_candidates=AsyncMock(return_value=[MusicSearchCandidate("id", "Title", "Artist")]))
    service = MusicSearchService(api)
    assert len((await service.search(title="Title", artist="Artist")).songs) == 1
    assert api.search_candidates.await_count == 1
    api.search_candidates.return_value = []
    result = await service.search(keyword="unavailable")
    assert result.content()["songs"] == []
    assert result.content()["scope"] == "returned_candidates"
    assert result.content()["exhaustive"] is False
