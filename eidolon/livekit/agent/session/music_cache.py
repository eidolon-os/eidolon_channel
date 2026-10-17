from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from .music_trace import current_music_trace, redact_music_url

logger = logging.getLogger("agent.session.music")


class MusicDownloadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MusicSource:
    song_id: str
    url: str
    expected_size: int | None = None
    expected_sha256: str | None = None


MusicSourceResolver = Callable[[str], Awaitable[MusicSource]]


class MusicCache:
    def __init__(
        self,
        music_dir: Path,
        *,
        source_resolver: MusicSourceResolver,
        http_session: aiohttp.ClientSession,
        max_file_bytes: int = 50 * 1024 * 1024,
        chunk_size: int = 64 * 1024,
    ) -> None:
        self._music_dir = music_dir.resolve(strict=True)
        self._source_resolver = source_resolver
        self._http_session = http_session
        self._max_file_bytes = max_file_bytes
        self._chunk_size = chunk_size

    async def get(self, song_id: str) -> Path:
        target = self._target(song_id)
        if target.is_file():
            logger.info("music cache hit song_id=%s path=%s %s", song_id, target, _trace_fields())
            return target

        source = await self._source_resolver(song_id)
        logger.info(
            "music cache source resolved song_id=%s url=%s %s",
            song_id,
            redact_music_url(source.url),
            _trace_fields(),
        )
        self._validate_source(source)
        if source.song_id != song_id:
            raise MusicDownloadError("music source identity does not match request")

        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
        try:
            await self._download(source, temporary)
            os.replace(temporary, target)
            logger.info("music cache download completed song_id=%s path=%s %s", song_id, target, _trace_fields())
            return target
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _target(self, song_id: str) -> Path:
        digest = hashlib.sha256(song_id.encode("utf-8")).hexdigest()
        return self._music_dir / f"{digest}.mp3"

    def _validate_source(self, source: MusicSource) -> None:
        parsed = urlparse(source.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise MusicDownloadError("music source must be an HTTP(S) URL")
        if source.expected_size is not None and source.expected_size > self._max_file_bytes:
            raise MusicDownloadError("music source exceeds configured size limit")
        if source.expected_sha256 is not None and len(source.expected_sha256) != 64:
            raise MusicDownloadError("music source digest must be SHA-256")

    async def _download(self, source: MusicSource, temporary: Path) -> None:
        digest = hashlib.sha256()
        size = 0
        async with self._http_session.get(
            source.url,
            allow_redirects=False,
            headers={"Accept": "audio/mpeg"},
        ) as response:
            logger.info(
                "music source response song_id=%s url=%s http_status=%s content_type=%s %s",
                source.song_id,
                redact_music_url(str(response.url)),
                response.status,
                response.headers.get("Content-Type", "-"),
                _trace_fields(),
            )
            if response.status != 200:
                raise MusicDownloadError(f"music source returned HTTP {response.status}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > self._max_file_bytes:
                raise MusicDownloadError("music source exceeds configured size limit")
            with temporary.open("wb") as output:
                async for chunk in self._chunks(response.content):
                    size += len(chunk)
                    if size > self._max_file_bytes:
                        raise MusicDownloadError("music source exceeds configured size limit")
                    output.write(chunk)
                    digest.update(chunk)

        if source.expected_size is not None and size != source.expected_size:
            raise MusicDownloadError("music source size does not match metadata")
        if source.expected_sha256 is not None and digest.hexdigest() != source.expected_sha256:
            raise MusicDownloadError("music source digest does not match metadata")
        with temporary.open("rb") as input_file:
            if input_file.read(3) != b"ID3":
                raise MusicDownloadError("music source is not an MP3 file")

    async def _chunks(self, content: aiohttp.StreamReader) -> AsyncIterator[bytes]:
        async for chunk in content.iter_chunked(self._chunk_size):
            yield chunk


def _trace_fields() -> str:
    trace = current_music_trace()
    if trace is None:
        return "turn_id=- call_id=- request_id=- playback_ref=-"
    return (
        f"turn_id={trace.turn_id} call_id={trace.call_id} "
        f"request_id={trace.request_id} playback_ref={trace.playback_ref}"
    )
