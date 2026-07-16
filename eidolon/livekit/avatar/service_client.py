"""Async HTTP client for the digital-human (talking-head) video service.

The service is audio-driven: you POST a complete audio clip (plus an optional
face image) and it streams back a lip-synced, fragmented MP4 / MPEG-TS response
(H.264 Baseline + AAC-LC). See ``eidolon/livekit/avatar/README.md`` for the
measured contract (Phase 0). This client only speaks HTTP; decoding the returned
container into raw frames is :mod:`eidolon.livekit.avatar.decoder`'s job.

The service takes a *whole* audio segment per request (batch-per-utterance), but
its response is progressively chunked — first bytes arrive ~150 ms after POST —
so callers should consume :meth:`DigitalHumanServiceClient.stream` as an async
iterator and start decoding immediately rather than buffering the whole body.
"""

from __future__ import annotations

import io
import json
import logging
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("agent.avatar.service")


@dataclass(frozen=True)
class StreamVideoParams:
    """Per-request shaping for ``POST /api/stream_video``.

    ``device_info`` mirrors the fields the service's own test page sends; the
    resolution / fps knobs are honored server-side (Phase 0: requesting
    ``512x512`` yields 448x448, ``prefer_fps=25`` yields 25 fps).
    """

    width: int = 448
    height: int = 448
    fps: float = 25.0
    fmt: str = "mp4"  # "mp4" (fragmented) or "ts"
    jpeg_quality: int = 80

    def device_info(self) -> str:
        return json.dumps(
            {
                "prefer_fps": self.fps,
                "format": self.fmt,
                "jpeg_quality": self.jpeg_quality,
                "max_video_resolution": f"{self.width}x{self.height}",
                "screen_width": self.width,
                "screen_height": self.height,
            },
            separators=(",", ":"),
        )


def pcm16_to_wav(pcm: bytes, *, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw int16 little-endian PCM in a minimal WAV container.

    The service accepts a standard audio file; TTS reaches us as raw PCM frames
    (via LiveKit ``DataStream``), so we re-wrap one segment's worth of PCM into a
    WAV blob before POSTing.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class DigitalHumanServiceClient:
    """Thin async wrapper over the digital-human service's streaming endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        params: StreamVideoParams | None = None,
        request_timeout_sec: float = 30.0,
        connect_timeout_sec: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._params = params or StreamVideoParams()
        self._timeout = aiohttp.ClientTimeout(
            total=None,  # streaming body may outlast `total`; bound the phases instead
            sock_connect=connect_timeout_sec,
            sock_read=request_timeout_sec,
        )
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def stream_video(
        self,
        *,
        wav_bytes: bytes,
        image_bytes: bytes | None = None,
        chunk_size: int = 16 * 1024,
    ) -> AsyncIterator[bytes]:
        """POST one audio segment and yield the response body chunks as they arrive.

        Cancelling the async iterator (e.g. on barge-in) aborts the in-flight
        HTTP request — aiohttp closes the connection when the response context
        exits, so the server stops generating.
        """
        session = await self._ensure_session()
        form = aiohttp.FormData()
        form.add_field("audio", wav_bytes, filename="segment.wav", content_type="audio/wav")
        if image_bytes is not None:
            form.add_field(
                "image", image_bytes, filename="face.jpg", content_type="image/jpeg"
            )
        form.add_field("device_info", self._params.device_info())
        form.add_field("format", self._params.fmt)

        url = f"{self._base_url}/api/stream_video"
        async with session.post(url, data=form) as resp:
            if resp.status != 200:
                body = (await resp.read())[:512]
                raise RuntimeError(
                    f"digital-human service returned HTTP {resp.status}: {body!r}"
                )
            async for chunk in resp.content.iter_chunked(chunk_size):
                if chunk:
                    yield chunk

    async def default_avatar_jpeg(self) -> bytes:
        """Fetch the service's default face image (used for the idle frame)."""
        session = await self._ensure_session()
        async with session.get(f"{self._base_url}/api/default_avatar") as resp:
            resp.raise_for_status()
            return await resp.read()
