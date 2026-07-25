"""Async client for the digital-human streaming interface ``/ws/audio_stream``.

The client streams 16 kHz mono float32 PCM up as the agent speaks and receives a
fragmented MP4 back in real time — first frames arrive one first-frame latency
after the first audio chunk, not after the whole utterance (the batch
``/api/stream_video`` path). Decoding the returned fMP4 is
:mod:`eidolon.livekit.avatar.progressive_decoder`'s job; this client only speaks
the WebSocket protocol (see ``docs/avatar/ws-audio-stream-api-spec.txt``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator

import aiohttp

logger = logging.getLogger("agent.avatar.streaming")

# Report a delivery gap this long or longer — under the smallest gap that can
# outlast the jitter buffer, so the log shows the profile, not just the failures.
_GAP_LOG_S = 0.4

# Once the input is complete, treat this much silence from the service as "the
# utterance is fully delivered". Comfortably above its normal inter-burst rhythm
# (measured 0.7–1.0 s) and far below its ~5 s input-idle timeout, which is the
# stall we are avoiding.
_TAIL_QUIET_S = 1.5


def _ws_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.startswith("https://"):
        b = "wss://" + b[len("https://") :]
    elif b.startswith("http://"):
        b = "ws://" + b[len("http://") :]
    return f"{b}/ws/audio_stream"


class DittoStreamSession:
    """One open ``/ws/audio_stream`` session: send audio, receive fMP4 segments."""

    def __init__(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        negotiated: dict,
    ) -> None:
        self._ws = ws
        self.negotiated = negotiated
        self._closed = False
        self._input_complete = False

    async def send_audio(self, pcm_f32le: bytes) -> None:
        if self._closed or not pcm_f32le:
            return
        await self._ws.send_bytes(pcm_f32le)

    def mark_input_complete(self) -> None:
        """Declare that no more audio is coming for this utterance.

        The service only finalizes after ~5 s of input silence, holding the last
        of the utterance until then; there is no "flush now" command (``stop``
        discards what it hasn't sent). So once the caller has fed its trailing
        silence, :meth:`segments` stops waiting as soon as the service goes quiet
        rather than sitting through that timeout.
        """
        self._input_complete = True

    async def request_stop(self) -> None:
        """Abort generation (barge-in). Discards anything not yet sent."""
        if self._closed:
            return
        try:
            await self._ws.send_str(json.dumps({"cmd": "stop"}))
        except Exception:
            logger.debug("[streaming] stop send failed", exc_info=True)

    async def segments(self) -> AsyncIterator[bytes]:
        """Yield fMP4 fragments as they arrive.

        Ends on the service's ``end``, or — once the caller has declared the
        input complete — as soon as the service goes quiet for longer than its
        normal inter-burst rhythm, which means the utterance has been fully
        delivered and only the idle timeout remains.
        """
        prev = time.monotonic()
        first = True
        while True:
            timeout = _TAIL_QUIET_S if self._input_complete else None
            try:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.info("[streaming] utterance delivered; not waiting out the idle timeout")
                return
            if msg.type == aiohttp.WSMsgType.BINARY:
                now = time.monotonic()
                gap = now - prev
                if first:
                    logger.info("[streaming] first segment after %.2fs", gap)
                    first = False
                elif gap > _GAP_LOG_S:
                    logger.warning("[streaming] source gap %.2fs between segments", gap)
                prev = now
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                kind = data.get("type")
                if kind == "end":
                    return
                if kind == "ping":
                    try:
                        await self._ws.send_str(json.dumps({"cmd": "pong"}))
                    except Exception:
                        pass
            else:
                return

    async def aclose(self) -> None:
        """Close this utterance's socket. The client's HTTP session lives on so
        the next utterance reuses the connection pool (no fresh TLS handshake)."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            logger.debug("[streaming] ws close failed", exc_info=True)


def encode_cond_image(image_bytes: bytes) -> str:
    """The ``cond_image_base64`` data URI for a face. Encode once and reuse: the
    face is fixed for a session while a new socket is opened per utterance."""
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")


class DittoStreamClient:
    """Opens :class:`DittoStreamSession`s against a digital-human service.

    One socket per utterance (so an idle conversation never holds the service,
    which handles a single session at a time), but one HTTP session for the
    client's lifetime, so consecutive utterances reuse the pooled TLS connection.
    """

    def __init__(
        self,
        base_url: str,
        *,
        verify_ssl: bool = False,
        connect_timeout_sec: float = 5.0,
        ready_timeout_sec: float = 10.0,
    ) -> None:
        self._url = _ws_url(base_url)
        self._verify_ssl = verify_ssl
        self._connect_timeout = connect_timeout_sec
        self._ready_timeout = ready_timeout_sec
        self._http: aiohttp.ClientSession | None = None

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def open(
        self,
        *,
        cond_image_b64: str | None,
        prefer_fps: float,
        screen_width: int,
        screen_height: int,
        opus_frame_ms: float = 20.0,
        jpeg_quality: int = 60,
        fast_start_samples: int = 0,
    ) -> DittoStreamSession:
        http = self._ensure_http()
        ws = await asyncio.wait_for(
            http.ws_connect(self._url, ssl=None if self._verify_ssl else False),
            timeout=self._connect_timeout,
        )

        start: dict = {
            "cmd": "start",
            "prefer_fps": prefer_fps,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "prefer_opus_frame_ms": opus_frame_ms,
            "jpeg_quality": jpeg_quality,
        }
        if fast_start_samples > 0:
            start["fast_start_samples"] = fast_start_samples
        if cond_image_b64:
            start["cond_image_base64"] = cond_image_b64
        try:
            await ws.send_str(json.dumps(start))
            negotiated = await asyncio.wait_for(
                self._await_ready(ws), timeout=self._ready_timeout
            )
        except Exception:
            await ws.close()
            raise
        return DittoStreamSession(ws, negotiated)

    async def aclose(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None

    async def _await_ready(self, ws: aiohttp.ClientWebSocketResponse) -> dict:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            status = data.get("status")
            if status == "ready":
                return data.get("negotiated", {}) or {}
            if status == "error":
                raise RuntimeError(f"digital-human service refused stream: {data!r}")
        raise RuntimeError("digital-human service closed before ready")
