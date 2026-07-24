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
from collections.abc import AsyncIterator

import aiohttp

logger = logging.getLogger("agent.avatar.streaming")


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
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        negotiated: dict,
    ) -> None:
        self._session = session
        self._ws = ws
        self.negotiated = negotiated
        self._closed = False

    async def send_audio(self, pcm_f32le: bytes) -> None:
        if self._closed or not pcm_f32le:
            return
        await self._ws.send_bytes(pcm_f32le)

    async def request_stop(self) -> None:
        """Signal end-of-audio so the service flushes and sends ``end``."""
        if self._closed:
            return
        try:
            await self._ws.send_str(json.dumps({"cmd": "stop"}))
        except Exception:
            logger.debug("[streaming] stop send failed", exc_info=True)

    async def segments(self) -> AsyncIterator[bytes]:
        """Yield fMP4 fragments as they arrive; stop on the ``end`` message."""
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                kind = data.get("type")
                if kind == "end":
                    break
                if kind == "ping":
                    try:
                        await self._ws.send_str(json.dumps({"cmd": "pong"}))
                    except Exception:
                        pass
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                break

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            logger.debug("[streaming] ws close failed", exc_info=True)
        finally:
            await self._session.close()


class DittoStreamClient:
    """Opens :class:`DittoStreamSession`s against a digital-human service."""

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

    async def open(
        self,
        *,
        image_bytes: bytes | None,
        prefer_fps: float,
        screen_width: int,
        screen_height: int,
        opus_frame_ms: float = 20.0,
        jpeg_quality: int = 60,
        fast_start_samples: int = 0,
    ) -> DittoStreamSession:
        session = aiohttp.ClientSession()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(self._url, ssl=None if self._verify_ssl else False),
                timeout=self._connect_timeout,
            )
        except Exception:
            await session.close()
            raise

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
        if image_bytes:
            start["cond_image_base64"] = (
                "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
            )
        try:
            await ws.send_str(json.dumps(start))
            negotiated = await asyncio.wait_for(
                self._await_ready(ws), timeout=self._ready_timeout
            )
        except Exception:
            await ws.close()
            await session.close()
            raise
        return DittoStreamSession(session, ws, negotiated)

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
