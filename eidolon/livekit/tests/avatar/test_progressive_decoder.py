"""Hermetic test for the streaming progressive decoder.

Synthesizes a fragmented MP4 (the container shape the ``/ws/audio_stream``
service emits), feeds it to ``progressive_decode`` in small chunks, and asserts
frames decode out — no real digital-human service involved.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import av
import numpy as np
from livekit import rtc

from eidolon.livekit.avatar.progressive_decoder import progressive_decode

WIDTH, HEIGHT, FPS, N_FRAMES = 64, 48, 25, 12


def _make_fragmented_mp4() -> bytes:
    """A streamable fMP4 (empty_moov+frag) with N_FRAMES of H.264 video."""
    buf = io.BytesIO()
    container = av.open(
        buf,
        mode="w",
        format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
    )
    try:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        for i in range(N_FRAMES):
            arr = np.full((HEIGHT, WIDTH, 3), (i * 17) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():  # flush
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


async def _chunks(data: bytes, size: int = 2048) -> AsyncIterator[bytes]:
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def test_progressive_decode_yields_video_frames() -> None:
    body = _make_fragmented_mp4()
    frames = [
        f
        async for f in progressive_decode(_chunks(body), output_sample_rate=24000)
    ]
    video = [f for f in frames if isinstance(f, rtc.VideoFrame)]
    assert len(video) == N_FRAMES
    assert all(f.width == WIDTH and f.height == HEIGHT for f in video)
    assert all(f.type == rtc.VideoBufferType.I420 for f in video)


async def test_progressive_decode_aborts_without_hanging() -> None:
    """Barge-in: closing the iterator after one frame must not deadlock the
    decode thread / feed task."""
    body = _make_fragmented_mp4()
    agen = progressive_decode(_chunks(body), output_sample_rate=24000)
    first = await agen.__anext__()
    assert isinstance(first, rtc.VideoFrame)
    await agen.aclose()  # returns → reader.close() unblocked the decode thread


async def test_progressive_decode_empty_stream_is_clean() -> None:
    async def _empty() -> AsyncIterator[bytes]:
        return
        yield  # pragma: no cover

    frames = [f async for f in progressive_decode(_empty())]
    assert frames == []
