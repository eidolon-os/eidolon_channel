"""Hermetic test for the streaming progressive decoder.

Synthesizes a fragmented MP4 (the container shape the ``/ws/audio_stream``
service emits), feeds it to ``progressive_decode`` in small chunks, and asserts
frames decode out — no real digital-human service involved.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from itertools import groupby

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


def _make_av_fragmented_mp4() -> bytes:
    """A streamable fMP4 with BOTH tracks, muxed track-grouped like the service:
    a run of video frames, then that span's audio."""
    buf = io.BytesIO()
    container = av.open(
        buf,
        mode="w",
        format="mp4",
        options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
    )
    try:
        vs = container.add_stream("libx264", rate=FPS)
        vs.width, vs.height, vs.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        aud = container.add_stream("aac", rate=24000)
        aud.layout = "mono"
        packets: list = []
        samples = 0
        for i in range(N_FRAMES):
            arr = np.full((HEIGHT, WIDTH, 3), (i * 17) % 256, dtype=np.uint8)
            packets += list(vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
            a = np.zeros((1, 1024), dtype=np.int16)
            af = av.AudioFrame.from_ndarray(a, format="s16", layout="mono")
            af.sample_rate = 24000
            af.pts = samples
            samples += 1024
            packets += list(aud.encode(af))
        packets += list(vs.encode()) + list(aud.encode())
        for p in packets:
            container.mux(p)
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


async def test_progressive_decode_interleaves_audio_and_video() -> None:
    """The service groups each fragment's packets by track (VVVVVVAAAAA). Emitting
    that order starves the synchronizer's audio source (choppy audio), so frames
    must come out interleaved by timestamp instead of in demux order."""
    body = _make_av_fragmented_mp4()
    kinds = [
        "A" if isinstance(f, rtc.AudioFrame) else "V"
        async for f in progressive_decode(_chunks(body), output_sample_rate=24000)
    ]
    assert "A" in kinds and "V" in kinds
    # Judge the steady state: at the very start one track necessarily leads (the
    # window has nothing from the other track to interleave with yet). What must
    # not happen is track-grouped output once both are flowing — with no window
    # the whole stream stays grouped, so this still catches the regression.
    steady = kinds[len(kinds) // 2 :]
    longest = max(len(list(g)) for _, g in groupby(steady))
    assert longest <= 3, f"un-interleaved run of {longest}: {''.join(steady[:40])}"


async def test_progressive_decode_empty_stream_is_clean() -> None:
    async def _empty() -> AsyncIterator[bytes]:
        return
        yield  # pragma: no cover

    frames = [f async for f in progressive_decode(_empty())]
    assert frames == []
