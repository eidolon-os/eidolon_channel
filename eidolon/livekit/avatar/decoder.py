"""Decode the digital-human service's streamed container into rtc frames.

The service returns a fragmented MP4 / MPEG-TS body (H.264 Baseline + AAC-LC).
LiveKit's :class:`AvatarRunner` wants raw ``rtc.VideoFrame`` (I420) and
``rtc.AudioFrame`` (int16 PCM), fed through an :class:`rtc.AVSynchronizer` that
paces video at a **fixed** ``video_fps`` and plays audio in real time.

Two Phase-0 facts shape this module:

1. Decode is ~200x realtime, so buffering one segment and decoding off-thread is
   cheap and keeps the code deadlock-free. (A progressive decoder can drop in
   later behind the same async-iterator interface.)
2. The service's *actual* emitted frame count does not match its declared fps and
   varies with clip duration (~15.5–18 fps for a requested 25). Feeding those
   frames to a fixed-fps synchronizer would drift lip-sync. So we **retime** the
   decoded video to a constant ``target_fps`` against the authoritative audio
   duration (nearest source frame per output slot) and interleave audio+video by
   presentation time. Video then spans exactly the audio duration → stays synced
   regardless of the source's variable/nominal rate.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator

import av
import numpy as np
from livekit import rtc

logger = logging.getLogger("agent.avatar.decoder")


def _video_to_rtc(frame: "av.VideoFrame") -> rtc.VideoFrame:
    # yuv420p to_ndarray is contiguous (h*3//2, w) with no stride padding →
    # tobytes() is a tightly-packed I420 buffer (Y + U + V), exactly what
    # rtc.VideoFrame(type=I420) expects (verified: bytes == w*h*3//2).
    arr = frame.to_ndarray(format="yuv420p")
    return rtc.VideoFrame(
        width=frame.width,
        height=frame.height,
        type=rtc.VideoBufferType.I420,
        data=arr.tobytes(),
    )


def _audio_to_rtc(frame: "av.AudioFrame", sample_rate: int) -> rtc.AudioFrame:
    arr = frame.to_ndarray()  # s16, shape (1, samples) after resampling to mono
    samples = int(arr.shape[-1])
    return rtc.AudioFrame(
        data=arr.astype(np.int16).tobytes(),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def decode_container(
    body: bytes,
    *,
    audio_sample_rate: int = 24000,
    target_fps: float = 25.0,
) -> list[rtc.VideoFrame | rtc.AudioFrame]:
    """Decode a container blob → time-ordered, retimed rtc frames (blocking).

    Returns audio frames (int16 mono @ ``audio_sample_rate``) and video frames
    (I420) interleaved by presentation time, with video resampled to a constant
    ``target_fps`` spanning the audio duration. Runs in a worker thread via
    :func:`decode_stream`; also directly unit-testable.
    """
    container = av.open(io.BytesIO(body), mode="r")
    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=audio_sample_rate)
        vids: list[tuple[float, rtc.VideoFrame]] = []
        auds: list[tuple[float, rtc.AudioFrame]] = []
        total_samples = 0
        for frame in container.decode():
            t = float(frame.time) if frame.time is not None else None
            if isinstance(frame, av.VideoFrame):
                vids.append((t if t is not None else len(vids), _video_to_rtc(frame)))
            elif isinstance(frame, av.AudioFrame):
                for rf in resampler.resample(frame):
                    at = float(rf.time) if rf.time is not None else (total_samples / audio_sample_rate)
                    rtc_a = _audio_to_rtc(rf, audio_sample_rate)
                    total_samples += rtc_a.samples_per_channel
                    auds.append((at, rtc_a))
    finally:
        try:
            container.close()
        except Exception:
            pass

    if not vids:
        return [a for _, a in auds]

    duration = (total_samples / audio_sample_rate) if total_samples else (vids[-1][0] + 1.0 / target_fps)
    vids.sort(key=lambda x: x[0])
    v_times = [t for t, _ in vids]

    # Retime video: one output frame per 1/target_fps slot across `duration`,
    # each mapped to the nearest source frame by presentation time. Duplicating a
    # frame just reuses the (immutable) rtc.VideoFrame — cheap.
    import bisect

    n_out = max(1, round(target_fps * duration))
    retimed: list[tuple[float, rtc.VideoFrame]] = []
    for j in range(n_out):
        t = (j + 0.5) / target_fps
        idx = bisect.bisect_left(v_times, t)
        if idx <= 0:
            src = 0
        elif idx >= len(vids):
            src = len(vids) - 1
        else:
            src = idx if (v_times[idx] - t) < (t - v_times[idx - 1]) else idx - 1
        retimed.append((t, vids[src][1]))

    # Merge audio + retimed video by time (audio first on ties so it leads).
    merged: list[tuple[float, int, rtc.VideoFrame | rtc.AudioFrame]] = []
    for t, a in auds:
        merged.append((t, 0, a))
    for t, v in retimed:
        merged.append((t, 1, v))
    merged.sort(key=lambda x: (x[0], x[1]))
    return [f for _, _, f in merged]


async def decode_stream(
    chunks: AsyncIterator[bytes],
    *,
    audio_sample_rate: int = 24000,
    target_fps: float = 25.0,
) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame]:
    """Buffer one segment's response, decode+retime off-thread, yield in order.

    Cancelling during download aborts the HTTP stream (aiohttp closes the
    connection); cancelling during the short decode waits for it.
    """
    body = bytearray()
    async for chunk in chunks:
        body += chunk
    if not body:
        logger.warning("[decoder] empty response body; nothing to decode")
        return
    frames = await asyncio.to_thread(
        decode_container, bytes(body), audio_sample_rate=audio_sample_rate, target_fps=target_fps
    )
    for f in frames:
        yield f
