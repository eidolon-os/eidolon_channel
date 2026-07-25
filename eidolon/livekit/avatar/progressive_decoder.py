"""Progressive decoder for the digital-human service's *streaming* fMP4.

The batch decoder (:mod:`eidolon.livekit.avatar.decoder`) buffers a whole segment
and retimes video against the total audio duration — it cannot emit a frame until
the utterance is complete. The streaming interface (``/ws/audio_stream``) instead
emits a fragmented MP4 continuously, with correct PTS at a consistent fps (the
service aligns fps to the OPUS frame length), so we can decode fragments **as they
arrive** and hand frames to LiveKit's ``AVSynchronizer`` the moment they exist —
the face starts within one first-frame latency of the first audio chunk. No manual
retiming: we trust the container's PTS/fps.

PyAV demux/decode is blocking, so it runs in a worker thread reading from a
blocking byte reader we feed from the async segment stream; decoded ``rtc`` frames
cross back to the event loop through an ``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from bisect import insort
from collections.abc import AsyncIterator

import av
from livekit import rtc

from .decoder import _audio_to_rtc, _video_to_rtc

logger = logging.getLogger("agent.avatar.progressive_decoder")

# How long to hold decoded frames before emitting, so audio and video can be
# re-interleaved by timestamp. This is the one place the avatar path adds latency
# of its own, so it is tuned to the smallest value that still interleaves: replayed
# against live output, the longest single-track run is 6 at 0 ms, 4 at 100 ms,
# 3 at 150 ms, and 2 from 200 ms on — going beyond 200 ms buys nothing.
INTERLEAVE_WINDOW_S = 0.2

# If nothing new arrives for this long, release whatever the interleave window is
# holding: the service renders in bursts, and a held frame during a burst gap is
# a frame the consumer is starving for.
IDLE_FLUSH_S = 0.1


class _BlockingStreamReader:
    """A file-like whose ``read`` blocks until fed bytes arrive or EOF.

    Fed from the async side (:meth:`feed` / :meth:`feed_eof`), read from the PyAV
    decode thread. A streamable fMP4 (``empty_moov+frag``) needs only sequential
    reads. :meth:`close` unblocks a waiting read so barge-in can abort promptly.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._eof = False
        self._closed = False
        self._cond = threading.Condition()

    def feed(self, data: bytes) -> None:
        if not data:
            return
        with self._cond:
            self._buf += data
            self._cond.notify_all()

    def feed_eof(self) -> None:
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def read(self, n: int = -1) -> bytes:
        with self._cond:
            while True:
                if self._closed:
                    return b""
                if n is None or n < 0:
                    if self._eof:
                        out = bytes(self._buf)
                        self._buf.clear()
                        return out
                elif len(self._buf) >= n:
                    out = bytes(self._buf[:n])
                    del self._buf[:n]
                    return out
                elif self._eof:
                    out = bytes(self._buf)
                    self._buf.clear()
                    return out
                self._cond.wait()


def _decode_into(
    reader: _BlockingStreamReader,
    loop: asyncio.AbstractEventLoop,
    out_queue: "asyncio.Queue[object]",
    sentinel: object,
    *,
    output_sample_rate: int,
) -> None:
    """Blocking PyAV decode loop (worker thread).

    Pushes ``(presentation_time, kind, rtc_frame)`` — the timestamp lets the
    consumer re-interleave audio and video, which the demux order does not do
    (see :func:`progressive_decode`). ``kind`` 0 = audio, 1 = video, so audio
    leads on ties (same rule as the batch decoder).
    """
    container = None
    try:
        container = av.open(reader, mode="r", format="mp4")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=output_sample_rate)
        audio_samples = 0
        for frame in container.decode():
            if isinstance(frame, av.VideoFrame):
                t = float(frame.time) if frame.time is not None else 0.0
                loop.call_soon_threadsafe(
                    out_queue.put_nowait, (t, 1, _video_to_rtc(frame))
                )
            elif isinstance(frame, av.AudioFrame):
                for rf in resampler.resample(frame):
                    rtc_a = _audio_to_rtc(rf, output_sample_rate)
                    t = (
                        float(rf.time)
                        if rf.time is not None
                        else audio_samples / output_sample_rate
                    )
                    audio_samples += rtc_a.samples_per_channel
                    loop.call_soon_threadsafe(out_queue.put_nowait, (t, 0, rtc_a))
    except Exception:
        # A closed reader (barge-in) or a truncated stream surfaces here; the
        # sentinel below still ends the async iterator cleanly.
        logger.debug("[progressive_decoder] decode loop ended", exc_info=True)
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        loop.call_soon_threadsafe(out_queue.put_nowait, sentinel)


async def progressive_decode(
    segments: AsyncIterator[bytes],
    *,
    output_sample_rate: int = 24000,
    interleave_window_s: float = INTERLEAVE_WINDOW_S,
) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame]:
    """Yield rtc frames, re-interleaved by presentation time, as fragments arrive.

    The service emits each fragment's packets grouped by track (``VVVVVVAAAAA``),
    not interleaved by time. Forwarding that order starves the synchronizer: a
    run of video frames fills its small video queue and blocks, so no audio is
    captured meanwhile and the audio source underruns — choppy audio, and
    "frame capture was behind schedule" warnings. So we hold a short window and
    emit in timestamp order (audio first on ties, like the batch decoder), which
    restores the natural per-frame backpressure. The window is the only added
    latency, and it must exceed one fragment's span to fully interleave it.

    Cancelling the iterator (barge-in) unblocks the decode thread and stops the
    feed promptly.
    """
    reader = _BlockingStreamReader()
    loop = asyncio.get_running_loop()
    out_queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    async def _feed() -> None:
        try:
            async for seg in segments:
                reader.feed(seg)
        finally:
            reader.feed_eof()

    feed_task = asyncio.create_task(_feed())
    decode_task = asyncio.create_task(
        asyncio.to_thread(
            _decode_into,
            reader,
            loop,
            out_queue,
            sentinel,
            output_sample_rate=output_sample_rate,
        )
    )
    # (presentation_time, kind, seq, frame) held until the window has passed.
    # ``kind`` puts audio before video at the same timestamp; ``seq`` keeps ties
    # deterministic and stops the sort from ever comparing frames themselves.
    pending: list[tuple[float, int, int, rtc.VideoFrame | rtc.AudioFrame]] = []
    seq = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(out_queue.get(), timeout=IDLE_FLUSH_S)
            except asyncio.TimeoutError:
                # The service went quiet mid-turn (it renders in bursts). Nothing
                # more will arrive to interleave with, and these are exactly the
                # frames the consumer needs to ride out the gap — release them
                # instead of holding them for a timestamp that may never come.
                while pending:
                    yield pending.pop(0)[3]
                continue
            if item is sentinel:
                break
            time_s, kind, frame = item  # type: ignore[misc]
            insort(pending, (time_s, kind, seq, frame))
            seq += 1
            cutoff = pending[-1][0] - interleave_window_s
            while pending and pending[0][0] <= cutoff:
                yield pending.pop(0)[3]
        for _, _, _, frame in pending:  # drain what the window still holds
            yield frame
    finally:
        reader.close()
        feed_task.cancel()
        await asyncio.gather(feed_task, decode_task, return_exceptions=True)
