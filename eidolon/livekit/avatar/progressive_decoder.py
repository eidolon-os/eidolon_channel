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
from collections.abc import AsyncIterator

import av
from livekit import rtc

from .decoder import _audio_to_rtc, _video_to_rtc

logger = logging.getLogger("agent.avatar.progressive_decoder")


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
    """Blocking PyAV decode loop (worker thread): push rtc frames to ``out_queue``."""
    container = None
    try:
        container = av.open(reader, mode="r", format="mp4")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=output_sample_rate)
        for frame in container.decode():
            if isinstance(frame, av.VideoFrame):
                loop.call_soon_threadsafe(out_queue.put_nowait, _video_to_rtc(frame))
            elif isinstance(frame, av.AudioFrame):
                for rf in resampler.resample(frame):
                    loop.call_soon_threadsafe(
                        out_queue.put_nowait, _audio_to_rtc(rf, output_sample_rate)
                    )
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
) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame]:
    """Yield rtc frames as ``segments`` (fMP4 fragments) arrive.

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
    try:
        while True:
            item = await out_queue.get()
            if item is sentinel:
                break
            yield item  # type: ignore[misc]
    finally:
        reader.close()
        feed_task.cancel()
        await asyncio.gather(feed_task, decode_task, return_exceptions=True)
