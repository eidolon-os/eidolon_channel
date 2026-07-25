"""Shared output machinery for the digital-human ``VideoGenerator`` variants.

Both the batch (``/api/stream_video``) and streaming (``/ws/audio_stream``)
generators feed LiveKit's ``AvatarRunner`` identically: an output queue of rtc
frames that ``__aiter__`` drains, emitting a still *idle frame* whenever no
generated frame is ready so the published track never stalls. Only the
audio-input + service-call halves differ; this base owns the common output side.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time

import av
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

logger = logging.getLogger("agent.avatar.video_gen")

# Seconds of media to hold between the service and the synchronizer. Must exceed
# the service's inter-burst gap (measured ~0.8 s) with headroom; at 0.83 s of
# buffer the queue drained to empty ~half the time and speech stuttered.
BUFFER_SECONDS = 2.5


def image_to_i420(jpeg_or_png: bytes, width: int, height: int) -> rtc.VideoFrame:
    """Decode a still image and scale/convert it to an I420 ``rtc.VideoFrame``."""
    container = av.open(io.BytesIO(jpeg_or_png))
    try:
        frame = next(container.decode(video=0))
    finally:
        try:
            container.close()
        except Exception:
            pass
    frame = frame.reformat(width=width, height=height, format="yuv420p")
    return rtc.VideoFrame(
        width=width,
        height=height,
        type=rtc.VideoBufferType.I420,
        data=frame.to_ndarray(format="yuv420p").tobytes(),
    )


def drain(q: asyncio.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break


class DHVideoGeneratorBase(VideoGenerator):
    """Common output/idle side of the talking-head ``VideoGenerator``.

    Subclasses fill ``_out_queue`` with ``rtc`` frames + ``AudioSegmentEnd`` and
    implement ``push_audio`` / ``clear_buffer`` / ``warmup`` / ``aclose``.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        target_fps: float,
        output_sample_rate: int,
        face_image: bytes | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._out_sr = output_sample_rate
        self._face_image = face_image
        # Jitter buffer between the service and the synchronizer. The service
        # generates in bursts — measured: ~1.4 s of media delivered at once, then
        # ~0.8 s of silence while the next batch renders — and the synchronizer
        # holds only ~100 ms itself, so this queue is what bridges those gaps.
        # Entries are mixed audio+video, so a frame is worth roughly
        # 1/(fps * 1.65) s of media; BUFFER_SECONDS of headroom over the observed
        # gap keeps audio from underrunning (a starved queue = an audible stall).
        self._out_queue: asyncio.Queue[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd] = (
            asyncio.Queue(maxsize=max(2, int(target_fps * 1.65 * BUFFER_SECONDS)))
        )
        self._idle_frame: rtc.VideoFrame | None = None
        self._closed = False
        self._av_sync: rtc.AVSynchronizer | None = None

    def attach_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        """Let barge-in flush the synchronizer the runner publishes through.

        ``AvatarRunner`` clears *our* queue on barge-in but leaves its
        ``AVSynchronizer`` (and the audio source) holding already-paired frames,
        so audio and video drain independently after a hard stop and the two
        tracks end at different moments — visible as A/V desync at the cut. The
        worker hands us the synchronizer after starting the runner so
        :meth:`_flush_output` can cut both legs at the same point.
        """
        self._av_sync = av_sync

    async def _flush_output(self) -> None:
        """Drop everything still queued for publication (ours + the runner's).

        Order matters: clear our queue first, then the synchronizer, so a frame
        in flight can't slip into the synchronizer after it was flushed.
        """
        drain(self._out_queue)
        av_sync = self._av_sync
        if av_sync is None:
            return
        try:
            await av_sync.clear_queue()
        except Exception:
            logger.debug("[video_gen] av_sync flush failed", exc_info=True)

    async def _prepare_idle_frame(self, img: bytes | None) -> None:
        """Decode the still shown between generated frames (same face the service
        animates). Best-effort — a failure just disables the idle frame."""
        if img is None:
            return
        try:
            self._idle_frame = await asyncio.to_thread(
                image_to_i420, img, self._width, self._height
            )
        except Exception:
            logger.exception("[video_gen] idle-frame decode failed")

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        idle_period = 1.0 / self._target_fps
        starved_since: float | None = None
        while not self._closed:
            try:
                item = await asyncio.wait_for(self._out_queue.get(), timeout=idle_period)
            except asyncio.TimeoutError:
                # Buffer empty while the consumer wants a frame — this is the
                # stall the listener hears. Time it so the log distinguishes
                # "buffer too small" from "source stopped producing".
                if starved_since is None:
                    starved_since = time.monotonic()
                if self._idle_frame is not None:
                    yield self._idle_frame
                continue
            if starved_since is not None:
                stalled = time.monotonic() - starved_since
                starved_since = None
                if stalled > 0.2:
                    logger.warning(
                        "[video_gen] output starved %.2fs (buffer empty, cap=%d)",
                        stalled,
                        self._out_queue.maxsize,
                    )
            yield item
