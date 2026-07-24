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

import av
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

logger = logging.getLogger("agent.avatar.video_gen")


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
        self._out_queue: asyncio.Queue[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd] = (
            asyncio.Queue(maxsize=max(2, int(target_fps * 2)))
        )
        self._idle_frame: rtc.VideoFrame | None = None
        self._closed = False

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
        while not self._closed:
            try:
                item = await asyncio.wait_for(self._out_queue.get(), timeout=idle_period)
            except asyncio.TimeoutError:
                if self._idle_frame is not None:
                    yield self._idle_frame
                continue
            yield item
