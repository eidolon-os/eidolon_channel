"""Batch ``VideoGenerator`` bridging the digital-human service into AvatarRunner.

Data flow inside one avatar worker:

    agent TTS PCM ──(DataStream)──▶ DataStreamAudioReceiver ──▶ push_audio()
        └─ accumulate one segment's PCM until AudioSegmentEnd
    AudioSegmentEnd ──▶ enqueue segment ──▶ [processor task]
        └─ PCM→WAV → POST /api/stream_video → decode+retime → frames → _out_queue → AudioSegmentEnd
    AvatarRunner._forward_video ◀── __aiter__() ◀── _out_queue (or idle frame when empty)

The shared output/idle side lives in :class:`DHVideoGeneratorBase`; this class
owns the batch audio-buffering + per-segment POST. The streaming variant is
:class:`~eidolon.livekit.avatar.streaming_video_generator.StreamingDHVideoGenerator`.

Barge-in: LiveKit routes ``clear_buffer`` (a ``lk.clear_buffer`` RPC from the
agent) here — we abort the in-flight request, drop pending segments, and flush
buffered frames so the face stops promptly.
"""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from ._video_gen_base import DHVideoGeneratorBase, drain
from .decoder import decode_stream
from .service_client import DigitalHumanServiceClient, pcm16_to_wav

logger = logging.getLogger("agent.avatar.video_gen")


class EidolonDHVideoGenerator(DHVideoGeneratorBase):
    """Turn agent TTS audio segments into a synchronized talking-head A/V stream."""

    def __init__(
        self,
        service: DigitalHumanServiceClient,
        *,
        width: int,
        height: int,
        target_fps: float,
        output_sample_rate: int = 24000,
        face_image: bytes | None = None,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            target_fps=target_fps,
            output_sample_rate=output_sample_rate,
            face_image=face_image,
        )
        self._service = service

        self._seg_buf = bytearray()
        self._in_sr: int | None = None
        self._in_ch = 1

        self._seg_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._processor_task: asyncio.Task | None = None
        self._seg_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ setup
    async def warmup(self) -> None:
        """Prepare the idle frame (same face the service animates) and start the processor."""
        img = self._face_image
        if img is None:
            try:
                img = await self._service.default_avatar_jpeg()
            except Exception:
                logger.warning("[video_gen] default_avatar fetch failed; idle frame disabled")
                img = None
        await self._prepare_idle_frame(img)
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._run_processor(), name="avatar-processor")

    # ------------------------------------------------------- VideoGenerator API
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._finalize_segment()
            return
        if self._in_sr is None:
            self._in_sr = frame.sample_rate
            self._in_ch = frame.num_channels
        self._seg_buf += bytes(frame.data)

    async def _finalize_segment(self) -> None:
        if not self._seg_buf or self._in_sr is None:
            self._seg_buf.clear()
            return
        logger.info(
            "[video_gen] audio segment received pcm_bytes=%d in_sr=%d",
            len(self._seg_buf),
            self._in_sr,
        )
        wav = pcm16_to_wav(bytes(self._seg_buf), sample_rate=self._in_sr, num_channels=self._in_ch)
        self._seg_buf = bytearray()
        await self._seg_queue.put(wav)

    async def clear_buffer(self) -> None:
        """Barge-in: drop pending input, in-flight request, and buffered output."""
        self._seg_buf.clear()
        drain(self._seg_queue)
        if self._seg_task is not None and not self._seg_task.done():
            self._seg_task.cancel()
            await asyncio.gather(self._seg_task, return_exceptions=True)
        drain(self._out_queue)

    # -------------------------------------------------------------- processor
    async def _run_processor(self) -> None:
        while not self._closed:
            wav = await self._seg_queue.get()
            self._seg_task = asyncio.create_task(self._process_segment(wav))
            try:
                await self._seg_task
            except asyncio.CancelledError:
                # cancelled by clear_buffer (barge-in) — keep serving next segments
                pass
            except Exception:
                logger.exception("[video_gen] segment processing failed")
            finally:
                self._seg_task = None

    async def _process_segment(self, wav: bytes) -> None:
        chunks = self._service.stream_video(wav_bytes=wav, image_bytes=self._face_image)
        vcount = acount = 0
        logger.info("[video_gen] segment POST start wav_bytes=%d", len(wav))
        try:
            async for frame in decode_stream(
                chunks, audio_sample_rate=self._out_sr, target_fps=self._target_fps
            ):
                if isinstance(frame, rtc.VideoFrame):
                    vcount += 1
                else:
                    acount += 1
                await self._out_queue.put(frame)
            await self._out_queue.put(AudioSegmentEnd())
            logger.info("[video_gen] segment done video_frames=%d audio_frames=%d", vcount, acount)
        except asyncio.CancelledError:
            logger.info("[video_gen] segment aborted (barge-in) video_frames=%d", vcount)
            raise
        except Exception:
            logger.exception("[video_gen] stream/decode failed for segment")

    async def aclose(self) -> None:
        self._closed = True
        if self._seg_task is not None and not self._seg_task.done():
            self._seg_task.cancel()
        if self._processor_task is not None and not self._processor_task.done():
            self._processor_task.cancel()
            await asyncio.gather(self._processor_task, return_exceptions=True)
        await self._service.aclose()
