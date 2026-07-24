"""Streaming ``VideoGenerator``: feed TTS to Ditto ``/ws/audio_stream`` live.

Data flow inside one avatar worker (streaming mode):

    agent TTS PCM ──▶ push_audio() ──▶ resample to 16 kHz mono float32
        └─ first frame of a turn opens a DittoStreamSession (cond_image seeded)
    resampled chunks ──▶ session.send_audio()  (streamed as produced)
    session.segments() (fMP4) ──▶ progressive_decode ──▶ frames ──▶ _out_queue
    AudioSegmentEnd ──▶ session.request_stop() ──▶ drain ──▶ AudioSegmentEnd out

The face starts within one first-frame latency of the first audio chunk instead
of after the whole utterance. Idle frames are intentionally NOT emitted between
turns: the worker's track then carries generated frames only while speaking, and
clients render their local idle. Barge-in aborts the in-flight turn.
"""

from __future__ import annotations

import asyncio
import logging

import av
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from ._video_gen_base import DHVideoGeneratorBase, drain
from .ditto_streaming_client import DittoStreamClient, DittoStreamSession
from .progressive_decoder import progressive_decode

logger = logging.getLogger("agent.avatar.streaming_gen")

_STREAM_SAMPLE_RATE = 16000  # /ws/audio_stream requires 16 kHz mono float32
_END = object()


class StreamingDHVideoGenerator(DHVideoGeneratorBase):
    """Turn agent TTS into a live talking-head via the streaming interface."""

    def __init__(
        self,
        client: DittoStreamClient,
        *,
        width: int,
        height: int,
        target_fps: float,
        output_sample_rate: int = 24000,
        face_image: bytes | None = None,
        opus_frame_ms: float = 20.0,
        jpeg_quality: int = 60,
        fast_start_samples: int = 8000,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            target_fps=target_fps,
            output_sample_rate=output_sample_rate,
            face_image=face_image,
        )
        self._client = client
        self._opus_frame_ms = opus_frame_ms
        self._jpeg_quality = jpeg_quality
        self._fast_start_samples = fast_start_samples
        self._turn: _StreamTurn | None = None

    async def warmup(self) -> None:
        # No idle frame in streaming mode: the worker publishes generated frames
        # only while speaking; clients own idle rendering. (Nothing to prewarm.)
        return

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            if self._turn is not None:
                self._turn.end()
            return
        if self._closed:
            return
        turn = self._turn
        if turn is None or turn.finished:
            if turn is not None:
                await turn.abort()  # new speech supersedes a stale turn
            turn = _StreamTurn(self)
            self._turn = turn
        turn.feed(frame)

    async def clear_buffer(self) -> None:
        """Barge-in: abort the in-flight turn and flush buffered output."""
        if self._turn is not None:
            await self._turn.abort()
            self._turn = None
        drain(self._out_queue)

    async def aclose(self) -> None:
        self._closed = True
        if self._turn is not None:
            await self._turn.abort()
            self._turn = None


class _StreamTurn:
    """One speaking turn: opens a session, streams resampled audio, and pumps the
    progressively-decoded frames onto the generator's output queue."""

    def __init__(self, gen: StreamingDHVideoGenerator) -> None:
        self._gen = gen
        self._audio_in: asyncio.Queue[object] = asyncio.Queue()
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=_STREAM_SAMPLE_RATE)
        self.finished = False
        self._task = asyncio.create_task(self._run())

    def feed(self, frame: rtc.AudioFrame) -> None:
        try:
            for chunk in self._resample(frame):
                self._audio_in.put_nowait(chunk)
        except Exception:
            logger.debug("[streaming_gen] resample failed", exc_info=True)

    def end(self) -> None:
        self._audio_in.put_nowait(_END)

    async def abort(self) -> None:
        self.finished = True
        if not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def _resample(self, frame: rtc.AudioFrame) -> list[bytes]:
        arr = np.frombuffer(bytes(frame.data), dtype=np.int16).reshape(1, -1)
        layout = "mono" if frame.num_channels == 1 else "stereo"
        af = av.AudioFrame.from_ndarray(arr, format="s16", layout=layout)
        af.sample_rate = frame.sample_rate
        out: list[bytes] = []
        for rf in self._resampler.resample(af):
            out.append(rf.to_ndarray().astype(np.float32).tobytes())
        return out

    async def _run(self) -> None:
        session: DittoStreamSession | None = None
        sender: asyncio.Task | None = None
        try:
            session = await self._gen._client.open(
                image_bytes=self._gen._face_image,
                prefer_fps=self._gen._target_fps,
                screen_width=self._gen._width,
                screen_height=self._gen._height,
                opus_frame_ms=self._gen._opus_frame_ms,
                jpeg_quality=self._gen._jpeg_quality,
                fast_start_samples=self._gen._fast_start_samples,
            )
            sender = asyncio.create_task(self._send(session))
            async for frame in progressive_decode(
                session.segments(), output_sample_rate=self._gen._out_sr
            ):
                await self._gen._out_queue.put(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[streaming_gen] turn failed")
        finally:
            self.finished = True
            if sender is not None and not sender.done():
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if session is not None:
                await session.aclose()
            try:
                self._gen._out_queue.put_nowait(AudioSegmentEnd())
            except asyncio.QueueFull:
                await self._gen._out_queue.put(AudioSegmentEnd())

    async def _send(self, session: DittoStreamSession) -> None:
        while True:
            item = await self._audio_in.get()
            if item is _END:
                await session.request_stop()
                return
            await session.send_audio(item)  # type: ignore[arg-type]
