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

from ._video_gen_base import DHVideoGeneratorBase
from .ditto_streaming_client import DittoStreamClient, DittoStreamSession, encode_cond_image
from .progressive_decoder import progressive_decode

logger = logging.getLogger("agent.avatar.streaming_gen")

_STREAM_SAMPLE_RATE = 16000  # /ws/audio_stream requires 16 kHz mono float32
_END = object()

# Silence fed after an utterance so the service can render the tail of the real
# speech at its normal pace. Measured: 1 s is enough for the whole utterance to
# land before the service's input-idle timeout would hold it back.
_FLUSH_TAIL_S = 1.0


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
        # The face never changes for a session, so encode it once instead of
        # base64-ing ~100 KB on every utterance.
        self._cond_image_b64 = encode_cond_image(face_image) if face_image else None

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
        # Audio after a turn's input closed belongs to a *new* utterance. Feeding
        # it to the old turn would drop it silently — its sender has stopped —
        # so supersede the turn even while it is still draining its tail.
        if turn is None or turn.finished or turn.input_done:
            if turn is not None:
                await turn.abort()
            turn = _StreamTurn(self)
            self._turn = turn
        turn.feed(frame)

    async def clear_buffer(self) -> None:
        """Barge-in: abort the in-flight turn and flush buffered output.

        Flushing reaches the runner's synchronizer too, so audio and video stop
        at the same point instead of each draining its own residue.
        """
        if self._turn is not None:
            await self._turn.abort()
            self._turn = None
        await self._flush_output()

    async def aclose(self) -> None:
        self._closed = True
        if self._turn is not None:
            await self._turn.abort()
            self._turn = None
        await self._client.aclose()


class _StreamTurn:
    """One speaking turn: opens a session, streams resampled audio, and pumps the
    progressively-decoded frames onto the generator's output queue."""

    def __init__(self, gen: StreamingDHVideoGenerator) -> None:
        self._gen = gen
        self._audio_in: asyncio.Queue[object] = asyncio.Queue()
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=_STREAM_SAMPLE_RATE)
        self.finished = False
        self.input_done = False
        self._session: DittoStreamSession | None = None
        self._task = asyncio.create_task(self._run())

    def feed(self, frame: rtc.AudioFrame) -> None:
        try:
            for chunk in self._resample(frame):
                self._audio_in.put_nowait(chunk)
        except Exception:
            logger.debug("[streaming_gen] resample failed", exc_info=True)

    def end(self) -> None:
        """End of this utterance's audio."""
        self.input_done = True
        self._audio_in.put_nowait(_END)

    async def abort(self) -> None:
        """Barge-in: tell the service to stop generating, then drop the turn.

        This is where ``stop`` belongs — cancelling is the intent, and it keeps
        the service from generating for speech the user already interrupted.
        """
        self.finished = True
        session = self._session
        if session is not None:
            try:
                await session.request_stop()
            except Exception:
                logger.debug("[streaming_gen] stop on abort failed", exc_info=True)
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
        frames = 0
        try:
            session = await self._gen._client.open(
                cond_image_b64=self._gen._cond_image_b64,
                prefer_fps=self._gen._target_fps,
                screen_width=self._gen._width,
                screen_height=self._gen._height,
                opus_frame_ms=self._gen._opus_frame_ms,
                jpeg_quality=self._gen._jpeg_quality,
                fast_start_samples=self._gen._fast_start_samples,
            )
            self._session = session
            sender = asyncio.create_task(self._send(session))
            async for frame in progressive_decode(
                session.segments(), output_sample_rate=self._gen._out_sr
            ):
                frames += 1
                await self._gen._out_queue.put(frame)
            logger.info("[streaming_gen] turn done frames=%d", frames)
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
            # Only close a segment we actually published. Signalling the end of a
            # turn that produced nothing makes the runner report playback finished
            # for audio that was never captured ("playback_finished called more
            # times than playback segments were captured").
            if frames:
                try:
                    self._gen._out_queue.put_nowait(AudioSegmentEnd())
                except asyncio.QueueFull:
                    await self._gen._out_queue.put(AudioSegmentEnd())

    async def _send(self, session: DittoStreamSession) -> None:
        """Stream this utterance's audio, then flush its tail.

        The service is audio-driven and always lags its input, so when the audio
        simply stops it has no way to render the last of the utterance: it holds
        that remainder until a ~5 s input-idle timeout, which the listener hears
        as a stall mid-sentence. ``stop`` is not the answer either — it aborts,
        discarding the undelivered remainder. Feeding a second of silence gives
        the service the lookahead to finish the real speech at its normal pace
        (measured: the full utterance lands before the idle timeout would even
        begin), after which the socket can close on the quiet.
        """
        while True:
            item = await self._audio_in.get()
            if item is _END:
                break
            await session.send_audio(item)  # type: ignore[arg-type]
        silence = np.zeros(_STREAM_SAMPLE_RATE // 10, dtype=np.float32).tobytes()
        for _ in range(int(_FLUSH_TAIL_S * 10)):
            await session.send_audio(silence)
        session.mark_input_complete()
