"""BatchPipeline — MANUAL/BATCH mode: audio blob via LiveKit Room, no AgentSession."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import livekit as lk
    from livekit.rtc import AudioFrame, Room

from .factory import SharedStageFactory
from .pipeline.base import BasePipeline
from .pipeline.llm import LlmInput
from .pipeline.types import PipelineCallbacks, PipelineState

logger = logging.getLogger("agent")


class BatchPipeline(BasePipeline):
    """
    BATCH mode: audio blob uploaded via LiveKit Room, no AgentSession.

    Listens for audio tracks published by participants. When a track ends,
    processes the full audio blob: STT -> LLM -> TTS. Result audio is
    published back to the same Room.

    Flow::

        User publishes audio blob to Room
        -> BatchPipeline detects track ended
        -> STT.recognize(audio_blob)
        -> LLM.chat(transcript)
        -> TTS.synthesize(response)
        -> Publish result audio to Room

    Usage::

        factory = SharedStageFactory(config)
        pipeline = BatchPipeline(factory)
        await pipeline.run(room)
    """

    # Round 8 R8.6 — was a hardcoded ``timeout=30.0`` (BatchPipeline.B3 in
    # the audit). Configurable so a deployment doing dictation / longer
    # turns can extend it. Kept as a class attribute (rather than dataclass
    # config) since BatchPipeline is one shot and rarely tuned.
    AUDIO_TRACK_TIMEOUT: float = 30.0

    def __init__(
        self,
        factory: SharedStageFactory,
        *,
        callbacks: PipelineCallbacks | None = None,
        audio_track_timeout: float | None = None,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        self._processing: bool = False
        # Round 8 R8.6 — replace ``while room.isconnected: sleep(1.0)``
        # polling with an event-driven shutdown signal (mirrors what
        # streaming.py did for ``session.on("close")``). Avoids the
        # silent-stall failure mode where ``room.isconnected`` doesn't
        # flip in time.
        self._room_disconnected_event: asyncio.Event = asyncio.Event()
        if audio_track_timeout is not None:
            self.AUDIO_TRACK_TIMEOUT = audio_track_timeout

    async def run(self, room: Room) -> None:
        """Start the batch pipeline. Blocks until room disconnects."""
        self._room = room
        self._started = True
        logger.info("[BatchPipeline] starting room=%s", room.name)

        # Warm up persistent-connection stages (STT/TTS) before the first
        # audio track arrives. For per-stream plugins (e.g. Bailian) this is
        # a no-op via hasattr-duck-typing in BasePipeline._warmup_stages.
        await self._warmup_stages()

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: "lk.Track",
            publication: "lk.RemoteTrackPublication",
            participant: "lk.RemoteParticipant",
        ) -> None:
            if track.kind == lk.TrackKind.AUDIO:
                asyncio.create_task(self._process_audio_track(track, participant))

        @room.on("disconnected")
        def on_disconnected(reason: object) -> None:
            logger.info(
                "[BatchPipeline] room.disconnected event reason=%s", reason,
            )
            self._room_disconnected_event.set()

        try:
            await self._room_disconnected_event.wait()
        except asyncio.CancelledError:
            logger.info("[BatchPipeline] cancelled")
            raise
        finally:
            logger.info("[BatchPipeline] room disconnected, shutting down")
            await self.shutdown()

    async def shutdown(self) -> None:
        """Tear down persistent stage connections then chain to parent."""
        logger.info("[BatchPipeline] shutting down")
        await self._shutdown_stages()
        await super().shutdown()

    async def _process_audio_track(
        self, track: "lk.Track", participant: "lk.RemoteParticipant"
    ) -> None:
        """Receive an audio track and process it when it ends."""
        if self._processing:
            logger.info("[BatchPipeline] already processing, skipping")
            return

        self._processing = True
        self._state = PipelineState.PROCESSING_AUDIO

        try:
            audio_frames: list[AudioFrame] = []
            track_done = asyncio.Event()

            @track.on("frame_received")
            def on_frame(frame: AudioFrame) -> None:
                audio_frames.append(frame)

            @track.on("track_ended")
            def on_track_ended() -> None:
                track_done.set()

            # Wait for track to end, with timeout. Configurable via
            # ``BatchPipeline(audio_track_timeout=...)`` (R8.6).
            try:
                await asyncio.wait_for(
                    track_done.wait(), timeout=self.AUDIO_TRACK_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[BatchPipeline] track timeout (%.1fs), processing anyway",
                    self.AUDIO_TRACK_TIMEOUT,
                )

            if not audio_frames:
                logger.warning("[BatchPipeline] no audio frames received")
                return

            # Merge frames into a blob
            audio_blob = self._frames_to_pcm_blob(audio_frames)
            logger.info(
                "[BatchPipeline] collected %d frames, audio_size=%d bytes",
                len(audio_frames),
                len(audio_blob),
            )

            # 1. STT
            t0 = time.monotonic()
            transcript = await self._factory.stt.recognize(audio_blob)
            stt_latency_ms = (time.monotonic() - t0) * 1000
            logger.info("[BatchPipeline] STT done transcript=%r latency_ms=%.1f", transcript[:80], stt_latency_ms)

            if not transcript.strip():
                logger.warning("[BatchPipeline] empty transcript, skipping")
                return

            self._callbacks.on_user_message(transcript)

            # 2. LLM
            self._state = PipelineState.GENERATING
            t1 = time.monotonic()
            self._callbacks.on_agent_started_speaking()

            llm_output = await self._factory.llm.chat(LlmInput(text=transcript))
            llm_latency_ms = (time.monotonic() - t1) * 1000
            response_text = llm_output.text if hasattr(llm_output, "text") else llm_output
            logger.info("[BatchPipeline] LLM done response=%r latency_ms=%.1f", response_text[:80], llm_latency_ms)

            # 3. TTS + publish to room
            self._state = PipelineState.SPEAKING
            t2 = time.monotonic()

            if self._room is None:
                logger.warning("[BatchPipeline] no room, cannot publish TTS audio")
                return

            total_frames = 0
            async for frame in self._factory.tts.synthesize(response_text):
                await self._room.local_publish_audio(frame)
                total_frames += 1

            tts_latency_ms = (time.monotonic() - t2) * 1000
            logger.info(
                "[BatchPipeline] TTS done published=%d frames latency_ms=%.1f",
                total_frames,
                tts_latency_ms,
            )

            self._callbacks.on_agent_message(response_text)
            self._callbacks.on_agent_ended_speaking()
            self._callbacks.on_agent_response_done()

        except asyncio.CancelledError:
            self._state = PipelineState.IDLE
            raise
        except Exception as e:
            logger.exception("[BatchPipeline] error processing audio track")
            self._callbacks.on_error(e)
        finally:
            self._processing = False
            self._state = PipelineState.IDLE

    @staticmethod
    def _frames_to_pcm_blob(frames: list[AudioFrame]) -> bytes:
        """Concatenate audio frames into a raw PCM blob."""
        parts: list[bytes] = []
        for frame in frames:
            parts.append(bytes(frame.data))
        return b"".join(parts)
