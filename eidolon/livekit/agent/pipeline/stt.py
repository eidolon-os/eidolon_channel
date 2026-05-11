"""Speech-to-Text (STT) stage for the voice pipeline.

Provider-agnostic wrapper around any LiveKit-compatible ``stt.STT`` instance.
The concrete plugin is built by the caller (typically in ``server.py`` based
on ``cfg.stt_provider``) and injected via :class:`SttStage`'s constructor.
This stage knows nothing about specific providers.

To add a new STT provider:
1. Implement ``livekit.agents.stt.STT`` (and ``RecognizeStream``) in a new
   ``plugins/stt/<provider>/`` module.
2. Add a ``<PROVIDER>STTConfig`` dataclass that loads from
   ``<PROVIDER>_STT_*`` env vars.
3. Reference it in ``common/config.py``'s ``AgentConfig`` and add a branch
   in ``factory.py::SharedStageFactory._build_stt`` that constructs your
   plugin and wraps it in ``SttStage(your_plugin_instance)``.
4. Set ``STT_PROVIDER=<provider>`` in ``.env``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import stt as lk_stt

logger = logging.getLogger("pipeline.stt")


@dataclass
class SttParams:
    """Parameters for the STT stage's batch ``recognize()`` helper.

    Note: only ``sample_rate`` is currently used (to construct synthetic
    ``AudioFrame``s for one-shot recognition). ``language`` and ``itn`` are
    plugin-specific concerns that should live in the plugin's own config
    (``BailianSTTConfig`` / ``SenseTimeSTTConfig``).
    """

    language: str = "zh"
    sample_rate: int = 16000
    itn: bool = True


class SttStage:
    """Plugin-agnostic STT stage.

    Wraps any LiveKit-compatible ``stt.STT`` instance and exposes a uniform
    interface (warmup / shutdown / recognize / stream) to the pipeline.
    The stage doesn't know which provider it wraps — that decision is made
    by the caller, which constructs the plugin and passes it to
    ``SttStage(stt=...)``.
    """

    def __init__(
        self,
        stt: "lk_stt.STT",
        *,
        params: SttParams | None = None,
    ) -> None:
        self._stt = stt
        self._params = params or SttParams()
        logger.info(
            "[SttStage] initialized plugin=%s language=%s sample_rate=%d",
            type(stt).__name__,
            self._params.language,
            self._params.sample_rate,
        )

    @property
    def stt(self) -> "lk_stt.STT":
        """The underlying LiveKit STT plugin instance.

        Pass this directly to :class:`AgentSession` — any concrete subclass
        of ``livekit.agents.stt.STT`` is compatible.
        """
        return self._stt

    async def warmup(self) -> None:
        """Warm up the STT connection if the underlying plugin supports it.

        Plugins with persistent connections (those exposing a ``warmup()``
        method) open their WebSocket eagerly so the first utterance doesn't
        pay the handshake cost. Plugins without ``warmup()`` are no-op'd
        via the ``hasattr`` check below.
        """
        if hasattr(self._stt, "warmup"):
            try:
                logger.info("[SttStage] warming up STT...")
                await self._stt.warmup()
                logger.info("[SttStage] STT warmup complete")
            except Exception:
                logger.exception(
                    "[SttStage] STT warmup failed, continuing anyway"
                )

    async def shutdown(self) -> None:
        """Close any persistent STT connection if the plugin supports it.

        Plugins without a ``shutdown()`` method are no-op'd.
        """
        if hasattr(self._stt, "shutdown"):
            try:
                logger.info("[SttStage] shutting down STT connection")
                await self._stt.shutdown()
            except Exception:
                logger.exception("[SttStage] STT shutdown error")

    async def recognize(self, audio: bytes) -> str:
        """Transcribe a complete audio buffer in one-shot mode.

        Used in manual mode when the client sends a pre-recorded audio blob.
        Delegates to the plugin's ``recognize()`` (which may not be supported
        by all plugins — streaming-only plugins should raise NotImplementedError).

        Args:
            audio: Raw PCM audio bytes (16-bit, mono, ``sample_rate`` Hz).

        Returns:
            The transcribed text.
        """
        from livekit import rtc

        logger.debug("[SttStage] recognize() audio_size=%d bytes", len(audio))

        num_samples = len(audio) // 2  # 16-bit PCM
        frame = rtc.AudioFrame(
            data=audio,
            sample_rate=self._params.sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
        )

        result = await self._stt.recognize([frame])
        return self._extract_text(result)

    async def recognize_streaming(self, audio: bytes) -> str:
        """Transcribe audio via the plugin's streaming API (one-shot).

        Plugin-agnostic alternative to :meth:`recognize` — uses the streaming
        path that every LiveKit STT plugin must support. Slightly higher
        latency, but works with streaming-only plugins (e.g. SenseTime STT).
        """
        from livekit import rtc
        from livekit.agents import stt as lk_stt

        num_samples = len(audio) // 2
        frame = rtc.AudioFrame(
            data=audio,
            sample_rate=self._params.sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
        )

        stream = self._stt.stream()
        stream.push_frame(frame)
        stream.end_input()

        transcript_parts: list[str] = []
        async for event in stream:
            if event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                if event.alternatives:
                    transcript_parts.append(event.alternatives[0].text)

        return "".join(transcript_parts)

    def stream(self) -> "lk_stt.RecognizeStream":
        """Create a streaming transcription session.

        Used in streaming mode. The caller pushes audio frames via
        ``push_frame()`` and iterates over the returned stream for
        transcription events. The framework (AgentSession) calls
        ``flush()`` / ``end_input()`` automatically at the end of
        each user turn.

        Returns the plugin's ``RecognizeStream`` subclass (e.g. a
        ``SenseTimeSpeechStream`` or ``BailianFunASRSpeechStream``);
        callers should program against the abstract
        :class:`livekit.agents.stt.RecognizeStream` interface.
        """
        return self._stt.stream()

    @staticmethod
    def _extract_text(event) -> str:
        """Extract text from a SpeechEvent."""
        from livekit.agents import stt as lk_stt

        if event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
            if event.alternatives:
                return event.alternatives[0].text or ""
        return ""
