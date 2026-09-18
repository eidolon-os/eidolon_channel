"""Text-to-Speech (TTS) provider stage for voice pipelines.

Provider-agnostic wrapper around any LiveKit-compatible ``tts.TTS`` instance.
The concrete plugin is built by the caller (typically in ``server.py`` based
on ``cfg.tts_provider``) and injected via :class:`TtsStage`'s constructor.
This stage knows nothing about specific providers.

To add a new TTS provider:
1. Implement ``livekit.agents.tts.TTS`` (and ``SynthesizeStream``) in a new
   ``plugins/tts/<provider>/`` module.
2. Add a ``<PROVIDER>TTSConfig`` dataclass that loads from
   ``<PROVIDER>_TTS_*`` env vars.
3. Reference it in ``common/config.py``'s ``AgentConfig`` and add a branch
   in ``factory.py::SharedStageFactory._build_tts`` that constructs your
   plugin and wraps it in ``TtsStage(your_plugin_instance)``.
4. Set ``TTS_PROVIDER=<provider>`` in ``.env``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from livekit.agents import tts as lk_tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from livekit.agents import tts as lk_tts
    from livekit.rtc import AudioFrame

logger = logging.getLogger("providers.tts")


class OutputScopedTTS(lk_tts.TTS):
    """Delegate synthesis unchanged; scope provider errors to speech output.

    LiveKit's default TTS error listener counts failed syntheses toward closing
    the whole AgentSession. That lifecycle is too broad for a multimodal device.
    Keep the real stream exception, metrics and provider timing, but expose the
    error as output_error to our existing output observer. No retries, audio
    generation, transcript synchronization or fallback providers live here.
    """

    def __init__(self, inner: lk_tts.TTS):
        super().__init__(
            capabilities=inner.capabilities,
            sample_rate=inner.sample_rate,
            num_channels=inner.num_channels,
        )
        self._inner = inner
        self._listeners = {}
        for source, target in (
            ("metrics_collected", "metrics_collected"),
            ("provider_event", "provider_event"),
            ("error", "output_error"),
        ):

            def callback(event, target=target):
                self.emit(target, event)

            self._listeners[source] = callback
            inner.on(source, callback)

    @property
    def model(self):
        return self._inner.model

    @property
    def provider(self):
        return self._inner.provider

    @property
    def label(self):
        return self._inner.label

    @property
    def markup(self):
        return self._inner.markup

    def _set_expressive(self, enabled):
        super()._set_expressive(enabled)
        self._inner._set_expressive(enabled)

    def synthesize(self, text, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return self._inner.synthesize(text, conn_options=conn_options)

    def stream(self, *, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return self._inner.stream(conn_options=conn_options)

    def prewarm(self):
        self._inner.prewarm()

    async def aclose(self):
        for event, callback in self._listeners.items():
            self._inner.off(event, callback)
        self._listeners.clear()
        await self._inner.aclose()


@dataclass
class TtsParams:
    """Parameters for the TTS stage."""

    sample_rate: int = 32000
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    voice: str = "female_0033_a"


class TtsStage:
    """Plugin-agnostic TTS stage.

    Wraps any LiveKit-compatible ``tts.TTS`` instance and exposes a uniform
    interface (warmup / shutdown / synthesize / stream) to the pipeline.
    The stage doesn't know which provider it wraps — that decision is made
    by the caller, which constructs the plugin and passes it to
    ``TtsStage(tts=...)``.
    """

    def __init__(
        self,
        tts: "lk_tts.TTS",
        *,
        params: TtsParams | None = None,
    ) -> None:
        self._tts = tts
        self._output_tts: OutputScopedTTS | None = None
        self._params = params or TtsParams()
        logger.info(
            "[TtsStage] initialized plugin=%s voice=%s sample_rate=%d speed=%.2f",
            type(tts).__name__,
            self._params.voice,
            self._params.sample_rate,
            self._params.speed,
        )

    @property
    def tts(self) -> "lk_tts.TTS":
        """The underlying LiveKit TTS plugin instance.

        Pass this directly to :class:`AgentSession` — any concrete subclass
        of ``livekit.agents.tts.TTS`` is compatible.
        """
        if self._output_tts is None:
            self._output_tts = OutputScopedTTS(self._tts)
        return self._output_tts

    async def warmup(self) -> None:
        """Warm up the TTS connection if the underlying plugin supports it.

        Plugins with persistent connections (those exposing a ``warmup()``
        method) open their WebSocket eagerly. Plugins without ``warmup()``
        are no-op'd via the ``hasattr`` check. Idempotent.
        """
        if hasattr(self._tts, "warmup"):
            try:
                logger.info("[TtsStage] warming up TTS...")
                await self._tts.warmup()
                logger.info("[TtsStage] TTS warmup complete")
            except Exception:
                logger.exception("[TtsStage] TTS warmup failed, continuing anyway")

    async def shutdown(self) -> None:
        """Close the persistent TTS connection if the plugin supports it."""
        if hasattr(self._tts, "shutdown"):
            try:
                logger.info("[TtsStage] shutting down TTS connection")
                await self._tts.shutdown()
            except Exception:
                logger.exception("[TtsStage] TTS shutdown error")

    async def synthesize(self, text: str) -> AsyncGenerator["AudioFrame", None]:
        """Synthesize text in one-shot mode (manual mode).

        Usage::

            async for frame in tts_stage.synthesize("Hello, how can I help?"):
                await play_audio(frame)
        """
        logger.debug("[TtsStage] synthesize() text=%r", text[:80])

        stream = self._tts.synthesize(text)
        try:
            async for audio in stream:
                yield audio.frame
        finally:
            await stream.aclose()
            if getattr(stream, "done", False):
                with contextlib.suppress(BaseException):
                    _ = stream.exception

    async def synthesize_all(self, text: str) -> list["AudioFrame"]:
        """Synthesize text and return all audio frames as a list."""
        frames: list["AudioFrame"] = []
        async for frame in self.synthesize(text):
            frames.append(frame)
        return frames

    def stream(self) -> "lk_tts.SynthesizeStream":
        """Create a streaming synthesis session.

        Used in streaming mode. The caller pushes text tokens via the
        returned stream's ``push()`` method and iterates over it for
        audio frames.

        Usage::

            stream = tts_stage.stream()
            stream.push("Hello ")
            stream.push("world")
            stream.flush()
            async for audio in stream:
                await play_audio(audio.frame)
        """
        return self._tts.stream()
