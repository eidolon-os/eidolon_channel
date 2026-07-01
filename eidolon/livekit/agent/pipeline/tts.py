"""Text-to-Speech (TTS) stage for the voice pipeline.

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
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from livekit.agents import tts as lk_tts
    from livekit.rtc import AudioFrame

logger = logging.getLogger("pipeline.tts")


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
        return self._tts

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
                logger.exception(
                    "[TtsStage] TTS warmup failed, continuing anyway"
                )

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
