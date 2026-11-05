"""VAD stage — provider-agnostic wrapper, mirrors SttStage / TtsStage.

The stage holds a LiveKit-compatible ``vad.VAD`` instance and exposes the
same lifecycle hooks (warmup / shutdown) as :class:`SttStage` /
:class:`TtsStage`. The actual frame-by-frame VAD inference happens inside
``AgentSession`` (which receives the raw VAD via ``stage.vad``); this
wrapper exists for **dependency injection symmetry** and **lifecycle
management**, not for driving VAD itself.

To add a new VAD provider, add it to :func:`load_vad`. Both the worker
prewarm and ``SharedStageFactory`` build through it, so a provider can never
be warmed by one path and silently missing from the other.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import vad as lk_vad

    from eidolon.livekit.common.config.schema import VadPolicyConfig

logger = logging.getLogger("providers.vad")

FIRERED_PROVIDERS = frozenset({"firered", "firered_pvad"})
SILERO_PROVIDERS = frozenset({"silero"})
DISABLED_PROVIDERS = frozenset({"none", "disabled"})


def canonical_vad_provider(provider: str) -> str:
    """The provider name reported in status and logs."""
    provider = provider.lower()
    if provider in FIRERED_PROVIDERS:
        return "firered_pvad"
    return provider


def load_vad(provider: str, vad_cfg: "VadPolicyConfig") -> "lk_vad.VAD | None":
    """Build the configured VAD; ``None`` only for an explicitly disabled VAD.

    Every provider reads the same turn-policy thresholds into the same
    window-classifier state machine (``plugins/vad/streaming.py``). Their
    probabilities are not recalibrated to each other, so the same threshold can
    trip more or less readily after a provider change.
    """
    provider = provider.lower()
    if provider in DISABLED_PROVIDERS:
        return None
    options = dict(
        min_speech_duration=vad_cfg.min_speech_duration_ms / 1000.0,
        min_silence_duration=vad_cfg.min_silence_duration_ms / 1000.0,
        prefix_padding_duration=vad_cfg.prefix_padding_ms / 1000.0,
        max_buffered_speech=vad_cfg.max_buffered_speech_ms / 1000.0,
        activation_threshold=vad_cfg.activation_threshold,
    )
    if provider in SILERO_PROVIDERS:
        from eidolon.livekit.plugins.vad.silero import SileroVAD

        return SileroVAD.load(**options)
    if provider in FIRERED_PROVIDERS:
        from eidolon.livekit.plugins.vad.firered import FireredPvadVAD

        return FireredPvadVAD.load(**options)
    raise ValueError(f"unsupported VAD provider: {provider}")


class VadStage:
    """Plugin-agnostic VAD stage.

    Wraps any LiveKit-compatible ``vad.VAD`` instance. The wrapped VAD is
    accessed via the :attr:`vad` property and passed directly to
    ``AgentSession`` (which expects a raw ``lk_vad.VAD``, not a stage).
    """

    def __init__(self, vad: "lk_vad.VAD") -> None:
        self._vad = vad
        logger.info("[VadStage] initialized plugin=%s", type(vad).__name__)

    @property
    def vad(self) -> "lk_vad.VAD":
        """The underlying LiveKit VAD instance — pass to ``AgentSession``."""
        return self._vad

    async def warmup(self) -> None:
        """Warm up the VAD if the underlying plugin supports it.

        Most VAD plugins preload at ``VAD.load()`` time and don't need a
        separate warmup; this is a no-op for them via the ``hasattr`` check.
        """
        if hasattr(self._vad, "warmup"):
            try:
                logger.info("[VadStage] warming up VAD...")
                await self._vad.warmup()
            except Exception:
                logger.exception(
                    "[VadStage] VAD warmup failed, continuing anyway"
                )

    async def shutdown(self) -> None:
        """Tear down the VAD if the underlying plugin supports it."""
        if hasattr(self._vad, "shutdown"):
            try:
                logger.info("[VadStage] shutting down VAD")
                await self._vad.shutdown()
            except Exception:
                logger.exception("[VadStage] VAD shutdown error")
