"""VAD stage — provider-agnostic wrapper, mirrors SttStage / TtsStage.

The stage holds a LiveKit-compatible ``vad.VAD`` instance and exposes the
same lifecycle hooks (warmup / shutdown) as :class:`SttStage` /
:class:`TtsStage`. The actual frame-by-frame VAD inference happens inside
``AgentSession`` (which receives the raw VAD via ``stage.vad``); this
wrapper exists for **dependency injection symmetry** and **lifecycle
management**, not for driving VAD itself.

To add a new VAD provider, edit
``factory.py::SharedStageFactory._build_vad``. The stage layer doesn't
need to change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import vad as lk_vad

logger = logging.getLogger("providers.vad")


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
