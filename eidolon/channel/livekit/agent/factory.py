"""Voice stage factory — constructs and owns the STT/LLM/TTS/VAD stage stack.

Both :class:`StreamingPipeline` and :class:`BatchPipeline` receive the same
factory instance, so they share the underlying model connections and stages.

Two construction paths:

1. :meth:`SharedStageFactory.from_config` — production path. Reads the active
   provider for each stage from :class:`AgentConfig` and builds plugin
   instances accordingly. This is where provider→plugin selection lives.

2. :meth:`SharedStageFactory.__init__` — test/customisation path. Takes pre-built
   stage instances directly so tests can inject mocks and applications can
   bypass the default provider selection if needed.

Adding a new STT/TTS provider only requires editing the corresponding
``_build_<stage>`` method below; pipeline / server.py code stays untouched.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from livekit.agents import llm as lk_llm
    from livekit.agents import vad as lk_vad
    from livekit.rtc import Room

    from eidolon.channel.livekit.common.config import AgentConfig

from .pipeline.llm import LlmParams, LivekitLlmStage
from .pipeline.stt import SttParams, SttStage
from .pipeline.tts import TtsParams, TtsStage
from .pipeline.vad import VadStage

logger = logging.getLogger("agent")


class SharedStageFactory:
    """Constructs and owns shared stage instances for the voice pipeline.

    Use :meth:`from_config` to build everything from an :class:`AgentConfig`,
    or pass stages explicitly to :meth:`__init__` for testing / advanced use.
    """

    def __init__(
        self,
        *,
        llm: "lk_llm.LLM",
        stt: SttStage,
        tts: TtsStage,
        vad: "lk_vad.VAD | None" = None,
        llm_params: LlmParams | None = None,
    ) -> None:
        if llm is None:
            raise ValueError("llm must not be None")
        if stt is None:
            raise ValueError("stt must not be None")
        if tts is None:
            raise ValueError("tts must not be None")

        self.stt: SttStage = stt
        self.tts: TtsStage = tts
        self.llm = LivekitLlmStage(llm=llm, params=llm_params or LlmParams())
        # Wrap raw VAD in VadStage for symmetry with stt / tts. AgentSession
        # still receives the raw VAD via stage.vad property.
        self.vad: VadStage | None = VadStage(vad) if vad is not None else None

        logger.info(
            "[SharedStageFactory] initialized stt=%s llm=%s tts=%s vad=%s",
            type(self.stt).__name__,
            type(self.llm).__name__,
            type(self.tts).__name__,
            type(self.vad).__name__ if self.vad else None,
        )

    # ------------------------------------------------------------------
    # Production path: build everything from AgentConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: "AgentConfig",
        *,
        prebuilt_vad: "lk_vad.VAD | None" = None,
        livekit_room: "Room | None" = None,
    ) -> "SharedStageFactory":
        """Build all stages from the agent's configuration.

        Args:
            cfg: The :class:`AgentConfig` (typically loaded from .env).
            prebuilt_vad: If supplied (e.g. from worker prewarm), this VAD
                instance is used instead of building a fresh one. Useful when
                a worker process loads the VAD model once at startup and
                reuses it across jobs.
            livekit_room: When ``REMOTE_AGENT_RPC_TARGET`` is set, used to
                build ``session_id`` for the remote agent (``livekit:<sid>``).

        Returns:
            A fully wired :class:`SharedStageFactory`.

        Raises:
            ValueError: If the LLM cannot be built (missing dependency or
                config) — STT/TTS provider mismatches surface from
                ``cfg._validate()`` at config-load time.
        """
        if cfg.remote_agent_rpc.target:
            from eidolon.channel.livekit.agent.remote_agent_rpc.grpc_llm import (
                RemoteAgentGrpcLlm,
            )

            sid = getattr(livekit_room, "sid", "") if livekit_room is not None else ""
            name = getattr(livekit_room, "name", "") if livekit_room is not None else ""
            session_key = sid or name or "unknown"
            llm = RemoteAgentGrpcLlm(
                target=cfg.remote_agent_rpc.target,
                session_id=f"livekit:{session_key}",
                locale=cfg.remote_agent_rpc.locale,
                display_model=cfg.llm.model or "remote_agent_rpc",
            )
            logger.info(
                "[SharedStageFactory] using RemoteAgentGrpcLlm target=%r session_id=%s",
                cfg.remote_agent_rpc.target,
                f"livekit:{session_key}",
            )
        else:
            llm = cls._build_llm(cfg)
            if llm is None:
                raise ValueError(
                    "LLM not configured. Set OPENAI_LLM_BASE_URL / "
                    "OPENAI_LLM_MODEL / OPENAI_LLM_API_KEY in your .env, "
                    "and ensure 'livekit-plugins-openai' is installed."
                )

        stt = cls._build_stt(cfg)
        tts = cls._build_tts(cfg)
        vad = prebuilt_vad if prebuilt_vad is not None else cls._build_vad(cfg)

        return cls(
            llm=llm,
            stt=stt,
            tts=tts,
            vad=vad,
            llm_params=LlmParams(model=cfg.llm.model, temperature=0.6),
        )

    # ------------------------------------------------------------------
    # Per-stage builders. Provider→plugin selection lives here. To add a new
    # provider for any stage, add a branch and update AgentConfig._validate.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm(cfg: "AgentConfig") -> "lk_llm.LLM | None":
        """Build the LLM (currently only OpenAI-compatible endpoints)."""
        try:
            from livekit.plugins import openai as lk_openai
        except ImportError:
            logger.error(
                "livekit-plugins-openai is not installed. "
                "Install with: pip install 'livekit-plugins-openai>=0.4'"
            )
            return None

        return lk_openai.LLM(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key or None,
            base_url=cfg.llm.base_url or None,
        )

    @staticmethod
    def _build_stt(cfg: "AgentConfig") -> SttStage:
        """Build the STT stage based on ``cfg.stt_provider``."""
        provider = cfg.stt_provider
        if provider == "bailian":
            from eidolon.channel.livekit.plugins.stt.bailian import BailianFunASRSTT

            plugin_cfg = cfg.bailian_stt
            stt = BailianFunASRSTT(config=plugin_cfg)
            params = SttParams(
                language=plugin_cfg.language,
                sample_rate=plugin_cfg.sample_rate,
                itn=plugin_cfg.itn,
            )
        elif provider == "sensetime":
            from eidolon.channel.livekit.plugins.stt.sensetime import SenseTimeSTT

            plugin_cfg = cfg.sensetime_stt
            stt = SenseTimeSTT(config=plugin_cfg)
            params = SttParams(
                language=plugin_cfg.language,
                sample_rate=plugin_cfg.sample_rate,
            )
        else:
            raise ValueError(
                f"Unknown STT provider: {provider!r} "
                f"(supported: 'bailian', 'sensetime')"
            )
        return SttStage(stt, params=params)

    @staticmethod
    def _build_tts(cfg: "AgentConfig") -> TtsStage:
        """Build the TTS stage based on ``cfg.tts_provider``."""
        provider = cfg.tts_provider
        if provider == "sensetime":
            from eidolon.channel.livekit.plugins.tts.sensetime import SenseTimeTTS

            plugin_cfg = cfg.sensetime_tts
            tts = SenseTimeTTS(config=plugin_cfg)
            params = TtsParams(
                sample_rate=plugin_cfg.sample_rate,
                speed=plugin_cfg.speed,
                voice=plugin_cfg.voice,
            )
        elif provider == "bailian":
            from eidolon.channel.livekit.plugins.tts.bailian import BailianTTS

            plugin_cfg = cfg.bailian_tts
            tts = BailianTTS(config=plugin_cfg)
            params = TtsParams(
                sample_rate=plugin_cfg.sample_rate,
                speed=plugin_cfg.speech_rate,
                voice=plugin_cfg.voice,
            )
        else:
            raise ValueError(
                f"Unknown TTS provider: {provider!r} "
                f"(supported: 'sensetime', 'bailian')"
            )
        return TtsStage(tts, params=params)

    @staticmethod
    def _build_vad(cfg: "AgentConfig") -> "lk_vad.VAD | None":
        """Build the VAD instance based on ``cfg.vad_provider``.

        Returns ``None`` for ``"none"`` / ``"disabled"`` providers, or when
        the chosen provider is unavailable in the runtime environment.
        """
        provider = cfg.vad_provider.lower()

        if provider in ("firered", "firered_pvad"):
            try:
                from eidolon.channel.livekit.plugins.vad.firered import FireredPvadVAD
                return FireredPvadVAD.load(
                    min_speech_duration=0.2,
                    min_silence_duration=0.5,
                    activation_threshold=0.55,
                )
            except Exception as e:
                logger.warning("[SharedStageFactory] FireRed VAD not available: %s", e)
                # Fall through to silero or None below

        if provider in ("none", "disabled"):
            return None

        try:
            from livekit.agents.plugins import silero_vad
            return silero_vad.VAD.load()
        except ImportError:
            logger.warning(
                "[SharedStageFactory] silero_vad not installed; VAD disabled"
            )
            return None
