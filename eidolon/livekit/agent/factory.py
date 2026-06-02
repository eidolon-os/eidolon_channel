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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents import llm as lk_llm
    from livekit.agents import vad as lk_vad

    from eidolon.livekit.common.config import AgentConfig

from .pipeline.llm import LlmParams, LivekitLlmStage
from .pipeline.stt import SttParams, SttStage
from .pipeline.tts import TtsParams, TtsStage
from .pipeline.vad import VadStage

logger = logging.getLogger("agent")


def _build_device_token_source(
    *,
    cfg: "AgentConfig",
    livekit_room: "Any | None",
) -> "Any":
    """Phase 32.B: pick a device_token source for the gRPC LLM.

    Returns one of:
      - a zero-arg async callable (production / plan D) when
        runtime_admin is enabled, has a usable secret, AND we hold a
        LiveKit room ref to read participant identity from
      - a static string (legacy Phase 25 fallback) otherwise — equal
        to ``cfg.remote_agent_rpc.device_token``

    Always falls back gracefully rather than refusing to build: a
    misconfigured runtime_admin block shouldn't take channel down,
    operators get a clear warning in the log instead. Phase 32.D will
    remove the static fallback once 32.B is verified.
    """
    rt = cfg.runtime_admin
    legacy = (cfg.remote_agent_rpc.device_token or "").strip()

    if not rt.enabled:
        logger.info(
            "[device_token] runtime_admin disabled → using legacy static "
            "token from remote_agent_rpc.device_token"
        )
        return legacy

    # Resolve secret: env (loaded via _secret() at config time) →
    # ~/eidolon/run/jwt-secret (shared with eidolon-agent).
    from eidolon.livekit.agent.runtime.token_signer import resolve_shared_secret
    secret = resolve_shared_secret(rt.jwt_secret)
    if not secret:
        logger.warning(
            "[device_token] PAIRING_JWT_SECRET empty and "
            "~/eidolon/run/jwt-secret missing — falling back to legacy "
            "static token. Start eidolon-agent once so it persists the "
            "secret, or set the env var explicitly."
        )
        return legacy

    if livekit_room is None:
        logger.warning(
            "[device_token] runtime_admin enabled but no LiveKit room "
            "reference — falling back to legacy static token. (This path "
            "is mostly hit by tests; production passes livekit_room.)"
        )
        return legacy

    # Build the resolver. The httpx client lives for the resolver's
    # lifetime — same lifetime as the LLM, which is the LK job. We
    # close it lazily; one channel-worker handles one job at a time
    # so leaks are bounded.
    import httpx
    from eidolon.livekit.agent.runtime import (
        AdminResolveClient,
        make_device_token_resolver,
    )

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0),
        trust_env=False,  # avoid macOS Clash :7890 hijacking loopback
    )
    admin = AdminResolveClient(http_client, rt.admin_api_url)
    return make_device_token_resolver(
        room=livekit_room,
        admin=admin,
        jwt_secret=secret,
        jwt_algorithm=rt.jwt_algorithm,
        ttl_seconds=rt.device_token_ttl_seconds,
    )


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
        livekit_session_key: str = "",
        livekit_room: "Any | None" = None,
    ) -> "SharedStageFactory":
        """Build all stages from the agent's configuration.

        Args:
            cfg: The :class:`AgentConfig` (typically loaded from .env).
            prebuilt_vad: If supplied (e.g. from worker prewarm), this VAD
                instance is used instead of building a fresh one. Useful when
                a worker process loads the VAD model once at startup and
                reuses it across jobs.
            livekit_session_key: When ``REMOTE_AGENT_RPC_TARGET`` is set and
                ``livekit_room`` is None, used as the static brain-side
                ``conversation_id`` as ``<prefix>:<session_key>``. Mainly for
                tests; production should pass ``livekit_room`` instead.
                Falls back to ``"unknown"`` when empty.
            livekit_room: D1 (plan Phase D) — pass the LiveKit ``Room`` so the
                factory builds a *lazy* ``conversation_id`` resolver that
                includes the first remote participant's identity. The
                resolver runs on every chat() call — by then participants
                have connected (session.start() has run, the user has spoken).
                Composes as ``<prefix>:<participant_identity>:<room.name>`` so
                the brain isolates history per (user, session) and a fresh
                room name no longer triggers a cold start for an already-warm
                user. Falls back to ``<prefix>:<room.name>`` when no
                participant is connected yet (defensive — shouldn't happen at
                chat() time since the user has already spoken).

        Returns:
            A fully wired :class:`SharedStageFactory`.

        Raises:
            ValueError: If the LLM cannot be built (missing dependency or
                config) — STT/TTS provider mismatches surface from
                ``cfg._validate()`` at config-load time.
        """
        if cfg.providers.brain_provider == "eidolon_agent":
            from eidolon.livekit.agent.eidolon_agent_rpc import (
                EidolonAgentGrpcLlm,
            )
            from eidolon.livekit.agent.eidolon_agent_rpc.session import TlsConfig

            prefix = cfg.remote_agent_rpc.conversation_id_prefix
            session_key = livekit_session_key.strip() or "unknown"

            # D1: conversation_id is either lazy (when we hold a room ref —
            # production path) or static (tests / fallback when called without
            # a room). The resolver runs once per chat() call; cost is a dict
            # lookup + a format string, no caching needed.
            if livekit_room is not None:
                room_ref = livekit_room  # closed over by the resolver
                room_name_static = (
                    getattr(room_ref, "name", None) or session_key
                )

                def _resolve_cid() -> str:
                    try:
                        participants = list(
                            getattr(room_ref, "remote_participants", {}).values()
                        )
                        room_name = (
                            getattr(room_ref, "name", None) or room_name_static
                        )
                        if participants:
                            ident = getattr(participants[0], "identity", "") or "anon"
                            return f"{prefix}:{ident}:{room_name}"
                        # Defensive — no participant yet (very early in job
                        # lifecycle, before user speech). Brain will still
                        # accept; once the participant connects, subsequent
                        # turns get the identity-bearing id.
                        return f"{prefix}:{room_name}"
                    except Exception:
                        # Never let a resolver bug block a turn.
                        return f"{prefix}:{room_name_static}"

                conversation_id: "str | Any" = _resolve_cid
                log_cid_descr = f"<lazy:participant + {room_name_static}>"
            else:
                conversation_id = f"{prefix}:{session_key}"
                log_cid_descr = conversation_id

            tls = TlsConfig(
                mode=cfg.remote_agent_rpc.tls_mode,
                ca_path=cfg.remote_agent_rpc.tls_ca_path,
                client_cert_path=cfg.remote_agent_rpc.tls_client_cert_path,
                client_key_path=cfg.remote_agent_rpc.tls_client_key_path,
            )

            # Phase 32.B: pick the device_token source.
            #   * runtime_admin.enabled + room ref + secret resolvable
            #     → per-session resolver (production / plan D)
            #   * else: static cfg.remote_agent_rpc.device_token
            #     (legacy fallback, will be deleted in 32.D)
            device_token_source = _build_device_token_source(
                cfg=cfg, livekit_room=livekit_room
            )

            llm = EidolonAgentGrpcLlm(
                target=cfg.remote_agent_rpc.target,
                device_token=device_token_source,
                conversation_id=conversation_id,
                display_model=cfg.llm.model or "eidolon_agent",
                tls=tls,
            )
            logger.info(
                "[SharedStageFactory] using EidolonAgentGrpcLlm target=%r "
                "conversation_id=%s tls_mode=%s device_token=%s",
                cfg.remote_agent_rpc.target,
                log_cid_descr,
                cfg.remote_agent_rpc.tls_mode,
                "<resolver>" if callable(device_token_source) else "<static>",
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
            llm_params=LlmParams(
                model=cfg.llm.model,
                temperature=cfg.llm.temperature or 0.6,
            ),
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

        # G4 (2026-05-16): pass tunables through only if the user actually
        # set them. None → omit the kwarg entirely so framework's NOT_GIVEN
        # default applies. (Mixing NOT_GIVEN/None semantics with **kwargs
        # filtering keeps us decoupled from the plugin's sentinel API.)
        extra_kwargs: dict[str, Any] = {}
        if cfg.llm.temperature is not None:
            extra_kwargs["temperature"] = cfg.llm.temperature
        if cfg.llm.timeout is not None:
            import httpx

            extra_kwargs["timeout"] = httpx.Timeout(cfg.llm.timeout)
        if cfg.llm.max_completion_tokens is not None:
            extra_kwargs["max_completion_tokens"] = cfg.llm.max_completion_tokens

        return lk_openai.LLM(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key or None,
            base_url=cfg.llm.base_url or None,
            **extra_kwargs,
        )

    @staticmethod
    def _build_stt(cfg: "AgentConfig") -> SttStage:
        """Build the STT stage based on ``cfg.stt_provider``."""
        provider = cfg.providers.stt_provider
        if provider == "bailian":
            from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT

            plugin_cfg = cfg.bailian_stt
            stt = BailianFunASRSTT(config=plugin_cfg)
            params = SttParams(
                language=plugin_cfg.language,
                sample_rate=plugin_cfg.sample_rate,
                itn=plugin_cfg.itn,
            )
        elif provider == "sensetime":
            from eidolon.livekit.plugins.stt.sensetime import SenseTimeSTT

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
        provider = cfg.providers.tts_provider
        if provider == "sensetime":
            from eidolon.livekit.plugins.tts.sensetime import SenseTimeTTS

            plugin_cfg = cfg.sensetime_tts
            tts = SenseTimeTTS(config=plugin_cfg)
            params = TtsParams(
                sample_rate=plugin_cfg.sample_rate,
                speed=plugin_cfg.speed,
                voice=plugin_cfg.voice,
            )
        elif provider == "bailian":
            from eidolon.livekit.plugins.tts.bailian import BailianTTS

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
        provider = cfg.providers.vad_provider.lower()
        vad_cfg = cfg.turn_policy.vad

        if provider in ("firered", "firered_pvad"):
            try:
                from eidolon.livekit.plugins.vad.firered import FireredPvadVAD
                return FireredPvadVAD.load(
                    min_speech_duration=vad_cfg.min_speech_duration_ms / 1000.0,
                    min_silence_duration=vad_cfg.min_silence_duration_ms / 1000.0,
                    prefix_padding_duration=vad_cfg.prefix_padding_ms / 1000.0,
                    max_buffered_speech=vad_cfg.max_buffered_speech_ms / 1000.0,
                    activation_threshold=vad_cfg.activation_threshold,
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
