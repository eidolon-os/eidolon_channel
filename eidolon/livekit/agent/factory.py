"""Voice stage factory — constructs and owns the STT/LLM/TTS/VAD stage stack.

Both half-duplex and full-duplex pipelines receive the same factory instance,
so they share the underlying model connections and stages.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents import llm as lk_llm
    from livekit.agents import vad as lk_vad

    from eidolon.livekit.common.config import AgentConfig

from .providers.llm import LlmParams, LivekitLlmStage
from .providers.stt import SttParams, SttStage
from .providers.tts import TtsParams, TtsStage
from .providers.vad import VadStage

logger = logging.getLogger("agent")


@dataclass(frozen=True)
class RealtimeStageBundle:
    """STT/TTS/VAD stages that can be built without a room-bound LLM."""

    stt: SttStage
    tts: TtsStage
    vad: VadStage | None = None


def _build_device_token_source(
    *,
    cfg: "AgentConfig",
    livekit_room: "Any | None",
) -> "Any":
    """Build the per-session device-token resolver used by the gRPC LLM.

    The only token source is the runtime resolver — which reads
    ``participant.identity`` from the LiveKit room, resolves that explicit
    device binding, and signs a JWT with the shared HMAC secret. If any prerequisite is missing (secret
    absent, no room, runtime_admin disabled) we raise instead of
    silently using "alice" — the operator must fix the config rather
    than ship the wrong identity.

    Returns a zero-arg async callable (resolver). Raises ``RuntimeError``
    when prerequisites are missing; the caller surfaces it through
    log + session abort.
    """
    rt = cfg.runtime_admin

    if not rt.enabled:
        # Pre-32.D, this branch returned a static legacy token. Now we
        # refuse — the only way to disable runtime resolution is to
        # rewrite the call site, which forces a code review.
        raise RuntimeError(
            "[device_token] runtime_admin.enabled=false but the static "
            "fallback was removed in Phase 32.D. Set runtime_admin.enabled=true "
            "and ensure PAIRING_JWT_SECRET (or ~/eidolon/run/jwt-secret) "
            "is reachable."
        )

    # Resolve secret: env (loaded via _secret() at config time) →
    # ~/eidolon/run/jwt-secret (shared with eidolon-agent).
    from eidolon_sdk.biz.runtime import resolve_shared_secret

    secret = resolve_shared_secret(rt.jwt_secret)
    if not secret:
        raise RuntimeError(
            "[device_token] PAIRING_JWT_SECRET empty and "
            "~/eidolon/run/jwt-secret missing. Start eidolon-agent once "
            "so it persists the secret, or set the env var explicitly."
        )

    if livekit_room is None:
        # In production server.run_agent always passes ``ctx.room``,
        # so this is a test-path guard. Tests that exercise the full
        # gRPC LLM should use the resolver test doubles in
        # eidolon/livekit/tests/agent/runtime/ instead of building the
        # factory directly.
        raise RuntimeError(
            "[device_token] livekit_room is None — required to read "
            "participant identity. Pass ``livekit_room=ctx.room`` from "
            "the agent entrypoint."
        )

    resolve_client = _build_runtime_resolve_client(rt)
    from eidolon.livekit.agent.runtime import make_device_token_resolver

    return make_device_token_resolver(
        room=livekit_room,
        admin=resolve_client,
        jwt_secret=secret,
        jwt_algorithm=rt.jwt_algorithm,
        ttl_seconds=rt.device_token_ttl_seconds,
    )


def _build_runtime_resolve_client(rt: "Any") -> "Any":
    """Build local Eidolon Data resolver with optional admin HTTP fallback."""
    from eidolon.livekit.common.config import RuntimeAdminConfig

    runtime_defaults = RuntimeAdminConfig()

    local = None
    if getattr(rt, "data_resolve_enabled", True):
        try:
            from eidolon_data import DataStore
            from eidolon_data import load_settings as load_data_settings

            data_settings = load_data_settings()
            sqlite_path = Path(data_settings.sqlite_path).expanduser()
            if sqlite_path.exists():
                local_store = DataStore.open(data_settings)
                local = _DataStoreRuntimeResolveClient(local_store)
                logger.info("[device_token] using Eidolon Data resolver at %s", sqlite_path)
            else:
                logger.warning(
                    "[device_token] Eidolon Data SQLite not found at %s; "
                    "runtime resolve will use admin fallback if enabled",
                    sqlite_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[device_token] Eidolon Data resolver unavailable: %s", exc)

    http = None
    if getattr(rt, "admin_fallback_enabled", True):
        import httpx
        from eidolon_sdk.biz.admin import AdminResolveClient

        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                float(
                    getattr(
                        rt,
                        "http_timeout_sec",
                        runtime_defaults.http_timeout_sec,
                    )
                ),
                connect=float(
                    getattr(
                        rt,
                        "http_connect_timeout_sec",
                        runtime_defaults.http_connect_timeout_sec,
                    )
                ),
            ),
            trust_env=False,  # avoid macOS Clash :7890 hijacking loopback
        )
        http = AdminResolveClient(http_client, rt.admin_api_url)

    if local is None and http is None:
        raise RuntimeError(
            "[device_token] no runtime resolve client configured. Enable "
            "runtime_admin.data_resolve_enabled or runtime_admin.admin_fallback_enabled."
        )
    if local is None:
        return http
    if http is None:
        return local
    return _FallbackResolveClient(primary=local, fallback=http)


class _FallbackResolveClient:
    def __init__(self, *, primary: "Any", fallback: "Any") -> None:
        self._primary = primary
        self._fallback = fallback

    async def resolve_device(self, device_id: str):
        return await self._resolve("resolve_device", device_id)

    async def resolve_owner(self, owner_id: str):
        return await self._resolve("resolve_owner", owner_id)

    async def _resolve(self, method: str, value: str):
        from eidolon_sdk.biz.admin import AdminResolveNotFound, AdminResolveUnreachable

        try:
            return await getattr(self._primary, method)(value)
        except (AdminResolveNotFound, AdminResolveUnreachable) as exc:
            logger.warning(
                "[device_token] Eidolon Data %s(%r) failed (%s); using admin fallback",
                method,
                value,
                exc,
            )
            return await getattr(self._fallback, method)(value)


class _DataStoreRuntimeResolveClient:
    """Local eidolon_data implementation of AdminResolveClient's resolve API."""

    def __init__(self, store: "Any") -> None:
        self._store = store

    async def resolve_owner(self, owner_id: str):
        from eidolon_sdk.biz.admin import (
            AdminResolveNotFound,
            AdminResolvePrecondition,
        )

        owner = await self._store.owners.get(owner_id)
        if owner is None:
            raise AdminResolveNotFound(f"owner {owner_id!r} is not registered in eidolon_data")
        if owner.status != "active":
            raise AdminResolvePrecondition(412, f"owner {owner_id!r} is {owner.status}")

        companions = [
            row
            for row in await self._store.companions.list_for_owner(owner_id)
            if row.status == "active"
        ]
        ready = [row for row in companions if row.default_memory_realm_id and row.current_genome_id]
        if not ready:
            raise AdminResolvePrecondition(
                412,
                f"owner {owner_id!r} has no active companion with memory/genome",
            )
        return await self._context_for_companion(ready[0], device_id=None)

    async def resolve_device(self, device_id: str):
        from eidolon_sdk.biz.admin import (
            AdminResolveNotFound,
            AdminResolvePrecondition,
        )

        device = await self._store.devices.get_device(device_id)
        if device is None:
            raise AdminResolveNotFound(f"device {device_id!r} is not registered in eidolon_data")
        if not device.owner_id:
            raise AdminResolvePrecondition(412, f"device {device_id!r} is not claimed")
        if device.status in {"disabled", "revoked"}:
            raise AdminResolvePrecondition(412, f"device {device_id!r} is {device.status}")
        if not device.bound_companion_id:
            raise AdminResolvePrecondition(412, f"device {device_id!r} is not bound to a companion")

        companion = await self._store.companions.get(device.bound_companion_id)
        if companion is None:
            raise AdminResolveNotFound(f"companion {device.bound_companion_id!r} not found")
        if companion.owner_id != device.owner_id:
            raise AdminResolvePrecondition(
                412,
                f"device {device_id!r} is bound outside owner {device.owner_id!r}",
            )
        return await self._context_for_companion(companion, device_id=device.device_id)

    async def _context_for_companion(self, companion: "Any", *, device_id: str | None):
        from eidolon_sdk.biz.admin import (
            AdminResolveNotFound,
            AdminResolvePrecondition,
            ResolvedContext,
        )

        if companion.status != "active":
            raise AdminResolvePrecondition(
                412,
                f"companion {companion.companion_id!r} is {companion.status}",
            )
        if not companion.default_memory_realm_id:
            raise AdminResolvePrecondition(
                412,
                f"companion {companion.companion_id!r} has no default memory realm",
            )
        if not companion.current_genome_id:
            raise AdminResolvePrecondition(
                412,
                f"companion {companion.companion_id!r} has no current genome",
            )
        realm = await self._store.memory_repo.get_realm(companion.default_memory_realm_id)
        if realm is None:
            raise AdminResolveNotFound(
                f"memory realm {companion.default_memory_realm_id!r} not found"
            )
        if realm.status != "active":
            raise AdminResolvePrecondition(
                412,
                f"memory realm {companion.default_memory_realm_id!r} is {realm.status}",
            )
        genome = await self._store.persona_repo.get_genome(companion.current_genome_id)
        if genome is None:
            raise AdminResolveNotFound(f"genome {companion.current_genome_id!r} not found")
        if genome.companion_id != companion.companion_id:
            raise AdminResolvePrecondition(
                412,
                f"genome {genome.genome_id!r} belongs to companion "
                f"{genome.companion_id!r}, not {companion.companion_id!r}",
            )
        if genome.status != "committed":
            raise AdminResolvePrecondition(
                412,
                f"genome {genome.genome_id!r} is {genome.status}",
            )
        if not genome.genome_hash:
            raise AdminResolvePrecondition(
                412,
                f"genome {genome.genome_id!r} has no genome_hash",
            )
        return ResolvedContext(
            owner_id=companion.owner_id,
            companion_id=companion.companion_id,
            memory_realm_id=companion.default_memory_realm_id,
            genome_id=companion.current_genome_id,
            schema_version=genome.schema_version,
            genome_hash=genome.genome_hash,
            realizer_version=genome.realizer_version,
            device_id=device_id,
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
        voiceprint_provider: "Any | None" = None,
        voiceprint_trust_paired_devices: bool = True,
        runtime_admin: "Any | None" = None,
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
        self.runtime_admin = runtime_admin
        self.voiceprint_provider = voiceprint_provider
        self.voiceprint_trust_paired_devices = voiceprint_trust_paired_devices
        self.voiceprint_service = None
        if voiceprint_provider is not None:
            from eidolon.livekit.agent.speaker_verification import (
                SpeakerVerificationService,
                VoiceprintStore,
            )

            self.voiceprint_service = SpeakerVerificationService(
                provider=voiceprint_provider,
                store=VoiceprintStore(
                    getattr(
                        voiceprint_provider,
                        "voiceprint_root",
                        "~/eidolon/voiceprints",
                    )
                ),
            )

        logger.info(
            "[SharedStageFactory] initialized stt=%s llm=%s tts=%s vad=%s voiceprint=%s",
            type(self.stt).__name__,
            type(self.llm).__name__,
            type(self.tts).__name__,
            type(self.vad).__name__ if self.vad else None,
            type(self.voiceprint_provider).__name__ if self.voiceprint_provider else None,
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
        prebuilt_voiceprint_provider: "Any | None" = None,
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
            prebuilt_voiceprint_provider: If supplied from worker prewarm,
                this CAMPPlus provider is reused across jobs instead of
                cold-loading the voiceprint model per room.
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
                room_name_static = getattr(room_ref, "name", None) or session_key

                def _resolve_cid() -> str:
                    try:
                        from eidolon.livekit.agent.runtime import resolve_device_id

                        participants = list(getattr(room_ref, "remote_participants", {}).values())
                        room_name = getattr(room_ref, "name", None) or room_name_static
                        if participants:
                            p = participants[0]
                            ident = getattr(p, "identity", "") or "anon"
                            # Voice rooms are per-session (device-<id>-<nonce>):
                            # keying the conversation on the volatile room name
                            # would start a fresh brain context on every JOIN. For
                            # a device, key on the stable participant identity
                            # alone (its LiveKit identity is constant across
                            # reconnects and single-format) so every JOIN from the
                            # same device continues one conversation. device_id is
                            # used only to DETECT a device — don't fold it into the
                            # key too: it can arrive in a different MAC spelling
                            # than `ident` (colon vs hyphen) and would split one
                            # device's history across two keys. Web/other
                            # participants carry no device_id and keep the
                            # room-scoped id.
                            is_device = resolve_device_id(getattr(p, "metadata", None)) is not None
                            if is_device:
                                return f"{prefix}:{ident}"
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

            # Phase 32.D: device_token is always the per-session resolver
            # (the static fallback was deleted). _build_device_token_source
            # raises RuntimeError if prerequisites are missing — operator
            # sees a clear failure rather than silently chatting as "alice".
            device_token_source = _build_device_token_source(cfg=cfg, livekit_room=livekit_room)

            llm = EidolonAgentGrpcLlm(
                target=cfg.remote_agent_rpc.target,
                device_token=device_token_source,
                conversation_id=conversation_id,
                display_model=cfg.llm.model or "eidolon_agent",
                tls=tls,
            )
            logger.info(
                "[SharedStageFactory] using EidolonAgentGrpcLlm target=%r "
                "conversation_id=%s tls_mode=%s device_token=<resolver>",
                cfg.remote_agent_rpc.target,
                log_cid_descr,
                cfg.remote_agent_rpc.tls_mode,
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
            voiceprint_provider=prebuilt_voiceprint_provider,
            voiceprint_trust_paired_devices=cfg.voiceprint.trust_paired_devices,
            runtime_admin=cfg.runtime_admin,
            llm_params=LlmParams(
                model=cfg.llm.model,
                temperature=cfg.llm.temperature or 0.6,
            ),
        )

    @classmethod
    def components_from_config(
        cls,
        cfg: "AgentConfig",
        *,
        prebuilt_vad: "lk_vad.VAD | None" = None,
    ) -> RealtimeStageBundle:
        """Build STT/TTS/VAD without initializing LLM or device-token state.

        Component benchmarks and other audio-only tools should not need a
        LiveKit room. Keeping this path separate prevents room-scoped brain
        wiring from leaking into realtime audio component checks.
        """
        stt = cls._build_stt(cfg)
        tts = cls._build_tts(cfg)
        raw_vad = prebuilt_vad if prebuilt_vad is not None else cls._build_vad(cfg)
        return RealtimeStageBundle(
            stt=stt,
            tts=tts,
            vad=VadStage(raw_vad) if raw_vad is not None else None,
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
                f"Unknown STT provider: {provider!r} (supported: 'bailian', 'sensetime')"
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
                f"Unknown TTS provider: {provider!r} (supported: 'sensetime', 'bailian')"
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
            logger.warning("[SharedStageFactory] silero_vad not installed; VAD disabled")
            return None
