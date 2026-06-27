"""Typed effective configuration for the LiveKit voice channel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from eidolon.livekit.plugins.stt.bailian.config import BailianSTTConfig
from eidolon.livekit.plugins.stt.sensetime.config import SenseTimeSTTConfig
from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.sensetime.config import SenseTimeTTSConfig


BrainProvider = Literal["direct_llm", "eidolon_agent"]
TurnPolicyProfile = Literal[
    "fast_e2e_like",
    "balanced_semantic",
    "patient_companion",
    "custom",
]


@dataclass(frozen=True)
class CoreConfig:
    livekit_url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "devkey_secret"
    host: str = "0.0.0.0"
    port: int = 8766


@dataclass(frozen=True)
class AgentBehaviorConfig:
    pipeline_mode: str = "streaming"
    instructions: str = "You are a helpful, friendly voice assistant. Keep responses concise."
    welcome_message: str = "你好！我是你的 AI 助手，请问有什么可以帮你的？"
    audio_sample_rate: int = 16000


@dataclass(frozen=True)
class ProvidersConfig:
    stt_provider: str = "sensetime"
    tts_provider: str = "sensetime"
    vad_provider: str = "firered"
    brain_provider: BrainProvider = "direct_llm"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = ""
    model: str = "gpt-4o-mini"
    api_key: str = ""
    temperature: float | None = None
    timeout: float | None = None
    max_completion_tokens: int | None = None


@dataclass(frozen=True)
class RemoteAgentRpcConfig:
    """Static config for the gRPC channel to eidolon-agent.

    ``device_token`` used to live here (Phase 25 era — channel signed
    a single long-lived JWT into config/.env and used it for every
    session). Phase 32.B introduced per-session token resolution via
    admin's /api/resolve, and Phase 32.D (this commit) removed the
    static fallback entirely. The runtime token now ALWAYS comes from
    :class:`RuntimeAdminConfig` + the shared HMAC secret.

    Reasoning: keeping a static fallback invited the "everyone is
    alice" bug (the legacy token's payload pinned user_id=alice). 凡是
    保留的 fallback 都会被某次匆忙的运维拿来用,然后忘了关。删掉
    比记得关好。
    """

    target: str = ""
    locale: str = "zh"
    conversation_id_prefix: str = "livekit"
    tls_mode: str = "off"
    tls_ca_path: str = ""
    tls_client_cert_path: str = ""
    tls_client_key_path: str = ""


@dataclass(frozen=True)
class RuntimeAdminConfig:
    """Phase 32.B: channel resolves participant identity → user / agent
    / template from Eidolon Data, then signs a device JWT using the shared
    HMAC secret. Admin HTTP can remain as a cross-process fallback.

    ``enabled`` must remain True for ``eidolon_agent``. The legacy static
    ``remote_agent_rpc.device_token`` fallback was removed, so setting
    ``enabled=false`` now fails loudly instead of silently chatting with a
    stale or shared identity.

    ``jwt_secret`` placeholder convention follows hub: the literal
    string ``PAIRING_JWT_SECRET`` in YAML means "read the env var of
    that name"; if both env and ~/eidolon/run/jwt-secret are empty,
    the resolver fails loud; there is no static-token fallback.
    """

    enabled: bool = True
    data_resolve_enabled: bool = True
    admin_fallback_enabled: bool = True
    admin_api_url: str = "http://127.0.0.1:9000"
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    device_token_ttl_seconds: int = 24 * 3600


@dataclass(frozen=True)
class VadPolicyConfig:
    activation_threshold: float = 0.50
    prefix_padding_ms: int = 300
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 500
    max_buffered_speech_ms: int = 60_000


@dataclass(frozen=True)
class EotPolicyConfig:
    eot_unlikely_threshold: float = 0.50
    tail_hang_silence_ms: int = 2_000


@dataclass(frozen=True)
class InterruptPolicyConfig:
    decision_timeout_ms: int = 500
    min_interim_chars: int = 2
    early_cancel_score_threshold: float = 0.70
    early_resume_score_threshold: float = 0.20
    transcript_evidence_gate_enabled: bool = True
    min_normal_interim_cjk_chars: int = 3
    latin_artifact_hold_max_chars: int = 4
    weak_signal_followup_hold_ms: int = 1500
    correction_topic_stability_window_ms: int = 120
    normal_interrupt_stability_window_ms: int = 350
    # Interrupt behavior flags (formerly the separate `interrupt_mode` axis,
    # collapsed into the single profile/config axis). Defaults = the stable
    # "balanced" behavior; the retired "responsive" mode flipped these on/off.
    fast_lexical_intents: bool = False
    stabilize_normal_interrupts: bool = True
    weak_signal_followup_hold: bool = True


@dataclass(frozen=True)
class DuckingPolicyConfig:
    enabled: bool = True
    fade_out_ms: int = 30
    fade_in_ms: int = 30
    suspend_volume: float = 0.0
    buffer_max_ms: int = 2_000
    cooldown_ms: int = 800


@dataclass(frozen=True)
class IdlePolicyConfig:
    followup_timeout_ms: int = 15_000
    # Hard-disconnect a session that has gone idle (no recognized speech and
    # no agent activity) for this long. Guards against a client that connects
    # and is never closed — STT keeps streaming audio (and billing) for the
    # whole connection even while silent. 0 (or negative) disables the
    # watchdog entirely. Default 60s. Applies to user_initiated sessions.
    disconnect_after_idle_ms: int = 60_000
    # Idle window for a proactive_initiated session (plan §3.3): a wake-up nobody
    # answers must be reclaimed quickly (I2/I6) rather than lingering on the
    # user_initiated 60s / half_duplex keep-alive. After this much silence a
    # proactive session ends gracefully (proactive_done). 0 disables.
    proactive_disconnect_after_idle_ms: int = 12_000


@dataclass(frozen=True)
class PreemptivePolicyConfig:
    """Speculative brain generation (LiveKit native preemptive_generation).

    Starts the brain on a stable interim/preflight transcript before commit;
    the framework reuses it if the final transcript matches, else cancels via
    our gRPC CancelTurn. ``preemptive_tts`` keeps output gated by our commit
    when False (no partial-audio leak); only the LLM is pre-warmed.

    Default OFF: a real-room A/B (2026-05-31) showed it adds no measurable
    first-audio benefit once ``bailian_stt.max_sentence_silence_ms`` is low
    (~400ms) — the fast FINAL leaves nothing to overlap — while still wasting
    27-58% of speculative brain turns (clean cancels, but real upstream load).
    Enable it for deployments that must keep a high endpointing silence (slow
    FINAL), where the overlap pays off.
    """

    enabled: bool = False
    preemptive_tts: bool = False


@dataclass(frozen=True)
class AttentionPolicyConfig:
    enabled: bool = True
    enforce: bool = False
    client_state_max_age_ms: int = 2_000
    require_direct_signal_during_playback: bool = True
    ignore_when_mic_muted: bool = True


@dataclass(frozen=True)
class TurnPolicyConfig:
    profile: TurnPolicyProfile = "balanced_semantic"
    vad: VadPolicyConfig = field(default_factory=VadPolicyConfig)
    eot: EotPolicyConfig = field(default_factory=EotPolicyConfig)
    interrupt: InterruptPolicyConfig = field(default_factory=InterruptPolicyConfig)
    ducking: DuckingPolicyConfig = field(default_factory=DuckingPolicyConfig)
    idle: IdlePolicyConfig = field(default_factory=IdlePolicyConfig)
    preemptive: PreemptivePolicyConfig = field(default_factory=PreemptivePolicyConfig)
    attention: AttentionPolicyConfig = field(default_factory=AttentionPolicyConfig)


@dataclass(frozen=True)
class ObservabilityConfig:
    structured_logs: bool = True
    metrics_enabled: bool = True
    timeline_debug_path: str = ""


@dataclass(frozen=True)
class VoiceprintConfig:
    enabled: bool = True
    provider: str = "3d_speaker"
    model: str = "campplus_zh_16k_common"
    root: str = "~/eidolon/voiceprints"
    model_dir: str = ""
    threshold: float = 0.31
    min_audio_ms: int = 1500
    prewarm: bool = True
    trust_paired_devices: bool = True


@dataclass(frozen=True)
class WorkerConfig:
    # None preserves the historical runtime default:
    # dev=0, prod=min(cpu_count, 4). Set 1+ to force startup prewarm workers.
    num_idle_processes: int | None = None


@dataclass(frozen=True)
class EffectiveAgentConfig:
    core: CoreConfig = field(default_factory=CoreConfig)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    remote_agent_rpc: RemoteAgentRpcConfig = field(default_factory=RemoteAgentRpcConfig)
    runtime_admin: RuntimeAdminConfig = field(default_factory=RuntimeAdminConfig)
    turn_policy: TurnPolicyConfig = field(default_factory=TurnPolicyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)

    bailian_stt: BailianSTTConfig = field(default_factory=BailianSTTConfig)
    sensetime_stt: SenseTimeSTTConfig = field(default_factory=SenseTimeSTTConfig)
    bailian_tts: BailianTTSConfig = field(default_factory=BailianTTSConfig)
    sensetime_tts: SenseTimeTTSConfig = field(default_factory=SenseTimeTTSConfig)

    @property
    def stt_provider(self) -> str:
        return self.providers.stt_provider

    @property
    def tts_provider(self) -> str:
        return self.providers.tts_provider

    @property
    def vad_provider(self) -> str:
        return self.providers.vad_provider

    def sanitized_dict(self) -> dict:
        data = asdict(self)
        for path in (
            ("core", "api_secret"),
            ("llm", "api_key"),
            ("bailian_stt", "api_key"),
            ("sensetime_stt", "api_key"),
            ("bailian_tts", "api_key"),
            ("sensetime_tts", "api_key"),
        ):
            cur = data
            for key in path[:-1]:
                cur = cur.get(key, {})
            if path[-1] in cur and cur[path[-1]]:
                cur[path[-1]] = "***"
        return data


# Short-term import stability for existing code/tests while the runtime moves
# to the EffectiveAgentConfig name.
AgentConfig = EffectiveAgentConfig
