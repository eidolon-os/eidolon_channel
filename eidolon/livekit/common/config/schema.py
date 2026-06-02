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
    agent_mode: str = "streaming"
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
    target: str = ""
    locale: str = "zh"
    # ``device_token`` is the legacy static token (Phase 25 era). Phase
    # 32.B replaced it with a per-session token resolver — see
    # ``RuntimeAdminConfig`` below. Kept for now as a fallback path
    # when the resolver is disabled / unreachable; Phase 32.D will
    # remove it entirely once the resolver has been verified in prod.
    device_token: str = ""
    conversation_id_prefix: str = "livekit"
    tls_mode: str = "off"
    tls_ca_path: str = ""
    tls_client_cert_path: str = ""
    tls_client_key_path: str = ""


@dataclass(frozen=True)
class RuntimeAdminConfig:
    """Phase 32.B: channel resolves participant identity → user / agent
    / template by querying admin's ``/api/resolve`` aggregator, then
    signs a device JWT using the shared HMAC secret.

    ``enabled=false`` keeps the legacy code path (channel uses
    ``remote_agent_rpc.device_token`` statically). Defaults to True on
    fresh installs; set False if admin is down and you need channel to
    still bring up demo sessions.

    ``jwt_secret`` placeholder convention follows hub: the literal
    string ``PAIRING_JWT_SECRET`` in YAML means "read the env var of
    that name"; if both env and ~/eidolon/run/jwt-secret are empty,
    the resolver fails loud and channel falls back to the legacy
    static token (or refuses the session if that's also empty).
    """

    enabled: bool = True
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
    hard_stop_lexicon: tuple[str, ...] = (
        "停",
        "停一下",
        "别说了",
        "不要说了",
        "打住",
        "闭嘴",
        "先别讲了",
    )
    topic_switch_lexicon: tuple[str, ...] = (
        "换个话题",
        "不聊这个",
        "别聊这个",
        "说点别的",
        "聊点别的",
        "我们聊点",
        "刚才那个不用了",
    )
    correction_lexicon: tuple[str, ...] = (
        "不是",
        "等一下",
        "我不是这个意思",
        "我刚才",
        "我刚才说",
        "我刚才说错了",
    )


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
class TurnPolicyConfig:
    profile: TurnPolicyProfile = "balanced_semantic"
    vad: VadPolicyConfig = field(default_factory=VadPolicyConfig)
    eot: EotPolicyConfig = field(default_factory=EotPolicyConfig)
    interrupt: InterruptPolicyConfig = field(default_factory=InterruptPolicyConfig)
    ducking: DuckingPolicyConfig = field(default_factory=DuckingPolicyConfig)
    idle: IdlePolicyConfig = field(default_factory=IdlePolicyConfig)
    preemptive: PreemptivePolicyConfig = field(default_factory=PreemptivePolicyConfig)


@dataclass(frozen=True)
class ObservabilityConfig:
    structured_logs: bool = True
    metrics_enabled: bool = True
    timeline_debug_path: str = ""


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
            ("remote_agent_rpc", "device_token"),
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
