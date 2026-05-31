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
    device_token: str = ""
    conversation_id_prefix: str = "livekit"
    tls_mode: str = "off"
    tls_ca_path: str = ""
    tls_client_cert_path: str = ""
    tls_client_key_path: str = ""


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
    """

    enabled: bool = True
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
