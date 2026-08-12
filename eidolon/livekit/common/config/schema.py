"""Typed effective configuration for the LiveKit voice channel."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
InterruptionOwner = Literal["channel", "livekit_native_adaptive"]


@dataclass(frozen=True)
class CoreConfig:
    livekit_url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "devkey_secret"
    host: str = "0.0.0.0"
    port: int = 8766


@dataclass(frozen=True)
class AgentBehaviorConfig:
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

    Runtime tokens are resolved from :class:`RuntimeAuthorityConfig` and the
    LiveKit participant identity. This section only describes where the agent
    gRPC service lives and how to connect to it.
    """

    target: str = ""
    locale: str = "zh"
    conversation_id_prefix: str = "livekit"
    tls_mode: str = "off"
    tls_ca_path: str = ""
    tls_client_cert_path: str = ""
    tls_client_key_path: str = ""


@dataclass(frozen=True)
class RuntimeAuthorityConfig:
    """OS runtime authorities used before entering the Agent/audio path.

    Kernel owns Device Mount/Attachment. System Data owns Companion runtime
    facts. Channel signs only the selected Owner/Companion session context.

    ``enabled`` must remain True for ``eidolon_agent``.

    ``jwt_secret`` placeholder convention follows hub: the literal
    string ``PAIRING_JWT_SECRET`` in YAML means "read the env var of
    that name"; if both env and the host runtime secret file are empty,
    the resolver fails loud.
    """

    enabled: bool = True
    kernel_api_url: str = "http://127.0.0.1:8083/api/kernel/v1"
    data_api_url: str = "http://127.0.0.1:8084"
    data_service_token_env: str = "EIDOLON_DATA_COMPANION_AUTHORITY_TOKEN"
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    device_token_ttl_seconds: int = 24 * 3600
    http_timeout_sec: float = 10.0
    http_connect_timeout_sec: float = 3.0


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
    transcript_revision_min_normalized_chars: int = 4


@dataclass(frozen=True)
class InterruptPolicyConfig:
    decision_timeout_ms: int = 450
    post_speech_evidence_timeout_ms: int = 6_000
    post_speech_evidence_min_speech_ms: int = 250
    # Shorter cap for the post-speech evidence wait when NO transcript has
    # arrived at all: a real interruption yields a transcript quickly (interim
    # during speech, final within a few hundred ms of VAD end), so total silence
    # past this grace is almost certainly a false trigger and the agent should
    # resume promptly instead of staying suspended for the full evidence window.
    post_speech_no_evidence_timeout_ms: int = 800
    framework_false_interruption_timeout_ms: int = 6_000
    stt_commit_transcript_timeout_ms: int = 5_000
    aec_warmup_ms: int | None = 1_000
    min_interim_chars: int = 2
    early_cancel_score_threshold: float = 0.70
    early_resume_score_threshold: float = 0.20
    transcript_evidence_gate_enabled: bool = True
    min_normal_interim_cjk_chars: int = 3
    latin_artifact_hold_max_chars: int = 4
    hard_stop_prefix_min_cjk_chars: int = 2
    redirect_prefix_min_cjk_chars: int = 3
    repeated_noise_min_chars: int = 2
    repeated_noise_max_chars: int = 6
    weak_signal_followup_hold_ms: int = 1500
    correction_topic_stability_window_ms: int = 120
    normal_interrupt_stability_window_ms: int = 350
    cancel_residual_commit_suppress_ms: int = 2_000
    # Interrupt behavior flags (formerly the separate `interrupt_mode` axis,
    # collapsed into the single profile/config axis). Defaults = the stable
    # "balanced" behavior; the retired "responsive" mode flipped these on/off.
    fast_lexical_intents: bool = False
    stabilize_normal_interrupts: bool = True
    weak_signal_followup_hold: bool = True


@dataclass(frozen=True)
class PttPolicyConfig:
    # Half-duplex push-to-talk owns its turn boundary explicitly: press opens a
    # complete audio segment, release closes it, then the segment is transcribed
    # once before committing or rejecting the turn.
    segment_stt_strategy: str = "streaming"
    segment_min_audio_ms: int = 80
    segment_max_audio_ms: int = 20_000
    segment_min_rms_ppm: int = 0
    segment_tap_to_stop_max_audio_ms: int = 900


@dataclass(frozen=True)
class DuckingPolicyConfig:
    enabled: bool = True
    fade_out_ms: int = 30
    fade_in_ms: int = 30
    suspend_volume: float = 0.0
    suspended_passthrough_enabled: bool = False
    suspended_passthrough_volume: float = 0.25
    buffer_max_ms: int = 2_000
    cooldown_ms: int = 800


@dataclass(frozen=True)
class FillerPolicyConfig:
    enabled: bool = False
    phrases: tuple[str, ...] = ("嗯...", "好的...", "让我想想...")
    fade_in_ms: int = 30
    fade_out_ms: int = 80
    silence_lead_in_ms: int = 120


@dataclass(frozen=True)
class IdlePolicyConfig:
    followup_timeout_ms: int = 15_000
    # Hard-disconnect a session that has gone idle (no recognized speech and
    # no agent activity) for this long. Guards against a client that connects
    # and is never closed — STT keeps streaming audio (and billing) for the
    # whole connection even while silent. 0 (or negative) disables the
    # watchdog entirely. Default 60s. Applies to user_initiated sessions.
    disconnect_after_idle_ms: int = 60_000
    # A verified presence wake opens with the normal welcome, then returns to
    # standby quickly if the owner does not engage. Kept separate from the
    # proactive report window because their opening behavior is different.
    presence_disconnect_after_idle_ms: int = 15_000
    # Idle window for a proactive_initiated session (plan §3.3): a wake-up nobody
    # answers must be reclaimed quickly (I2/I6) rather than lingering on the
    # user_initiated 60s / half_duplex keep-alive. After this much silence a
    # proactive session ends gracefully (proactive_done). 0 disables.
    proactive_disconnect_after_idle_ms: int = 12_000
    # Grace between session_end notification and room deletion, so the reliable
    # data packet has a chance to reach the client before it is kicked.
    disconnect_grace_ms: int = 300


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
    soft_duck_on_playback_speech_start: bool = True
    ignore_when_mic_muted: bool = True
    echo_min_normalized_chars: int = 2
    assistant_speech_recent_max_age_ms: int = 3_000


@dataclass(frozen=True)
class TurnPolicyConfig:
    profile: TurnPolicyProfile = "balanced_semantic"
    interruption_owner: InterruptionOwner = "channel"
    vad: VadPolicyConfig = field(default_factory=VadPolicyConfig)
    eot: EotPolicyConfig = field(default_factory=EotPolicyConfig)
    interrupt: InterruptPolicyConfig = field(default_factory=InterruptPolicyConfig)
    ptt: PttPolicyConfig = field(default_factory=PttPolicyConfig)
    ducking: DuckingPolicyConfig = field(default_factory=DuckingPolicyConfig)
    filler: FillerPolicyConfig = field(default_factory=FillerPolicyConfig)
    idle: IdlePolicyConfig = field(default_factory=IdlePolicyConfig)
    preemptive: PreemptivePolicyConfig = field(default_factory=PreemptivePolicyConfig)
    attention: AttentionPolicyConfig = field(default_factory=AttentionPolicyConfig)


@dataclass(frozen=True)
class ObservabilityConfig:
    structured_logs: bool = True
    metrics_enabled: bool = True
    timeline_debug_path: str = ""
    llm_first_delta_timeout_ms: int = 3_000
    stt_pending_provider_event_window_ms: int = 2_000
    stt_pending_provider_event_preroll_ms: int = 500
    stt_pending_provider_event_max_count: int = 32


@dataclass(frozen=True)
class VoiceprintConfig:
    enabled: bool = True
    provider: str = "3d_speaker"
    model: str = "campplus_zh_16k_common"
    root: str = field(
        default_factory=lambda: str(
            Path(os.environ.get("EIDOLON_STATE_ROOT", "~/eidolon/data")).expanduser()
            / "voiceprints"
        )
    )
    model_dir: str = ""
    threshold: float = 0.31
    min_audio_ms: int = 1500
    prewarm: bool = True
    trust_paired_devices: bool = True
    turn_max_audio_ms: int = 12_000
    accept_cache_ttl_ms: int = 180_000
    accept_cache_short_audio_max_ms: int = 3_000
    owner_commit_threshold: float = 0.58
    owner_short_audio_bypass_ms: int = 1_500


@dataclass(frozen=True)
class WorkerConfig:
    # None preserves the historical runtime default:
    # dev=0, prod=min(cpu_count, 4). Set 1+ to force startup prewarm workers.
    num_idle_processes: int | None = None
    # How long a worker process may spend in setup (model loading) before the
    # framework kills it. The framework's own default is 10s, which assumes
    # cloud hardware; a Raspberry Pi 5 needs ~21s just for the pVAD import and
    # the EOT tokenizer, so that default puts the worker in a permanent
    # spawn-kill loop that never registers a single usable process. The budget
    # is paid once per process, so it should be generous enough to cover slow
    # hardware under load while still failing a genuinely stuck setup.
    setup_timeout_sec: float = 120.0


@dataclass(frozen=True)
class AvatarConfig:
    """Digital-human (talking-head) video avatar.

    ``enabled`` is a global availability/kill-switch; whether a given session
    actually shows a video avatar is decided per-connection at runtime from the
    joining participant's metadata (``resolve_avatar_requested``). Both default
    off, so audio-only sessions are unaffected.

    ``fps`` is the constant rate the decoder retimes video to (and the rate we
    request from the service); the service's *actual* emitted rate varies, so we
    resample to this fixed target to keep lip-sync under ``AVSynchronizer``.
    """

    enabled: bool = False
    service_url: str = ""
    width: int = 448
    height: int = 448
    fps: float = 25.0
    format: str = "mp4"
    output_sample_rate: int = 24000
    request_timeout_sec: float = 30.0
    connect_timeout_sec: float = 5.0
    first_frame_timeout_sec: float = 8.0
    fallback_to_audio_on_failure: bool = True
    avatar_agent_name: str = "eidolon-avatar"
    worker_identity_prefix: str = "avatar"
    # When true, seed the service with the session companion's configured
    # display face (``cond_image``) resolved from Eidolon Data. Falls back to
    # the service's own default avatar when unconfigured/unresolvable. Turn off
    # to always use the default avatar regardless of per-companion config.
    cond_image_enabled: bool = True
    # Streaming ingestion (``/ws/audio_stream``): feed TTS audio as it is produced
    # and progressively decode the returned fMP4 so the face starts within one
    # first-frame latency of the first audio chunk, instead of waiting for the
    # whole utterance (the batch ``/api/stream_video`` path). This is the default
    # ingestion mode for avatar sessions (best latency); only affects sessions
    # that already have the avatar on (gated by ``enabled`` + per-request
    # ``avatar``), so audio-only sessions are untouched. Set false to fall back
    # to the batch path. The service emits consistent-fps frames with PTS, so the
    # progressive decoder needs no manual retiming.
    streaming: bool = True
    # OPUS output frame length the service aligns fps to (2.5/5/10/20/40/60).
    stream_opus_frame_ms: float = 20.0
    # Pre-pad this many 16 kHz silence samples before the first real audio. The
    # service then reaches its first inference window sooner — but it pays for
    # that by emitting one early chunk and then *waiting*, punching a silence
    # hole into the middle of the speech. Measured against the live service with
    # a continuous tone: 8000 → first segment 1.83 s but a 0.75 s hole; 4000 →
    # 1.64 s / 1.00 s hole; 2000 → 1.44 s / 1.15 s hole; 0 → 2.67 s and no hole
    # at all. A gap inside a word is far worse than starting a beat later, so
    # this defaults off; raise it only if a deployment prefers the trade.
    stream_fast_start_samples: int = 0


@dataclass(frozen=True)
class EffectiveAgentConfig:
    core: CoreConfig = field(default_factory=CoreConfig)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    remote_agent_rpc: RemoteAgentRpcConfig = field(default_factory=RemoteAgentRpcConfig)
    runtime_authority: RuntimeAuthorityConfig = field(default_factory=RuntimeAuthorityConfig)
    turn_policy: TurnPolicyConfig = field(default_factory=TurnPolicyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    voiceprint: VoiceprintConfig = field(default_factory=VoiceprintConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)

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
