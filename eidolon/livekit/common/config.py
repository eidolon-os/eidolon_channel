"""Shared configuration for the LiveKit voice agent.

Composed of small, per-concern config dataclasses:

- :class:`CoreConfig` — LiveKit connection + agent server identity
- :class:`AgentBehaviorConfig` — pipeline mode, system instructions
- :class:`LLMConfig` — LLM provider config (OpenAI-compatible)
- Per-provider plugin configs — :class:`BailianSTTConfig`, :class:`SenseTimeSTTConfig`,
  :class:`SenseTimeTTSConfig` — each owns its own env-var loading via
  ``field(default_factory=lambda: os.environ.get(...))``

The top-level :class:`AgentConfig` aggregates these by composition. The
``stt_provider`` / ``tts_provider`` / ``vad_provider`` fields decide which
plugin config is actually used at runtime; the rest are pre-loaded so users
can switch providers via env var without restarting the agent.

Loading order (highest priority first):
1. Existing ``os.environ`` entries (shell, container, process manager, etc.)
2. Values merged from the file at ``EIDOLON_CHANNEL_LIVEKIT_ENV`` (required;
   ``load_dotenv(..., override=False)`` so existing env entries win over file)
3. Code defaults declared on each config dataclass

``EIDOLON_CHANNEL_LIVEKIT_ENV`` must name an **existing regular file**; otherwise
:func:`AgentConfig.from_env` raises ``ValueError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from eidolon.livekit.plugins.stt.bailian.config import BailianSTTConfig
from eidolon.livekit.plugins.stt.sensetime.config import SenseTimeSTTConfig
from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig
from eidolon.livekit.plugins.tts.sensetime.config import SenseTimeTTSConfig


def _bootstrap_dotenv() -> None:
    """Load the env file at ``EIDOLON_CHANNEL_LIVEKIT_ENV`` into ``os.environ``.

    Raises:
        ValueError: If ``EIDOLON_CHANNEL_LIVEKIT_ENV`` is unset/empty or the
            path is not an existing regular file.

    Idempotent and non-destructive: existing ``os.environ`` entries take
    priority (host environment > .env file on disk). Plugin configs
    (``BailianSTTConfig`` etc.) read directly from ``os.environ`` via their
    own ``default_factory``, so they pick up these values automatically once
    bootstrap has run.
    """
    from dotenv import load_dotenv

    raw = os.environ.get("EIDOLON_CHANNEL_LIVEKIT_ENV")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "EIDOLON_CHANNEL_LIVEKIT_ENV is not set or is empty. "
            "Set it to the path of your LiveKit channel env file (for example "
            "deploy/.livekit-channel.env)."
        )

    env_path = Path(str(raw).strip())
    if not env_path.is_file():
        raise ValueError(
            f"EIDOLON_CHANNEL_LIVEKIT_ENV={str(env_path)!r} must point to an "
            f"existing regular file."
        )

    load_dotenv(env_path, override=False)


# G4 (2026-05-16): tiny helpers for "env var unset/empty → None" semantics.
# Used by LLMConfig fields where ``None`` means "let the underlying plugin
# decide its own default" (NOT_GIVEN passthrough).
def _optional_float(raw: str) -> float | None:
    s = (raw or "").strip()
    return float(s) if s else None


def _optional_int(raw: str) -> int | None:
    s = (raw or "").strip()
    return int(s) if s else None


_DEFAULT_INSTRUCTIONS = (
    "You are a helpful, friendly voice assistant. Keep responses concise."
)

# Round 8 R8.9 — fixed welcome message, played via ``session.say()`` on
# agent enter rather than going through LLM with empty context. Avoids the
# previously-observed bug where LLM with only a system prompt would echo
# back internal instruction templates.
_DEFAULT_WELCOME = "你好！我是你的 AI 助手，请问有什么可以帮你的？"


@dataclass
class CoreConfig:
    """LiveKit connection + agent server identity."""

    livekit_url: str = "ws://localhost:7880"
    api_key: str = "devkey"
    api_secret: str = "devkey_secret"
    host: str = "0.0.0.0"
    port: int = 8766


@dataclass
class AgentBehaviorConfig:
    """Agent runtime behavior tunables."""

    agent_mode: str = "streaming"  # "streaming" | "batch"
    instructions: str = _DEFAULT_INSTRUCTIONS
    # Round 8 R8.9: fixed welcome played via ``session.say(welcome_message)``
    # on agent enter. Set to empty string to disable welcome.
    welcome_message: str = _DEFAULT_WELCOME
    # Round 8 R8.9: framework's ``false_interruption_timeout`` (default 2.0s).
    # When ≥ this many seconds elapse between VAD start_of_speech and an
    # actual STT transcript, framework treats the interrupt as false and
    # resumes the agent's speech. Chinese STT (SenseAudio) frequently takes
    # 3-5s to deliver final transcripts, which beats the default and makes
    # real interrupts misclassified. We bump to 6.0s. Set to None to
    # disable resumption (any VAD start permanently halts the agent until
    # next user transcript).
    false_interruption_timeout: float | None = 6.0

    # G9 fix (2026-05-17): framework's ``AgentSession(aec_warmup_duration=3.0)``
    # disables interruptions for the first N seconds of agent speech to
    # prevent echo from triggering false interrupts. Default 3.0s only
    # covers ~2/3 of a typical 23-char Chinese welcome — the remainder is
    # quietly interruptible, surprising users. Expose as env so deployments
    # can pick: 0 / None = always interruptible (power-user); 0.5-1.0 =
    # brief protect; longer = full welcome protected. ``None`` is the
    # "disable" sentinel per framework.
    aec_warmup_duration: float | None = 1.0

    # F1 fix (2026-05-16): framework's ``commit_user_turn(transcript_timeout=2.0)``
    # default was too short for Bailian FunASR FINAL latency on long Chinese
    # sentences (observed 2.0s+ end-to-end). The framework promoted the latest
    # INTERIM to FINAL, fired a doomed LLM call, then cancelled it when the real
    # FINAL arrived 100-500ms later. Bump to 5.0s; the framework's own EOT
    # detection still gates the actual turn commit, so this just widens the
    # patience window for STT to deliver.
    stt_commit_transcript_timeout: float = 5.0

    # G23 (2026-05-18): cross-turn transcript contamination filter. See
    # ``eidolon/livekit/plugins/stt/_transcript_gate.py`` for the full
    # rationale. When enabled, wraps the active STT plugin with a decorator
    # that drops INTERIM/FINAL events arriving within
    # ``stt_gate_suppress_window_ms`` of a prior FINAL — preventing the
    # framework's ``audio_recognition`` from concatenating next-turn
    # transcripts onto the just-finished turn (the bug that produced
    # phantom ``"嗯，听到了。"`` responses + wasted LLM calls in production).
    # Default OFF until real-world regression confirms suppressed_count
    # distribution is healthy.
    stt_transcript_gate_enabled: bool = False
    stt_gate_suppress_window_ms: int = 200

    # Room audio I/O sample rate (Hz). Unifies the entire audio output chain
    # — RoomIO, DuckingMixer, filler injection — at a single rate. Setting
    # this to match the TTS provider's native rate (e.g. 16000 for BailianTTS)
    # eliminates all resampling overhead in the framework. LiveKit's Opus
    # codec works at any rate; 16 kHz is adequate for Chinese speech quality
    # and reduces bandwidth ~33% vs the framework's default 24 kHz.
    audio_sample_rate: int = 16000


@dataclass
class LLMConfig:
    """LLM provider config (OpenAI-compatible endpoint).

    Env var names are ``OPENAI_LLM_*`` to keep the provider-prefix
    convention used by STT/TTS plugin configs. Today this is the only
    LLM provider supported; if more are added, introduce
    ``LLM_PROVIDER`` selector + new ``<provider>_LLM_*`` configs.
    """

    base_url: str = ""
    model: str = "gpt-4o-mini"
    api_key: str = ""
    # G4 (2026-05-16): expose tunables that ``lk_openai.LLM`` already
    # supports but we never plumbed through. ``None`` means "use the
    # plugin's NOT_GIVEN default", letting the framework decide.
    temperature: float | None = None
    timeout: float | None = None
    max_completion_tokens: int | None = None


@dataclass
class RemoteAgentRpcConfig:
    """Remote companion / agent via ``RemoteAgent`` gRPC (see ``eidolon/proto/.../remote_agent_rpc.proto``)."""

    # e.g. unix:///var/run/eidolon/remote_agent.sock or 127.0.0.1:50051
    target: str = ""
    locale: str = "zh"


@dataclass
class AgentConfig:
    """Top-level agent config, composed from per-concern sub-configs.

    The plugin configs (``bailian_stt``, ``sensetime_stt``, ``sensetime_tts``)
    auto-load from environment variables when the dataclass is instantiated
    (each plugin's config dataclass uses ``field(default_factory=...)``
    pointing at ``os.environ.get(...)``). Only the active provider's config
    is actually used at runtime — see ``stt_provider`` / ``tts_provider``.
    """

    core: CoreConfig = field(default_factory=CoreConfig)
    behavior: AgentBehaviorConfig = field(default_factory=AgentBehaviorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    remote_agent_rpc: RemoteAgentRpcConfig = field(default_factory=RemoteAgentRpcConfig)

    # Provider selection (env: STT_PROVIDER / TTS_PROVIDER / VAD_PROVIDER)
    stt_provider: str = "sensetime"   # "bailian" | "sensetime"
    tts_provider: str = "sensetime"   # "sensetime" | "bailian"
    vad_provider: str = "firered"     # "firered" | "silero" | "none"

    # Per-provider plugin configs (loaded from env via plugin's own factory).
    bailian_stt: BailianSTTConfig = field(default_factory=BailianSTTConfig)
    sensetime_stt: SenseTimeSTTConfig = field(default_factory=SenseTimeSTTConfig)
    bailian_tts: BailianTTSConfig = field(default_factory=BailianTTSConfig)
    sensetime_tts: SenseTimeTTSConfig = field(default_factory=SenseTimeTTSConfig)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build the config from environment variables (and the local .env).

        Steps:
        1. Load the file at ``EIDOLON_CHANNEL_LIVEKIT_ENV`` into ``os.environ``
           (raises ``ValueError`` if unset or not a file)
        2. Build core / behavior / llm with explicit env reads
        3. Plugin configs auto-load from ``os.environ`` via their ``default_factory``
        4. Run light validation; warn (not fail) on missing optional values
        """
        _bootstrap_dotenv()

        def get(key: str, fallback: str) -> str:
            return os.environ.get(key, fallback)

        try:
            port = int(get("AGENT_PORT", "8766"))
        except ValueError as e:
            raise ValueError(
                f"AGENT_PORT must be an integer, got {get('AGENT_PORT', '')!r}"
            ) from e

        cfg = cls(
            core=CoreConfig(
                livekit_url=get("LIVEKIT_URL", "ws://localhost:7880"),
                api_key=get("LIVEKIT_API_KEY", "devkey"),
                api_secret=get("LIVEKIT_API_SECRET", "devkey_secret"),
                host=get("AGENT_HOST", "0.0.0.0"),
                port=port,
            ),
            behavior=AgentBehaviorConfig(
                agent_mode=get("AGENT_MODE", "streaming").lower(),
                instructions=get("AGENT_INSTRUCTIONS", _DEFAULT_INSTRUCTIONS),
                welcome_message=get("AGENT_WELCOME_MESSAGE", _DEFAULT_WELCOME),
                false_interruption_timeout=(
                    None
                    if get("AGENT_FALSE_INTERRUPTION_TIMEOUT", "6.0").lower()
                    in ("none", "off", "disabled", "")
                    else float(get("AGENT_FALSE_INTERRUPTION_TIMEOUT", "6.0"))
                ),
                audio_sample_rate=int(get("AGENT_AUDIO_SAMPLE_RATE", "16000")),
                stt_commit_transcript_timeout=float(
                    get("AGENT_STT_COMMIT_TIMEOUT", "5.0")
                ),
                # G9 (2026-05-17): "" or "none"/"off"/"disabled" → None
                # (= disable AEC warmup, welcome immediately interruptible).
                aec_warmup_duration=(
                    None
                    if get("AGENT_AEC_WARMUP_DURATION", "1.0").lower()
                    in ("none", "off", "disabled", "")
                    else float(get("AGENT_AEC_WARMUP_DURATION", "1.0"))
                ),
                # G23 (2026-05-18): STT transcript gate. Default off.
                stt_transcript_gate_enabled=(
                    get("EIDOLON_STT_TRANSCRIPT_GATE_ENABLED", "false").lower()
                    in ("true", "1", "yes", "on")
                ),
                stt_gate_suppress_window_ms=int(
                    get("EIDOLON_STT_GATE_SUPPRESS_WINDOW_MS", "200")
                ),
            ),
            llm=LLMConfig(
                base_url=get("OPENAI_LLM_BASE_URL", ""),
                model=get("OPENAI_LLM_MODEL", "gpt-4o-mini"),
                api_key=get("OPENAI_LLM_API_KEY", ""),
                # G4 (2026-05-16): None when env unset/empty, letting the
                # OpenAI plugin's NOT_GIVEN defaults apply.
                temperature=_optional_float(get("OPENAI_LLM_TEMPERATURE", "")),
                timeout=_optional_float(get("OPENAI_LLM_TIMEOUT", "")),
                max_completion_tokens=_optional_int(
                    get("OPENAI_LLM_MAX_TOKENS", "")
                ),
            ),
            remote_agent_rpc=RemoteAgentRpcConfig(
                target=get("REMOTE_AGENT_RPC_TARGET", "").strip(),
                locale=(get("REMOTE_AGENT_RPC_LOCALE", "zh").strip() or "zh"),
            ),
            stt_provider=get("STT_PROVIDER", "sensetime").lower(),
            tts_provider=get("TTS_PROVIDER", "sensetime").lower(),
            vad_provider=get("VAD_PROVIDER", "firered").lower(),
            # bailian_stt / sensetime_stt / sensetime_tts: default_factory
            # triggers their own env-driven init when this AgentConfig is
            # instantiated.
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Light runtime validation; logs warnings but doesn't raise on
        missing optional values.

        Hard errors (raise ``ValueError``):
          - Unknown provider names (would crash at server.py build time anyway)
          - Invalid agent_mode

        Soft warnings:
          - Default LiveKit dev keys in non-localhost deployments
          - Missing API keys for the active provider
        """
        import logging
        log = logging.getLogger("agent.config")

        # Hard validation: provider names must be known
        if self.stt_provider not in ("bailian", "sensetime"):
            raise ValueError(
                f"Unknown stt_provider: {self.stt_provider!r} "
                f"(supported: 'bailian', 'sensetime')"
            )
        if self.tts_provider not in ("sensetime", "bailian"):
            raise ValueError(
                f"Unknown tts_provider: {self.tts_provider!r} "
                f"(supported: 'sensetime', 'bailian')"
            )
        if self.vad_provider not in ("firered", "firered_pvad", "silero", "none", "disabled"):
            raise ValueError(
                f"Unknown vad_provider: {self.vad_provider!r} "
                f"(supported: 'firered', 'silero', 'none')"
            )
        if self.behavior.agent_mode not in ("streaming", "batch"):
            raise ValueError(
                f"Unknown agent_mode: {self.behavior.agent_mode!r} "
                f"(supported: 'streaming', 'batch')"
            )

        # Soft warnings: missing API keys for the active provider
        if self.stt_provider == "bailian" and not self.bailian_stt.api_key:
            log.warning(
                "[AgentConfig] STT provider is 'bailian' but DASHSCOPE_API_KEY "
                "is empty; STT will fail at first utterance"
            )
        if self.stt_provider == "sensetime" and not self.sensetime_stt.api_key:
            log.warning(
                "[AgentConfig] STT provider is 'sensetime' but "
                "SENSETIME_STT_API_KEY / SENSEAUDIO_API_KEY is empty; STT "
                "will fail at first utterance"
            )
        if self.tts_provider == "sensetime" and not self.sensetime_tts.api_key:
            log.warning(
                "[AgentConfig] TTS provider is 'sensetime' but "
                "SENSETIME_TTS_API_KEY / SENSEAUDIO_API_KEY is empty; TTS "
                "will fail at warmup"
            )
        if self.tts_provider == "bailian" and not self.bailian_tts.api_key:
            log.warning(
                "[AgentConfig] TTS provider is 'bailian' but "
                "BAILIAN_TTS_API_KEY / DASHSCOPE_API_KEY is empty; TTS "
                "will fail at warmup"
            )
        if not self.remote_agent_rpc.target:
            if not self.llm.api_key and not self.llm.base_url:
                log.warning(
                    "[AgentConfig] Both OPENAI_LLM_API_KEY and OPENAI_LLM_BASE_URL "
                    "are empty; LLM will likely fail at first turn"
                )

        # Network / port validation
        if not (1 <= self.core.port <= 65535):
            raise ValueError(
                f"AGENT_PORT must be in [1, 65535], got {self.core.port}"
            )


def load_agent_config() -> AgentConfig:
    """Load the agent configuration from .env and environment variables."""
    return AgentConfig.from_env()
