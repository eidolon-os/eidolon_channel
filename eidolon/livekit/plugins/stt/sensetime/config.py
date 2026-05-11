"""Configuration for SenseTime SenseAudio STT."""
import os
from dataclasses import dataclass, field


@dataclass
class SenseTimeSTTConfig:
    """Configuration for SenseTime SenseAudio STT.

    Attributes:
        api_url: WebSocket endpoint URL for the SenseAudio STT API.
            Default: "wss://api.senseaudio.cn/ws/v1/audio/transcriptions"
        api_key: API key for authentication. Can also be set via
            SENSETIME_STT_API_KEY or SENSEAUDIO_API_KEY environment variable.
        model: STT model name. Default: "" (use server default)
        sample_rate: Audio sample rate in Hz. Default: 16000
        language: Language hint. Default: "zh"
        log_audio_diag: Whether to log per-segment audio diagnostics. Default: True
    """

    api_url: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_STT_API_URL", "wss://api.senseaudio.cn/ws/v1/audio/transcriptions")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_STT_API_KEY", "") or os.environ.get("SENSEAUDIO_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_STT_MODEL", "")
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("SENSETIME_STT_SAMPLE_RATE", "16000"))
    )
    language: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_STT_LANGUAGE", "zh")
    )
    log_audio_diag: bool = field(default=True)

    # ── Round 8 R8.3: stream-level reliability / self-healing ──────
    # When the WebSocket dies mid-session (network blip, server restart),
    # the stream attempts to reconnect inline rather than returning to
    # the framework (which never re-creates _STTPipeline). On exhaustion
    # of this budget, the stream emits an STT error and exits — the
    # framework can then close the AgentSession and the client reconnect.
    max_stream_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("SENSETIME_STT_MAX_STREAM_RETRIES", "3")
        )
    )
    # Backoffs in seconds, applied per-attempt (sequence; capped at last
    # value). 0.2 / 0.5 / 1.0 — fast first attempt, then ramp.
    stream_retry_backoffs: tuple[float, ...] = field(default=(0.2, 0.5, 1.0))

    # ── Round 8 R8.12.a: server-side VAD segmentation tuning ──────
    # SenseAudio's task_start protocol accepts a ``vad_setting`` block.
    # Without it, server uses internal defaults (observed ~500 ms silence
    # threshold), which split natural Chinese pauses (verb-particle
    # phrasing often has 400-700 ms gaps) and produce fragmented
    # transcripts that the EOT scoring layer can't repair.
    #
    # Bumping ``silence_duration_ms`` to 800 gives natural pauses room
    # while still detecting genuine end-of-utterance within ~1 s of
    # actual silence. Operators can override via env if their workload
    # needs faster turn detection (e.g. command-only) or longer (e.g.
    # dictation with thoughtful pauses).
    silence_duration_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("SENSETIME_STT_SILENCE_DURATION_MS", "800")
        )
    )
    # Minimum span (ms) for the server to register something as a speech
    # segment. Below this, the audio is treated as noise and discarded.
    # 300 ms matches SenseAudio's documented default; raised slightly
    # would filter more cough/click but risks dropping short
    # acknowledgements like "嗯" / "好".
    min_speech_duration_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("SENSETIME_STT_MIN_SPEECH_DURATION_MS", "300")
        )
    )
