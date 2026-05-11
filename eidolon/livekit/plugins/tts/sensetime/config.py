"""Configuration for SenseTime SenseAudio TTS."""
import os
from dataclasses import dataclass, field


@dataclass
class SenseTimeTTSConfig:
    """Configuration for SenseTime SenseAudio TTS.

    Attributes:
        api_url: WebSocket endpoint URL for the SenseAudio TTS API.
            Default: "wss://api.senseaudio.cn/ws/v1/t2a_v2"
        api_key: API key for authentication. Can also be set via
            SENSEAUDIO_API_KEY environment variable.
        model: TTS model name. Default: "senseaudio-tts-1.5-260319"
        voice: Voice ID. Default: "female_0033_a"
        sample_rate: Audio sample rate in Hz. Default: 32000
        bitrate: Audio bitrate. Default: 128000
        channel: Number of audio channels. Default: 1 (mono)
        audio_format: Audio format. Default: "pcm"
        speed: Speech speed (0.5-2.0). Default: 1.0
        vol: Volume (0-2.0). Default: 1.0
        pitch: Pitch adjustment (-500 to 500). Default: 0
        pool_size: WebSocket connection pool size. Default: 4 (Round 8 R8.12.d).
            Originally R8.7 set this to 2 based on a spike that measured
            single-cancel-then-recover. Production logs (2026-05-07)
            showed that during high-interaction sessions (frequent
            interrupts, sub-2s turn intervals), refill latency (~2s
            per conn — connect + task_start ack) couldn't keep up: every
            new acquire fell to slow path, defeating the pool's purpose.
            Bumping to 4 gives 3 spare conns for refill to catch up
            during a burst. Configurable via env for further tuning.
        log_audio_diag: Whether to log per-segment audio diagnostics. Default: True
    """

    api_url: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_TTS_API_URL", "wss://api.senseaudio.cn/ws/v1/t2a_v2")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_TTS_API_KEY", "") or os.environ.get("SENSEAUDIO_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_TTS_MODEL", "senseaudio-tts-1.5-260319")
    )
    voice: str = field(
        default_factory=lambda: os.environ.get("SENSETIME_TTS_VOICE", "female_0033_a")
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("SENSETIME_TTS_SAMPLE_RATE", "32000"))
    )
    bitrate: int = field(default=128000)
    channel: int = field(default=1)
    audio_format: str = field(default="pcm")
    speed: float = field(
        default_factory=lambda: float(os.environ.get("SENSETIME_TTS_SPEED", "1.0"))
    )
    vol: float = field(
        default_factory=lambda: float(os.environ.get("SENSETIME_TTS_VOL", "1.0"))
    )
    pitch: int = field(
        default_factory=lambda: int(os.environ.get("SENSETIME_TTS_PITCH", "0"))
    )
    pool_size: int = field(
        default_factory=lambda: int(
            os.environ.get("SENSETIME_TTS_POOL_SIZE", "4")
        )
    )
    log_audio_diag: bool = field(default=True)

    # ── Round 8 R8.2: SentenceAggregator thresholds ─────────────────
    # Buffer LLM tokens into sentence-sized batches before feeding to
    # SenseAudio TTS (which treats each task_continue as an independent
    # synthesis batch with ~600 ms gap). See _aggregator.py.
    aggregator_soft_min_chars: int = field(default=12)
    aggregator_hard_max_chars: int = field(default=60)
    aggregator_idle_ms: int = field(default=300)

    # ── Round 8 R8.6: timeouts now configurable ─────────────────────
    # Maximum time to wait for the first LLM token before giving up on
    # this synthesize stream (audio output stays silent). Hardcoded to
    # 15 s historically; configurable here so deployments with slow
    # LLM upstreams can raise it without forking.
    first_token_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("SENSETIME_TTS_FIRST_TOKEN_TIMEOUT", "15.0")
        )
    )
    # After input_done, how long to wait with zero audio frames before
    # treating the server as silent and force-exiting (fault detection,
    # never fires on the happy path).
    no_first_audio_timeout: float = field(default=5.0)
