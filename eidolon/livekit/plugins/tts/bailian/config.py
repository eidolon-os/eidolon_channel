"""Configuration for Bailian CosyVoice TTS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _resolve_api_key() -> str:
    return (
        os.environ.get("BAILIAN_TTS_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
    )


@dataclass
class BailianTTSConfig:
    """Typed config options for Bailian CosyVoice TTS plugin."""

    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_TTS_API_URL",
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        )
    )
    api_key: str = field(default_factory=_resolve_api_key)
    model: str = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_TTS_MODEL",
            "cosyvoice-v3-flash",
        )
    )
    voice: str = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_TTS_VOICE",
            "longanyang",
        )
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_TTS_SAMPLE_RATE", "16000"))
    )
    audio_format: str = field(
        default_factory=lambda: os.environ.get("BAILIAN_TTS_AUDIO_FORMAT", "pcm")
    )
    speech_rate: float = field(
        default_factory=lambda: float(os.environ.get("BAILIAN_TTS_SPEECH_RATE", "1.0"))
    )
    volume: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_TTS_VOLUME", "50"))
    )
    pitch: float = field(
        default_factory=lambda: float(os.environ.get("BAILIAN_TTS_PITCH", "1.0"))
    )
    pool_size: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_TTS_POOL_SIZE", "8"))
    )
    pool_refill_backoff: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_POOL_REFILL_BACKOFF", "1.5")
        )
    )
    pool_acquire_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_POOL_ACQUIRE_TIMEOUT", "12.0")
        )
    )
    task_started_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_TASK_STARTED_TIMEOUT", "15.0")
        )
    )
    task_finished_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_TASK_FINISHED_TIMEOUT", "20.0")
        )
    )
    first_token_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_FIRST_TOKEN_TIMEOUT", "15.0")
        )
    )
    no_first_audio_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_NO_FIRST_AUDIO_TIMEOUT", "5.0")
        )
    )
    aggregator_soft_min_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_TTS_AGGREGATOR_SOFT_MIN_CHARS", "12")
        )
    )
    aggregator_hard_max_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_TTS_AGGREGATOR_HARD_MAX_CHARS", "80")
        )
    )
    aggregator_idle_ms: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_TTS_AGGREGATOR_IDLE_MS", "300"))
    )
    log_audio_diag: bool = field(
        default_factory=lambda: os.environ.get("BAILIAN_TTS_LOG_AUDIO_DIAG", "true").lower()
        == "true"
    )
