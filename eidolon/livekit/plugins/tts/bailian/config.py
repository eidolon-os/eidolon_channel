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
    # F5 (2026-05-16): how many conns to warm SYNCHRONOUSLY at startup before
    # accepting the first turn. After warmup completes, the pool's background
    # refill brings it up to ``pool_size``. Lets cold-start finish faster
    # (e.g. 3 × 1.3s ≈ 3.5s instead of 4 × 1.3s ≈ 5.1s) while still keeping
    # the steady-state pool target high. Set equal to ``pool_size`` to disable.
    pool_size_bootstrap: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_TTS_POOL_SIZE_BOOTSTRAP", "3")
        )
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
    # F2 (2026-05-16) → G13 (2026-05-17): acquire-time staleness eviction
    # window. Default lowered from 25.0 to 15.0 after observing dashscope
    # going "silently dead" by age ~20s (round-2 incident). G10 WS heartbeat
    # is the primary defense now; this is belt + suspenders.
    pool_max_idle_sec: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_POOL_MAX_IDLE_SEC", "15.0")
        )
    )
    # G10 (2026-05-17): WS-level heartbeat interval. aiohttp sends a PING
    # every N seconds; PONG must arrive within N/2 seconds or ws.closed=True.
    # 15s/7.5s strikes balance — most utterances complete before first PING,
    # but dashscope server-side death (~20s observed) is detected within
    # 22.5s. Set to 0 / negative to disable (not recommended in production).
    ws_heartbeat_sec: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_WS_HEARTBEAT_SEC", "15.0")
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
    # G11 (2026-05-17): max wall-clock between consecutive LLM tokens. The
    # framework's input_ch may not yield StopAsyncIteration cleanly when
    # LLM stream ends (observed in round-2 incident); without this timeout
    # _input_loop hangs forever. 10s is generous — typical inter-token gap
    # is < 200ms even on the slowest LLMs.
    inter_token_timeout: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_TTS_INTER_TOKEN_TIMEOUT", "10.0")
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
    # G5 (2026-05-16): optional first-sentence aggressive flush. Helps when
    # the LLM body bursts in within ~200ms — the default soft_min=12 misses
    # early-punct flush opportunities and TTS first-byte ends up tied to
    # LLM-end. Empty / 0 → disabled (use ``aggregator_soft_min_chars``).
    # Default tuned for first-audio latency (2026-05-30): flush the FIRST TTS
    # segment as soon as a short prefix + any punctuation is available, instead
    # of waiting for the full soft_min (12). Real-room A/B: commit->first_audio
    # 820->759 p50 / 960->804 p95 (into the top-tier target band), 5/5 pass.
    # Trade-off: the first spoken segment is short; raise these for smoother
    # first-segment prosody at the cost of a later first byte.
    aggregator_first_sentence_soft_min_chars: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_SOFT_MIN_CHARS", "4")
        )
    )
    aggregator_first_sentence_flush_any_punct: bool = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_TTS_AGGREGATOR_FIRST_SENTENCE_FLUSH_ANY_PUNCT", "true"
        ).lower()
        == "true"
    )
    log_audio_diag: bool = field(
        default_factory=lambda: os.environ.get("BAILIAN_TTS_LOG_AUDIO_DIAG", "true").lower()
        == "true"
    )
