"""Configuration dataclass for BailianFunASRSTT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_api_key() -> str:
    """Resolve API key from env vars, preferring the agent's canonical name.

    Order of precedence:
      1. ``BAILIAN_STT_API_KEY``  — agent's canonical naming (per .env convention)
      2. ``DASHSCOPE_API_KEY``    — Aliyun's industry-standard env var (kept for
         users who follow Aliyun docs and don't want to rename)
    """
    return (
        os.environ.get("BAILIAN_STT_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
    )


@dataclass
class BailianSTTConfig:
    """Typed configuration options for the Bailian FunASR STT plugin.

    Env vars (in order of precedence):
      - ``BAILIAN_STT_API_URL``   → ``api_url``
      - ``BAILIAN_STT_API_KEY``   → ``api_key`` (or ``DASHSCOPE_API_KEY``)
      - ``BAILIAN_STT_MODEL``     → ``model``
      - ``BAILIAN_STT_SAMPLE_RATE`` → ``sample_rate``
      - ``BAILIAN_STT_LANGUAGE``  → ``language``
      - ``BAILIAN_STT_ITN``       → ``itn`` (``"true"``/``"false"``)
    """

    api_url: str = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_STT_API_URL",
            "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
        )
    )
    api_key: str = field(default_factory=_resolve_api_key)
    model: str = field(
        default_factory=lambda: os.environ.get(
            "BAILIAN_STT_MODEL", "fun-asr-realtime-2026-02-28"
        )
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_STT_SAMPLE_RATE", "16000"))
    )
    language: str = field(
        default_factory=lambda: os.environ.get("BAILIAN_STT_LANGUAGE", "zh")
    )
    itn: bool = field(
        default_factory=lambda: os.environ.get("BAILIAN_STT_ITN", "true").lower() == "true"
    )
    language_hints: str | None = None

    # Silence (ms) FunASR waits before declaring sentence end (the FINAL). The
    # DashScope default is 800ms, which dominates the post-speech-stop FINAL
    # latency and gates playback scheduling. Lowering it makes the FINAL arrive
    # sooner (and is bounded by the provider's 200-6000ms range). 0/None leaves
    # it unset (provider default).
    max_sentence_silence_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_STT_MAX_SENTENCE_SILENCE_MS", "400")
        )
    )

    # Connection keepalive (independent of the VAD billing gate below). DashScope
    # FunASR kills an idle recognition task after ~23s of no audio ("request
    # timeout after 23 seconds"). Between turns — especially during a long agent
    # reply when a half-duplex device closes its mic — no user audio flows, so the
    # freshly opened task would hit that timeout and churn a reconnect. When the
    # send loop sees no real audio for this interval it pushes a short silence
    # frame to keep the task alive (run-task sets heartbeat=True, so silent audio
    # is a valid keepalive). 0 / negative disables.
    keepalive_interval_sec: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_STT_KEEPALIVE_INTERVAL_SEC", "10.0")
        )
    )

    # G16 (2026-05-17): VAD-gated audio forwarding for cost reduction.
    # Defaults OFF (gate_enabled=False) — current operators see no behavior
    # change until they explicitly opt in via env. See ../_gate.py for the
    # full design. Quick summary:
    #   - GATED: silence period, only sends 1Hz silence keepalive
    #   - FORWARDING: speech period, normal audio passthrough
    #   - Preroll buffer captures pre-VAD-trigger audio (no first-word loss)
    #   - Energy fallback covers VAD false negatives
    gate_enabled: bool = field(
        default_factory=lambda: os.environ.get("BAILIAN_STT_GATE_ENABLED", "false").lower()
        == "true"
    )
    gate_preroll_ms: int = field(
        default_factory=lambda: int(os.environ.get("BAILIAN_STT_GATE_PREROLL_MS", "500"))
    )
    gate_tail_window_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_STT_GATE_TAIL_MS", "1500")
        )
    )
    gate_keepalive_interval_sec: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_STT_GATE_KEEPALIVE_INTERVAL_SEC", "1.0")
        )
    )
    gate_keepalive_frame_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("BAILIAN_STT_GATE_KEEPALIVE_FRAME_MS", "100")
        )
    )
    gate_vad_high_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_STT_GATE_VAD_HIGH", "0.6")
        )
    )
    gate_vad_low_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_STT_GATE_VAD_LOW", "0.3")
        )
    )
    gate_rms_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("BAILIAN_STT_GATE_RMS", "500")
        )
    )

    # Diagnostic audio dump (default OFF). When enabled, write the exact PCM the
    # recognizer receives — post-network, post-resample; 16 kHz mono — to a WAV
    # under ``dump_dir``, one file per stream run. Lets us listen to a device's
    # uplink audio to tell clean speech from echo/noise. Configure via
    # settings.yaml (``bailian_stt.dump_wav`` / ``dump_dir``); the env vars below
    # remain as a fallback default so it can also be flipped without editing yaml.
    dump_wav: bool = field(
        default_factory=lambda: os.environ.get("EIDOLON_STT_DUMP_WAV", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    dump_dir: str = field(
        default_factory=lambda: os.environ.get(
            "EIDOLON_STT_DUMP_DIR",
            str(
                Path(os.environ.get("EIDOLON_CACHE_ROOT", "~/eidolon/cache")).expanduser()
                / "debug/channel"
            ),
        )
    )
    # conn_options is intentionally omitted here — it is a LiveKit runtime
    # object that does not belong in a plain dataclass; the STT class
    # accepts it as a separate __init__ keyword argument.
