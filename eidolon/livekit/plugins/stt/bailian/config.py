"""Configuration dataclass for BailianFunASRSTT."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    # conn_options is intentionally omitted here — it is a LiveKit runtime
    # object that does not belong in a plain dataclass; the STT class
    # accepts it as a separate __init__ keyword argument.
