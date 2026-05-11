"""Bailian FunASR STT plugin for LiveKit Agents.

Exports
-------
BailianFunASRSTT
    Main STT class — implements the LiveKit STT plugin interface.
BailianFunASRSpeechStream
    Streaming transcription implementation.
BailianConnectionManager
    Low-level WebSocket connection manager.
models
    Typed dataclasses for FunASR server messages.
BailianSTTConfig
    Typed configuration dataclass for plugin initialization.
register_bailian_stt
    Registry function for third-party extensions.
create_bailian_stt
    Factory for creating a configured BailianFunASRSTT instance.
"""

from __future__ import annotations

__all__ = [
    # Core classes
    "BailianFunASRSTT",
    "BailianFunASRSpeechStream",
    "BailianConnectionManager",
    # Config
    "BailianSTTConfig",
    # Models
    "FunASREventType",
    "FunASRResultGenerated",
    "FunASRSentence",
    "FunASRTaskStarted",
    "FunASRTaskFailed",
    "FunASRTaskFinished",
    "FunASRWord",
    "FunASRHeartbeat",
    "parse_funasr_message",
    # Exceptions
    "BailianConnectionError",
    "BailianTaskFailedError",
    # Registry
    "register_bailian_stt",
    "get_bailian_stt",
    "create_bailian_stt",
]


def __getattr__(name: str):
    # Core classes — these import stt.py which imports other modules,
    # so we defer them until actually needed to avoid circular imports.
    if name == "BailianFunASRSTT":
        from .stt import BailianFunASRSTT

        return BailianFunASRSTT

    if name == "BailianFunASRSpeechStream":
        from .speech_stream import BailianFunASRSpeechStream

        return BailianFunASRSpeechStream

    if name == "BailianConnectionManager":
        from .connection_manager import BailianConnectionManager

        return BailianConnectionManager

    # Exceptions
    if name == "BailianConnectionError":
        from .connection_manager import BailianConnectionError

        return BailianConnectionError

    if name == "BailianTaskFailedError":
        from .connection_manager import BailianTaskFailedError

        return BailianTaskFailedError

    # Config
    if name == "BailianSTTConfig":
        from .config import BailianSTTConfig

        return BailianSTTConfig

    # Models — delegate to the models submodule
    if name in (
        "FunASREventType",
        "FunASRResultGenerated",
        "FunASRSentence",
        "FunASRTaskStarted",
        "FunASRTaskFailed",
        "FunASRTaskFinished",
        "FunASRWord",
        "FunASRHeartbeat",
        "parse_funasr_message",
    ):
        from . import models

        return getattr(models, name)

    # Registry
    if name in ("register_bailian_stt", "get_bailian_stt", "create_bailian_stt"):
        from . import registry

        return getattr(registry, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
