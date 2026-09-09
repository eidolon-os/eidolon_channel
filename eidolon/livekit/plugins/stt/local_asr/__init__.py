"""Recognition served by the Host itself.

The provider name is the capability name — `local_asr` — because Ops, the
component contract, the service manifest and this plugin all have to agree
about the same thing, and one shared string cannot disagree with itself.
"""

from __future__ import annotations

from .config import LocalAsrSTTConfig
from .endpoint import (
    LocalAsrEndpointError,
    resolve_port,
    resolve_ready_url,
    resolve_stream_url,
)
from .speech_stream import LocalAsrSpeechStream
from .stt import PROVIDER, LocalAsrSTT

__all__ = [
    "PROVIDER",
    "LocalAsrEndpointError",
    "LocalAsrSTT",
    "LocalAsrSTTConfig",
    "LocalAsrSpeechStream",
    "resolve_port",
    "resolve_ready_url",
    "resolve_stream_url",
]
