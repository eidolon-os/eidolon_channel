"""This Host's own speech synthesis, as a LiveKit TTS provider.

The provider name is the Host capability name, so "the config asks for local
speech" and "this Host can do local speech" compare as two identical strings
rather than through a table that could disagree with either side.
"""

from .config import LocalTtsConfig
from .endpoint import (
    LocalTtsEndpointError,
    resolve_port,
    resolve_ready_url,
    resolve_stream_url,
)
from .tts import PROVIDER, LocalTTS, LocalTtsError

__all__ = [
    "PROVIDER",
    "LocalTTS",
    "LocalTtsConfig",
    "LocalTtsEndpointError",
    "LocalTtsError",
    "resolve_port",
    "resolve_ready_url",
    "resolve_stream_url",
]
