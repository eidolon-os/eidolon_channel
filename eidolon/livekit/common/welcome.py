"""Welcome configuration and small, deployment-local PCM assets.

Strings retain the legacy TTS behavior; ``{audio: builtin:soft-ready}`` selects
a packaged sound. Custom paths are relative to the main settings YAML directory.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WelcomeAudio:
    audio: str = "builtin:soft-ready"


WelcomeMessage = str | WelcomeAudio
_BUILTIN_AUDIO = Path(__file__).resolve().parent / "assets" / "soft-ready.wav"


def parse_welcome_message(raw: Any, *, base_dir: Path) -> WelcomeMessage:
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict) or set(raw) != {"audio"}:
        raise ValueError("behavior.welcome_message must be text, empty text, or {audio: path}")
    source = raw["audio"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("behavior.welcome_message.audio must be a non-empty string")
    source = source.strip()
    if not source.startswith("builtin:"):
        if "://" in source:
            raise ValueError("welcome audio must be a local WAV file, not a URL")
        path = Path(source).expanduser()
        source = str((base_dir / path).resolve())
    welcome = WelcomeAudio(source)
    validate_welcome_message(welcome)
    return welcome


def welcome_audio_path(welcome: WelcomeAudio) -> Path:
    if welcome.audio == "builtin:soft-ready":
        return _BUILTIN_AUDIO
    if welcome.audio.startswith("builtin:"):
        raise ValueError(f"unknown welcome audio: {welcome.audio}")
    path = Path(welcome.audio)
    if not path.is_absolute():
        raise ValueError("welcome audio path must be resolved against the settings directory")
    return path


def validate_welcome_message(welcome: WelcomeMessage) -> None:
    if isinstance(welcome, str):
        return
    if not isinstance(welcome, WelcomeAudio):
        raise ValueError("behavior.welcome_message must be text or WelcomeAudio")
    path = welcome_audio_path(welcome)
    stat = path.stat()
    _read_pcm(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=16)
def _read_pcm(path: str, mtime_ns: int, size: int) -> tuple[bytes, int]:
    # Versioned cache keys allow an explicitly reloaded config to replace a file.
    with wave.open(path, "rb") as audio:
        rate = audio.getframerate()
        count = audio.getnframes()
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("welcome audio must be a mono PCM16 WAV")
        if not 8000 <= rate <= 48000 or not 0 < count <= rate * 10:
            raise ValueError("welcome WAV must be 8–48 kHz and at most 10 seconds")
        pcm = audio.readframes(count)
        if len(pcm) != count * 2:
            raise ValueError("welcome WAV is truncated")
        return pcm, rate


@lru_cache(maxsize=16)
def _resampled_pcm(path: str, mtime_ns: int, size: int, sample_rate: int) -> bytes:
    from livekit import rtc

    pcm, rate = _read_pcm(path, mtime_ns, size)
    if rate == sample_rate:
        return pcm
    resampler = rtc.AudioResampler(rate, sample_rate, num_channels=1)
    frames = resampler.push(bytearray(pcm)) + resampler.flush()
    return b"".join(bytes(frame.data) for frame in frames)


def prepare_welcome_audio(welcome: WelcomeMessage | None, *, sample_rate: int) -> bytes | None:
    """Load/resample once per worker and asset version, before entering a session."""
    if not isinstance(welcome, WelcomeAudio):
        return None
    path = welcome_audio_path(welcome)
    stat = path.stat()
    return _resampled_pcm(str(path), stat.st_mtime_ns, stat.st_size, sample_rate)
