from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class MusicTraceContext:
    turn_id: str
    call_id: str
    request_id: str
    playback_ref: str


_current: ContextVar[MusicTraceContext | None] = ContextVar(
    "music_trace_context", default=None
)


def current_music_trace() -> MusicTraceContext | None:
    return _current.get()


def set_music_trace(context: MusicTraceContext):
    return _current.set(context)


def reset_music_trace(token) -> None:
    _current.reset(token)


def redact_music_url(value: str) -> str:
    parts = urlsplit(value)
    query = [(key, "<redacted>") for key, _ in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
