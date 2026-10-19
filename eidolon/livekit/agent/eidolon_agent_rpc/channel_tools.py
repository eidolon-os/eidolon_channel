from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChannelToolCall:
    turn_id: str
    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChannelToolResult:
    turn_id: str
    call_id: str
    name: str
    ok: bool
    content: dict[str, object]
    error_code: str = ""
    error_message: str = ""
