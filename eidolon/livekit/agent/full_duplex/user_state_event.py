"""Normalized user-state event shape for full-duplex sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FullDuplexUserStateEvent:
    old_state: str = ""
    new_state: str = ""

    @classmethod
    def from_event(cls, event: Any) -> FullDuplexUserStateEvent:
        if isinstance(event, cls):
            return event
        return cls(
            old_state=getattr(event, "old_state", "") or "",
            new_state=getattr(event, "new_state", "") or "",
        )

    @property
    def started_speaking(self) -> bool:
        return self.new_state == "speaking"

    @property
    def stopped_speaking(self) -> bool:
        return self.old_state == "speaking" and self.new_state == "listening"
