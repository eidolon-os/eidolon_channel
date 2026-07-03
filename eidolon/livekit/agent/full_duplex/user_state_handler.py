"""Full-duplex user-state entry workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .user_state_event import FullDuplexUserStateEvent

logger = logging.getLogger("agent")


class FullDuplexUserStateHandler:
    """Route LiveKit user_state changes into full-duplex session effects."""

    def __init__(
        self,
        *,
        publish_companion_ui_state: Callable[[str, str], None],
        signal_stt_user_away: Callable[[], None],
        signal_stt_user_present: Callable[[], None],
        handle_speaking_started: Callable[[], None],
        handle_speaking_stopped: Callable[[], None],
    ) -> None:
        self._publish_companion_ui_state = publish_companion_ui_state
        self._signal_stt_user_away = signal_stt_user_away
        self._signal_stt_user_present = signal_stt_user_present
        self._handle_speaking_started = handle_speaking_started
        self._handle_speaking_stopped = handle_speaking_stopped

    def handle(self, event: Any) -> FullDuplexUserStateEvent:
        state_event = FullDuplexUserStateEvent.from_event(event)
        old = state_event.old_state
        new = state_event.new_state
        logger.info("[StreamingPipeline] user_state: %s -> %s", old, new)

        self._publish_companion_ui(old=old, new=new)
        self._sync_stt_presence(old=old, new=new)
        if state_event.started_speaking:
            self._handle_speaking_started()
        elif state_event.stopped_speaking:
            self._handle_speaking_stopped()
        return state_event

    def _publish_companion_ui(self, *, old: str, new: str) -> None:
        if new == "speaking":
            self._publish_companion_ui_state("listening", "user_state:speaking")
        elif old == "speaking" and new == "listening":
            self._publish_companion_ui_state("listening", "user_state:listening")
        elif new == "away":
            self._publish_companion_ui_state("idle", "user_state:away")

    def _sync_stt_presence(self, *, old: str, new: str) -> None:
        if new == "away":
            self._signal_stt_user_away()
        elif old == "away" and new in ("listening", "speaking"):
            self._signal_stt_user_present()
