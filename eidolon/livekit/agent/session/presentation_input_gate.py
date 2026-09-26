"""Prevent a separate speaker's output from becoming the input device's turn."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PresentationInputGate:
    """Half-duplex fence owned by one AgentSession, independent of UI/VAD state.

    AgentSession's speaking boundary follows its audio sink through playout,
    rather than TTS synthesis completion. Keep the existing ESP playback tail
    (1.2 seconds) to cover transport/device buffers and acoustic decay. The
    input device has no echo reference for the other room's speaker.
    """

    def __init__(self, session: Any, *, tail_seconds: float = 1.2) -> None:
        self._session = session
        self._tail_seconds = tail_seconds
        self._release: asyncio.TimerHandle | None = None
        self._closed = False
        self._blocked = False
        self._restore_enabled = False
        session.on("agent_state_changed", self._on_state)
        session.on("close", self.close)

    def _on_state(self, event: Any) -> None:
        if self._closed:
            return
        if event.new_state == "speaking":
            if self._release is not None:
                self._release.cancel()
                self._release = None
            if not self._blocked:
                self._restore_enabled = self._session.input.audio_enabled
                self._blocked = True
                self._session.input.set_audio_enabled(False)
                logger.info("[PresentationInputGate] input blocked during remote playback")
        elif self._blocked and self._release is None:
            self._release = asyncio.get_running_loop().call_later(
                self._tail_seconds, self._release_input,
            )

    def _release_input(self) -> None:
        self._release = None
        if self._closed:
            return
        self._blocked = False
        if self._restore_enabled:
            self._session.input.set_audio_enabled(True)
        logger.info("[PresentationInputGate] remote playback tail drained; input restored=%s",
                    self._restore_enabled)

    def close(self, *_: Any) -> None:
        self._closed = True
        if self._release is not None:
            self._release.cancel()
            self._release = None
        self._session.off("agent_state_changed", self._on_state)
        self._session.off("close", self.close)
        # Never reopen capture while the owning session is shutting down.
