"""Commit a completed user speech turn into LiveKit AgentSession."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import FillerManager

logger = logging.getLogger("agent.session.turn_commit")


class UserTurnCommitter:
    """Own the guarded ``commit_user_turn`` path after VAD end."""

    def commit_or_skip(
        self,
        *,
        session: Any,
        eot_model: Any,
        transcript: str,
        transcript_timeout: float,
        timeline: TurnTimeline | None,
        inject_interrupted_context: Callable[[], None],
        filler: FillerManager | None,
    ) -> bool:
        """Commit only when STT produced text; always reset EOT turn state.

        Returns ``True`` when ``session.commit_user_turn`` was called.
        """

        if transcript:
            eot_model.record_turn(
                transcript,
                is_complete=True,
                eot_score=eot_model._current_eot_score,
            )
            eot_model.reset()
            inject_interrupted_context()
            if timeline is not None:
                timeline.mark("turn_committed_at")
            session.commit_user_turn(transcript_timeout=transcript_timeout)
            if filler is not None and session.output.audio is not None:
                filler.inject(session.output.audio)
            return True

        eot_model.reset()
        logger.info(
            "[UserTurnCommitter] VAD-end with empty ASR; "
            "skipping commit_user_turn (AEC window / noise / STT hiccup)"
        )
        return False
