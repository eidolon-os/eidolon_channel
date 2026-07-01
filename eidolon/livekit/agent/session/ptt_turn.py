"""Finalize half-duplex push-to-talk turns."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.session.ptt_turn")


class PttTurnFinalizer:
    """Own the PTT release -> manual ``commit_user_turn`` path."""

    def commit_release(
        self,
        *,
        session: Any,
        had_speech: bool,
        reset_had_speech: Callable[[], None],
        reset_eot: Callable[[], None],
        transcript_timeout: float,
        timeline: TurnTimeline | None,
    ) -> bool:
        """Commit exactly once on release when the current hold contained speech."""

        if session is None:
            return False
        if not had_speech:
            logger.info("[ptt-manual] release with no speech this hold; skipping commit")
            return False

        reset_had_speech()
        try:
            reset_eot()
        except Exception:
            logger.debug("[ptt-manual] eot reset failed (non-fatal)", exc_info=True)

        if timeline is not None:
            timeline.mark("turn_committed_at")
            timeline.set_attr(
                "ptt_release_commit",
                {
                    "transcript_timeout_ms": int(round(transcript_timeout * 1000)),
                    "text_source_policy": "final_or_interim_on_timeout",
                },
            )

        try:
            session.commit_user_turn(transcript_timeout=transcript_timeout)
            logger.info(
                "[ptt-manual] PTT release -> commit_user_turn(transcript_timeout=%.1fs)",
                transcript_timeout,
            )
            return True
        except Exception:
            logger.exception("[ptt-manual] commit_user_turn on release failed")
            return False
