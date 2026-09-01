"""Publish committed turn decisions at the Channel/Agent boundary."""

from __future__ import annotations

import logging
from typing import Any

from eidolon_sdk.biz.dialogue_control import CommittedTurnDecision

from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.session.committed_turn")


def publish_committed_turn_decision(
    *,
    factory: Any,
    decision: CommittedTurnDecision,
    timeline: TurnTimeline | None,
) -> bool:
    """Attach one transcript-bound decision to the next remote-brain turn."""

    metadata = decision.as_metadata()
    if timeline is not None:
        timeline.set_attr("committed_turn_decision", metadata)
    try:
        llm_plugin = getattr(getattr(factory, "llm", None), "llm", None)
        setter = getattr(llm_plugin, "set_turn_decision_metadata", None)
        if setter is None:
            return False
        setter(metadata)
        return True
    except Exception:
        logger.debug(
            "failed to publish committed turn decision",
            exc_info=True,
        )
        return False
