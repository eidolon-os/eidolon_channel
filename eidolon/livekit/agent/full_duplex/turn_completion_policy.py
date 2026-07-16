"""Pure turn-completion policy helpers for full-duplex turns."""

from __future__ import annotations

from typing import Any

from ..session.voiceprint_reasons import is_voiceprint_inconclusive_reason


def select_combined_voiceprint_result(results: list[Any]) -> Any | None:
    """Select the effective voiceprint result from multiple candidate tasks."""

    if not results:
        return None
    inconclusive = None
    for result in results:
        if not bool(getattr(result, "commit_allowed", False)):
            if voiceprint_result_is_inconclusive(result):
                inconclusive = inconclusive or result
                continue
            return result
    if bool(getattr(results[-1], "commit_allowed", False)):
        return results[-1]
    if inconclusive is not None:
        return inconclusive
    return results[-1]


def voiceprint_result_is_inconclusive(result: Any) -> bool:
    reason = str(getattr(result, "commit_reason", "") or "")
    return is_voiceprint_inconclusive_reason(reason)
