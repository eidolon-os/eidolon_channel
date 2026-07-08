"""Pure turn-completion policy helpers for full-duplex turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..session.voiceprint_reasons import is_voiceprint_inconclusive_reason


@dataclass(frozen=True)
class LowEotCommitDecision:
    """Decision for whether a low-EOT turn should wait before commit."""

    should_defer: bool
    reason: str
    eot_score: float


def eot_score_from_model(eot_model: Any, *, default: float) -> float:
    """Read the current EOT score from either public or legacy model fields."""

    if eot_model is None:
        return default
    return float(
        getattr(
            eot_model,
            "current_eot_score",
            getattr(eot_model, "_current_eot_score", default),
        )
        or 0.0
    )


def eot_thinks_turn_complete(
    eot_model: Any,
    *,
    unlikely_threshold: float,
) -> bool:
    """Return the framework-completed hook's EOT completeness contract."""

    if eot_model is None:
        return True
    return eot_score_from_model(eot_model, default=1.0) >= unlikely_threshold


def decide_low_eot_commit_deferral(
    transcript: str,
    *,
    eot_score: float,
    unlikely_threshold: float,
    short_statement_defer_max_cjk_chars: int,
) -> LowEotCommitDecision:
    """Decide whether speech-stop commit should wait for more evidence."""

    if not transcript.strip():
        return LowEotCommitDecision(
            should_defer=False,
            reason="empty_transcript",
            eot_score=eot_score,
        )
    if eot_score < unlikely_threshold:
        return LowEotCommitDecision(
            should_defer=True,
            reason="eot_score_unlikely",
            eot_score=eot_score,
        )
    if looks_like_short_statement_continuation(
        transcript,
        max_cjk_chars=short_statement_defer_max_cjk_chars,
    ):
        return LowEotCommitDecision(
            should_defer=True,
            reason="short_statement_continuation",
            eot_score=eot_score,
        )
    return LowEotCommitDecision(
        should_defer=False,
        reason="turn_complete",
        eot_score=eot_score,
    )


def looks_like_short_statement_continuation(
    transcript: str,
    *,
    max_cjk_chars: int,
) -> bool:
    """Return true for short CJK fragments that often precede continuation."""

    text = transcript.strip()
    if not text:
        return False
    if any(mark in text for mark in ("?", "!", "？", "！")):
        return False
    cjk_chars = count_cjk_chars(text)
    if cjk_chars <= 0:
        return False
    if cjk_chars > max_cjk_chars:
        return False
    if text.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
        return False
    return text.endswith(("。", "，", ",", "、", "的", "了", "呢", "吧"))


def count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


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


def should_wait_for_inconclusive_voiceprint_merge(
    *,
    commit_reason: str,
    candidate_state: str,
    selected_text: str,
    transcript: str,
    short_statement_defer_max_cjk_chars: int,
) -> bool:
    """Pure merge-wait contract for inconclusive voiceprint results."""

    if not is_voiceprint_inconclusive_reason(commit_reason):
        return False
    if candidate_state == "waiting_merge":
        return True
    if candidate_state in {"committed", "rejected", ""}:
        return False
    selected = selected_text or transcript
    return looks_like_short_statement_continuation(
        selected,
        max_cjk_chars=short_statement_defer_max_cjk_chars,
    )
