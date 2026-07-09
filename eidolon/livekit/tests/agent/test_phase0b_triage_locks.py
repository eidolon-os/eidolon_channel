"""Regression locks for Phase 0b triage conclusions (2026-07-09).

The code-review leads triaged as "not a live bug" because production already
does the right thing. These tests pin that behavior so a future refactor can't
silently regress it. See
``docs/子项目/eidolon_channel/打断与轮次/全双工打断延迟实测复盘-20260709.md``.

Covered:
  - L4: the post-speech interruption commit reuses ``UserTurnCoordinator.
    finish_speech`` (it is *not* a hand-rolled commit), so the meta / empty
    rejection filters apply. We lock the filter at the ``finish_speech`` seam
    that ``commit_candidate`` delegates to.
  - L6: ``select_combined_voiceprint_result`` is order-deterministic — a
    definitive reject wins regardless of its position (and the caller feeds it
    ``asyncio.gather`` output, which preserves input order, not completion
    order).

(L11's refutation — callback cleared in ``finally`` + overwritten on the next
stream — is structural; a unit test would be brittle, so it is documented in
the retro rather than locked here.)
"""

from __future__ import annotations

from types import SimpleNamespace

from eidolon.livekit.agent.full_duplex.turn_completion_policy import (
    select_combined_voiceprint_result,
)
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


# --- L4: finish_speech rejects a non-actionable meta turn -------------------


def _coordinator() -> UserTurnCoordinator:
    return UserTurnCoordinator(
        merge_grace_sec=0.8,
        low_eot_delay_sec=0.8,
        clock=_Clock(),
    )


def test_finish_speech_rejects_non_actionable_meta_turn() -> None:
    # "我再说一下" is a non-actionable meta turn (prefix "我再说" + suffix "一下").
    # The post-speech committer (commit_candidate) delegates to finish_speech
    # and returns False on a reject decision, so this reject blocks a spurious
    # post-speech commit too.
    coordinator = _coordinator()
    coordinator.start_speech(timeline=TurnTimeline("turn-meta"))
    coordinator.add_transcript("我再说一下", is_final=True)

    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "reject"


def test_finish_speech_commits_ordinary_actionable_turn() -> None:
    # Contrast: a real request commits (proves the meta filter isn't over-broad).
    coordinator = _coordinator()
    coordinator.start_speech(timeline=TurnTimeline("turn-ok"))
    coordinator.add_transcript("帮我查一下天气", is_final=True)

    decision = coordinator.finish_speech(eot_score=0.9, should_defer=False)

    assert decision.action == "commit"
    assert decision.transcript == "帮我查一下天气"


# --- L6: voiceprint result selection is order-deterministic -----------------


def _allow() -> SimpleNamespace:
    return SimpleNamespace(commit_allowed=True, commit_reason="ok")


def _reject() -> SimpleNamespace:
    return SimpleNamespace(commit_allowed=False, commit_reason="voiceprint_mismatch")


def test_definitive_reject_wins_regardless_of_order() -> None:
    # A reject blocks commit whether it arrives first or last — the outcome does
    # not depend on task completion order (gather preserves input order anyway).
    assert select_combined_voiceprint_result([_reject(), _allow()]).commit_allowed is False
    assert select_combined_voiceprint_result([_allow(), _reject()]).commit_allowed is False


def test_all_allow_commits() -> None:
    assert select_combined_voiceprint_result([_allow(), _allow()]).commit_allowed is True


def test_empty_results_returns_none() -> None:
    assert select_combined_voiceprint_result([]) is None
