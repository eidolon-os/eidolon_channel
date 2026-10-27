"""Regression locks for Phase 0b triage conclusions (2026-07-09).

The code-review leads triaged as "not a live bug" because production already
does the right thing. These tests pin that behavior so a future refactor can't
silently regress it. See
``docs/子项目/eidolon_channel/打断与轮次/全双工打断延迟实测复盘-20260709.md``.

Covered:
  - L4 correction: the transcript assembler preserves accepted text and does
    not classify language with a phrase list. Semantic handling belongs to the
    policy/brain boundary.
  - L6 was retired with voiceprint admission; it is no longer product behavior.

(L11's refutation — callback cleared in ``finally`` + overwritten on the next
stream — is structural; a unit test would be brittle, so it is documented in
the retro rather than locked here.)
"""

from __future__ import annotations


from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.user_turn_coordinator import UserTurnCoordinator


# --- L4: framework completion preserves accepted transcript text ------------


def _coordinator() -> UserTurnCoordinator:
    return UserTurnCoordinator(speech_merge_grace_sec=0.8)


def test_framework_completion_does_not_classify_meta_language() -> None:
    coordinator = _coordinator()
    timeline = TurnTimeline("turn-meta")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("我再说一下", is_final=True)

    decision = coordinator.mark_framework_completed(
        transcript="我再说一下",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "我再说一下"


def test_framework_completion_commits_ordinary_actionable_turn() -> None:
    # Contrast: a real request commits (proves the meta filter isn't over-broad).
    coordinator = _coordinator()
    timeline = TurnTimeline("turn-ok")
    coordinator.start_speech(timeline=timeline)
    coordinator.add_transcript("帮我查一下天气", is_final=True)

    decision = coordinator.mark_framework_completed(
        transcript="帮我查一下天气",
        reason="framework_completed_turn",
        timeline=timeline,
    )

    assert decision.action == "commit"
    assert decision.transcript == "帮我查一下天气"
