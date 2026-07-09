"""Regression lock: no pure-text CJK fast-lane cancel (2026-07-09 measurement).

Decision record — see
``docs/子项目/eidolon_channel/打断与轮次/全双工打断延迟实测复盘-20260709.md``.

A "substantive CJK interim -> cancel immediately" fast-lane was tried to cut a
perceived ~1984ms barge-in latency, then **rejected** after web-dogfood
measurement:

  - The interruption/cancel LINK is already healthy (duck ~0.5ms, cancel
    ~552ms). The ~1984ms was ``speech→commit`` (the user's own speaking time),
    not a cancel delay.
  - A pure-text fast-lane broke the committed dogfood contract that *substantive
    text alone must not cancel* — cancelling needs an EOT semantic score or an
    explicit lexicon intent. Root cause: EOT is an end-of-turn signal, low
    mid-utterance by design, so a substantive interim is indistinguishable from
    a preamble ("我想问一下…") without a model. Cancelling on text alone is the
    over-cancel anti-pattern the team had already eliminated.

This test locks that decision: a substantive CJK interim with a low EOT score
must HOLD (wait for evidence), never CANCEL, and must not resolve via any
``fast_lane`` source. If someone re-introduces a text-length fast-lane, this
fails.
"""

from __future__ import annotations

import pytest

from eidolon.livekit.agent.turn_policy import Action, InterruptDecider

_LOW_SCORE = 0.10  # mid-utterance: EOT not yet confident (end-of-turn signal)


@pytest.mark.parametrize(
    "text",
    [
        "那你现在能帮我做什么",  # the dogfood phrase (long, substantive)
        "那你现在能",  # 5 CJK — would have tripped a length-based fast-lane
        "我想问一下你",  # a *preamble*: substantive but NOT an interrupt yet
        "嗯我给你弄了",  # dogfood: low-score long interim must wait
    ],
)
def test_substantive_cjk_interim_low_score_holds_not_cancel(text: str) -> None:
    d = InterruptDecider()
    decision = d.on_stt_interim(text, _LOW_SCORE, vad_active=True, agent_speaking=True)

    # Must NOT hard-cancel on text alone (no pure-text fast-lane).
    assert decision.action is Action.HOLD, (
        f"substantive CJK interim {text!r} at low EOT score must HOLD, "
        f"got {decision.action} ({decision.reason})"
    )
    assert decision.intent_source != "fast_lane"
    assert "fast_lane" not in decision.reason


def test_high_eot_score_still_cancels() -> None:
    # The healthy path is unchanged: a high EOT semantic score cancels. The
    # rejection above is specifically about *text length* short-circuiting the
    # score, not about cancelling ever.
    d = InterruptDecider()
    decision = d.on_stt_interim(
        "那你现在能帮我做什么", 0.85, vad_active=True, agent_speaking=True
    )
    assert decision.action is Action.CANCEL
    assert decision.intent_source == "eot"


def test_explicit_hard_stop_lexicon_still_cancels() -> None:
    # Explicit intent (lexicon) is a legitimate cancel signal — not affected by
    # the no-fast-lane rule.
    d = InterruptDecider()
    decision = d.on_stt_interim("停一下", 0.0, vad_active=True, agent_speaking=True)
    assert decision.action is Action.CANCEL
