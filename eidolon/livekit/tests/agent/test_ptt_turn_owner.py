from __future__ import annotations

from eidolon.livekit.agent.session.ptt_turn import (
    PttTurnOwner,
    PttTurnOwnerConfig,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def _owner(clock: Clock) -> PttTurnOwner:
    return PttTurnOwner(
        config=PttTurnOwnerConfig(
            empty_probe_sec=0.25,
            finalization_timeout_sec=1.2,
            post_vad_settle_sec=0.15,
            stable_interim_sec=0.7,
        ),
        clock=clock,
    )


def test_empty_quick_tap_rejects_after_probe_not_on_release() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press()
    decision = owner.release()

    assert decision.action == "none"
    assert decision.reason == "empty_probe"
    assert decision.next_delay_sec == 0.25

    decision = owner.resolve(now=clock.advance(0.25))

    assert decision.action == "reject"
    assert decision.reason == "empty_hold"


def test_vad_before_transcript_prevents_false_empty_release() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press()
    owner.vad_started(now=clock.advance(0.05))
    decision = owner.release(now=clock.advance(0.1))

    assert decision.action == "none"
    assert decision.reason == "awaiting_transcript_finalization"

    decision = owner.transcript("给我讲个小笑话。", is_final=True, now=clock.advance(0.45))

    assert decision.action == "commit"
    assert decision.reason == "final_transcript"
    assert decision.transcript == "给我讲个小笑话。"


def test_preempted_spoken_text_still_commits_as_user_turn() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press(preempted_agent_output=True)
    owner.vad_started(now=clock.advance(0.05))
    owner.transcript("停，不要说了。", is_final=True, now=clock.advance(0.2))
    decision = owner.release(now=clock.advance(0.1))

    assert decision.action == "commit"
    assert decision.reason == "final_transcript"
    assert decision.transcript == "停，不要说了。"
    assert decision.preempted_agent_output is True


def test_idle_hard_stop_still_commits_as_normal_ptt_utterance() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press(preempted_agent_output=False)
    owner.vad_started(now=clock.advance(0.05))
    owner.transcript("停，不要说了。", is_final=True, now=clock.advance(0.2))
    decision = owner.release(now=clock.advance(0.1))

    assert decision.action == "commit"
    assert decision.reason == "final_transcript"


def test_release_waits_for_late_final_instead_of_committing_stale_interim() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press()
    owner.vad_started(now=clock.advance(0.05))
    owner.transcript("你告诉我现在", is_final=False, now=clock.advance(0.2))
    decision = owner.release(now=clock.advance(0.3))

    assert decision.action == "none"
    assert decision.reason == "awaiting_transcript_finalization"

    decision = owner.resolve(now=clock.advance(0.3))
    assert decision.action == "none"

    decision = owner.transcript("你告诉我现在几点了。", is_final=True, now=clock.advance(0.25))

    assert decision.action == "commit"
    assert decision.transcript == "你告诉我现在几点了。"


def test_speech_without_final_commits_stable_interim_after_release_window() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press()
    owner.vad_started()
    owner.transcript("现在几点", is_final=False, now=clock.advance(0.2))
    owner.vad_stopped(now=clock.advance(0.1))
    owner.release(now=clock.advance(0.1))

    decision = owner.resolve(now=clock.advance(0.69))
    assert decision.action == "none"

    decision = owner.resolve(now=clock.advance(0.01))
    assert decision.action == "commit"
    assert decision.reason == "stable_interim_after_release"
    assert decision.transcript == "现在几点"


def test_speech_without_transcript_rejects_at_finalization_deadline() -> None:
    clock = Clock()
    owner = _owner(clock)

    owner.press()
    owner.vad_started(now=clock.advance(0.1))
    owner.release(now=clock.advance(0.1))
    decision = owner.resolve(now=clock.advance(1.19))
    assert decision.action == "none"

    decision = owner.resolve(now=clock.advance(0.01))

    assert decision.action == "reject"
    assert decision.reason == "speech_without_transcript"
