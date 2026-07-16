"""No-evidence fast resume for false barge-ins (2026-07, from web dogfood).

Web dogfood turn ``2e8f3cf3``: a ~700ms false trigger with **no transcript at
all** kept the agent suspended (silent) for the full
``post_speech_evidence_timeout_ms`` = 6000ms before resuming
(``deadline_hold_max_suspend_elapsed suspend=6.00s``).

A real interruption yields a transcript quickly (interim during speech, final
within a few hundred ms of VAD end). So total silence past a short no-evidence
grace is almost certainly a false trigger, and the orchestrator should stop
holding the duck deadline — letting the agent resume promptly — instead of
waiting the full evidence window. A real (STT-lagging) interrupt that *does*
produce a transcript still gets the full window.

These test the orchestrator's decision surface (``should_hold_deadline`` /
``max_suspend_sec``); the deadline handler turns "stop holding" into an actual
resume (covered end-to-end by the web dogfood re-run).
"""

from __future__ import annotations

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.session.interruption_orchestrator import (
    InterruptionOrchestrator,
    InterruptionState,
)

_EVIDENCE = 6.0
_NO_EVIDENCE = 0.8
_SPEECH_MS = 0.70  # long enough (> min_speech) to enter the post-speech wait


class _Clock:
    def __init__(self, t: float = 10.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _owner(clock: _Clock) -> InterruptionOrchestrator:
    return InterruptionOrchestrator(
        evidence_timeout_sec=_EVIDENCE,
        min_speech_sec=0.25,
        no_evidence_timeout_sec=_NO_EVIDENCE,
        clock=clock,
    )


def _enter_no_transcript_wait(clock: _Clock) -> InterruptionOrchestrator:
    owner = _owner(clock)
    owner.start_candidate(timeline=TurnTimeline("turn-1"))
    clock.t += _SPEECH_MS
    deferred = owner.defer_false_resume_after_speech_end(transcript="", duck_suspended=True)
    assert deferred is True
    assert owner.state is InterruptionState.SUSPENDED_POST_SPEECH_WAIT
    return owner


def test_no_transcript_holds_before_grace() -> None:
    clock = _Clock()
    owner = _enter_no_transcript_wait(clock)
    # still within the no-evidence grace -> keep holding (wait for lagging STT)
    clock.t += _NO_EVIDENCE - 0.2
    assert owner.should_hold_deadline() is True


def test_no_transcript_resumes_after_grace() -> None:
    clock = _Clock()
    owner = _enter_no_transcript_wait(clock)
    # past the no-evidence grace with still no transcript -> stop holding (resume),
    # far short of the full 6s evidence window.
    clock.t += _NO_EVIDENCE + 0.05
    assert owner.should_hold_deadline() is False


def test_no_transcript_max_suspend_capped() -> None:
    clock = _Clock()
    owner = _enter_no_transcript_wait(clock)
    assert owner.max_suspend_sec() == _NO_EVIDENCE  # not the full 6s


def test_no_transcript_max_suspend_caps_active_candidate_before_vad_end() -> None:
    clock = _Clock()
    owner = _owner(clock)
    owner.start_candidate(timeline=TurnTimeline("turn-1"))

    assert owner.max_suspend_sec() == _NO_EVIDENCE


def test_transcript_present_keeps_full_evidence_window() -> None:
    clock = _Clock()
    owner = _owner(clock)
    owner.start_candidate(timeline=TurnTimeline("turn-1"))
    clock.t += _SPEECH_MS
    # real interruption text arrived -> full window, no early resume
    owner.defer_false_resume_after_speech_end(transcript="那你现在", duck_suspended=True)
    clock.t += _NO_EVIDENCE + 0.05
    assert owner.should_hold_deadline() is True
    assert owner.max_suspend_sec() == _EVIDENCE


def test_transcript_arriving_during_grace_extends_to_full_window() -> None:
    clock = _Clock()
    owner = _enter_no_transcript_wait(clock)
    # a lagging STT interim lands within the grace -> now treated as real
    owner.note_transcript("那你现在", is_final=False)
    clock.t += _NO_EVIDENCE + 0.05
    assert owner.should_hold_deadline() is True
    assert owner.max_suspend_sec() == _EVIDENCE


def test_default_preserves_full_window_behavior() -> None:
    # No no_evidence_timeout_sec passed -> defaults to the full evidence window,
    # so pre-existing callers keep prior behavior (no early resume).
    clock = _Clock()
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=_EVIDENCE,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("turn-1"))
    clock.t += _SPEECH_MS
    owner.defer_false_resume_after_speech_end(transcript="", duck_suspended=True)
    clock.t += _NO_EVIDENCE + 0.05
    assert owner.should_hold_deadline() is True
    assert owner.max_suspend_sec() == _EVIDENCE
