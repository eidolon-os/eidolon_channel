"""Half-duplex push-to-talk turn ownership.

The PTT owner is deliberately side-effect-light: it does not call LiveKit and it
does not decide full-duplex barge-in.  It only models the half-duplex product
contract: press opens a hold, release closes the user audio window, then the
channel waits briefly for VAD/STT evidence before emitting one terminal outcome.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

PttTurnState = Literal[
    "idle",
    "held",
    "released_finalizing",
    "committed",
    "rejected",
]
PttDecisionAction = Literal["none", "commit", "reject"]
_TIME_EPSILON_SEC = 1e-9


@dataclass(frozen=True)
class PttTurnOwnerConfig:
    empty_probe_sec: float = 0.25
    finalization_timeout_sec: float = 1.2
    post_vad_settle_sec: float = 0.15
    stable_interim_sec: float = 0.7


@dataclass(frozen=True)
class PttTurnDecision:
    action: PttDecisionAction = "none"
    reason: str = ""
    transcript: str = ""
    state: PttTurnState = "idle"
    terminal: bool = False
    next_delay_sec: float | None = None
    preempted_agent_output: bool = False


class PttTurnOwner:
    """Own one half-duplex PTT turn at a time."""

    def __init__(
        self,
        *,
        config: PttTurnOwnerConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._config = config or PttTurnOwnerConfig()
        self._clock = clock or time.monotonic
        self._counter = 0
        self.reset()

    @property
    def state(self) -> PttTurnState:
        return self._state

    @property
    def speech_detected(self) -> bool:
        return self._speech_detected

    @property
    def selected_text(self) -> str:
        return self._final_text or self._latest_text

    def reset(self) -> None:
        self._state: PttTurnState = "idle"
        self._hold_id = ""
        self._speech_detected = False
        self._vad_active = False
        self._pressed_at: float | None = None
        self._released_at: float | None = None
        self._vad_stopped_at: float | None = None
        self._last_transcript_at: float | None = None
        self._deadline_at: float | None = None
        self._latest_text = ""
        self._final_text = ""
        self._preempted_agent_output = False

    def press(
        self,
        *,
        preempted_agent_output: bool = False,
        now: float | None = None,
    ) -> PttTurnDecision:
        current = self._now(now)
        self._counter += 1
        self._state = "held"
        self._hold_id = f"ptt-hold-{self._counter}"
        self._speech_detected = False
        self._vad_active = False
        self._pressed_at = current
        self._released_at = None
        self._vad_stopped_at = None
        self._last_transcript_at = None
        self._deadline_at = None
        self._latest_text = ""
        self._final_text = ""
        self._preempted_agent_output = bool(preempted_agent_output)
        return PttTurnDecision(
            reason="pressed",
            state=self._state,
            preempted_agent_output=self._preempted_agent_output,
        )

    def vad_started(self, *, now: float | None = None) -> PttTurnDecision:
        if self._state not in {"held", "released_finalizing"}:
            return PttTurnDecision(reason="vad_ignored_not_active", state=self._state)
        self._speech_detected = True
        self._vad_active = True
        self._vad_stopped_at = None
        self._promote_to_speech_deadline()
        return PttTurnDecision(reason="vad_started", state=self._state)

    def vad_stopped(self, *, now: float | None = None) -> PttTurnDecision:
        if self._state not in {"held", "released_finalizing"}:
            return PttTurnDecision(reason="vad_stop_ignored_not_active", state=self._state)
        self._vad_active = False
        self._vad_stopped_at = self._now(now)
        return self.resolve(now=self._vad_stopped_at)

    def transcript(
        self,
        text: str,
        *,
        is_final: bool,
        now: float | None = None,
    ) -> PttTurnDecision:
        stripped = text.strip()
        if not stripped:
            return PttTurnDecision(reason="empty_transcript_ignored", state=self._state)
        if self._state not in {"held", "released_finalizing"}:
            return PttTurnDecision(reason="transcript_ignored_not_active", state=self._state)
        current = self._now(now)
        self._speech_detected = True
        self._latest_text = stripped
        self._last_transcript_at = current
        if is_final:
            self._final_text = stripped
        self._promote_to_speech_deadline()
        return self.resolve(now=current)

    def release(self, *, now: float | None = None) -> PttTurnDecision:
        current = self._now(now)
        if self._state != "held":
            return PttTurnDecision(reason="release_ignored_not_held", state=self._state)
        self._state = "released_finalizing"
        self._released_at = current
        timeout = (
            self._config.finalization_timeout_sec
            if self._speech_detected or self.selected_text
            else self._config.empty_probe_sec
        )
        self._deadline_at = current + max(0.0, timeout)
        return self.resolve(now=current)

    def resolve(self, *, now: float | None = None) -> PttTurnDecision:
        if self._state != "released_finalizing":
            return PttTurnDecision(reason="not_finalizing", state=self._state)

        current = self._now(now)
        if self._final_text:
            return self._commit("final_transcript")

        if not self._speech_detected and not self._latest_text:
            if self._deadline_elapsed(current):
                return self._reject("empty_hold")
            return self._wait("empty_probe", current)

        if self._latest_text and self._interim_is_stable(current):
            return self._commit("stable_interim_after_release")

        if self._deadline_elapsed(current):
            if self._latest_text:
                return self._commit("finalization_timeout_best_transcript")
            return self._reject("speech_without_transcript")

        return self._wait("awaiting_transcript_finalization", current)

    def _commit(self, reason: str) -> PttTurnDecision:
        self._state = "committed"
        return PttTurnDecision(
            action="commit",
            reason=reason,
            transcript=self.selected_text,
            state=self._state,
            terminal=True,
            preempted_agent_output=self._preempted_agent_output,
        )

    def _reject(self, reason: str) -> PttTurnDecision:
        self._state = "rejected"
        return PttTurnDecision(
            action="reject",
            reason=reason,
            state=self._state,
            terminal=True,
            preempted_agent_output=self._preempted_agent_output,
        )

    def _wait(self, reason: str, now: float) -> PttTurnDecision:
        return PttTurnDecision(
            reason=reason,
            state=self._state,
            next_delay_sec=self._next_delay(now),
            preempted_agent_output=self._preempted_agent_output,
        )

    def _interim_is_stable(self, now: float) -> bool:
        if not self._latest_text or self._last_transcript_at is None:
            return False
        if self._released_at is None:
            return False
        if not self._elapsed_at_least(now, self._released_at, self._config.stable_interim_sec):
            return False
        if self._vad_active:
            return False
        if self._vad_stopped_at is not None:
            if not self._elapsed_at_least(
                now,
                self._vad_stopped_at,
                self._config.post_vad_settle_sec,
            ):
                return False
        return self._elapsed_at_least(
            now,
            self._last_transcript_at,
            self._config.stable_interim_sec,
        )

    def _deadline_elapsed(self, now: float) -> bool:
        return self._deadline_at is not None and now + _TIME_EPSILON_SEC >= self._deadline_at

    def _promote_to_speech_deadline(self) -> None:
        if self._state != "released_finalizing" or self._released_at is None:
            return
        speech_deadline = self._released_at + max(0.0, self._config.finalization_timeout_sec)
        if self._deadline_at is None or self._deadline_at < speech_deadline:
            self._deadline_at = speech_deadline

    def _next_delay(self, now: float) -> float | None:
        waits: list[float] = []
        if self._deadline_at is not None:
            waits.append(max(0.0, self._deadline_at - now))
        if self._latest_text and self._last_transcript_at is not None:
            waits.append(
                max(
                    0.0,
                    self._last_transcript_at + self._config.stable_interim_sec - now,
                )
            )
        if self._latest_text and self._released_at is not None:
            waits.append(
                max(
                    0.0,
                    self._released_at + self._config.stable_interim_sec - now,
                )
            )
        if self._vad_stopped_at is not None:
            waits.append(
                max(
                    0.0,
                    self._vad_stopped_at + self._config.post_vad_settle_sec - now,
                )
            )
        positive = [value for value in waits if value > 0.0]
        if not positive:
            return 0.0
        return min(positive)

    def _now(self, now: float | None) -> float:
        return float(self._clock() if now is None else now)

    def _elapsed_at_least(self, now: float, since: float, duration: float) -> bool:
        return now - since + _TIME_EPSILON_SEC >= duration
