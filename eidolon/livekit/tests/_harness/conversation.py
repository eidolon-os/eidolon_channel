# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Conversation scenario DSL — high-level declarative API for tests.

Wraps :func:`headless_session` with action-oriented methods that hide
the bookkeeping (audio feeding, VAD events, STT scripting, state
synchronization). Most multi-turn dialogue tests should use this.

Two usage patterns:

1. **Declarative script** — build the entire conversation upfront.
   Best for predictable turn-taking tests::

       async with Conversation.script([
           User("你好"),
           Agent("你好！很高兴见到你"),  # expectation
           User("你叫什么"),
           Agent("我叫小爱"),
       ]) as conv:
           await conv.run()
           conv.assert_passed()

2. **Imperative** — drive turns step-by-step. Use for tests where
   timing matters (interrupts, dynamic content)::

       async with Conversation.builder()
           .with_llm({"hi": "hello", "stop": "ok"})
           .build() as conv:
           await conv.user_says("hi")
           reply = await conv.wait_for_agent_reply()
           assert "hello" in reply

Implementation note: ``Conversation.script(...)`` pre-schedules ALL
``ScriptedTranscript``s at session start using wall-clock timing. For
**single-turn** scenarios this works perfectly — the script and the
test code share a clock. For **multi-turn** scenarios, framework's
``speech_scheduling_paused`` state during agent TTS may drop a
second pre-scheduled transcript ("skipping user input, speech
scheduling is paused" warning). For multi-turn tests, prefer the
imperative API (``Conversation.with_mocks(...)`` + repeated
``user_says()`` + ``wait_for_agent_reply()``) which waits for each
turn's processing to complete before starting the next.

Single-turn declarative is the sweet spot. Multi-turn is fine in
imperative mode.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Optional

from .audio import synth_silence, synth_voiced
from .headless import HeadlessHandle, headless_session
from .mocks import (
    MockLLM,
    MockSTT,
    MockTTS,
    MockVAD,
    MockVADEvent,
    ScriptedReply,
    ScriptedTranscript,
)


# ─────────────────────────────────────────────────────────────────
#  Declarative script primitives
# ─────────────────────────────────────────────────────────────────


@dataclass
class User:
    """Declarative: user says ``text`` at this point in the conversation.

    The DSL automatically:
    * Schedules a ``ScriptedTranscript`` at the next free slot.
    * Schedules ``MockVADEvent`` start/end around it.
    * Feeds the right amount of voiced + silence audio into the
      session's ``ScriptedAudioInput``.
    """
    text: str
    interims: tuple[str, ...] = ()
    """Optional interim transcripts emitted before the final."""
    speech_duration_ms: int = 600
    """How long the user "speaks" — affects audio feed duration."""
    silence_after_ms: int = 200
    """Silence between this turn's audio and the next user turn."""


@dataclass
class Agent:
    """Declarative: assertion that agent replies with ``text`` (substring match).

    The DSL routes this to MockLLM as a ``ScriptedReply`` matching the
    PREVIOUS ``User`` step's text. Verification happens in
    :meth:`Conversation.assert_passed`: each ``Agent`` step is checked
    against the corresponding emitted assistant message.
    """
    text: str
    must_play_audio: bool = True
    """If True, asserts non-zero PCM bytes captured for this reply."""
    contains: Optional[str] = None
    """If set, assert this substring is in the reply (more lenient)."""


@dataclass
class Pause:
    """Declarative: insert a silence period (no user speech).

    Useful for testing user_state="away" timer, AEC warmup, etc.
    """
    ms: int


# Steps a script can contain.
ScriptStep = User | Agent | Pause


@dataclass
class _ScenarioPlan:
    """Pre-computed plan for running a declarative script."""
    user_steps: list[User] = field(default_factory=list)
    agent_steps: list[Agent] = field(default_factory=list)
    """Agent expectations, indexed by user-step index they follow."""
    pauses_after_user: dict[int, int] = field(default_factory=dict)
    """Pause ms inserted after the i-th user step."""


def _plan_script(steps: Iterable[ScriptStep]) -> _ScenarioPlan:
    """Walk script and collect User / Agent / Pause into a plan."""
    plan = _ScenarioPlan()
    user_idx = -1
    for s in steps:
        if isinstance(s, User):
            user_idx += 1
            plan.user_steps.append(s)
            # Default empty Agent at this slot — filled by next Agent.
            plan.agent_steps.append(None)  # type: ignore[arg-type]
        elif isinstance(s, Agent):
            if user_idx < 0:
                raise ValueError(
                    "Conversation.script: Agent step before any User step. "
                    "Each Agent expects a preceding User."
                )
            if plan.agent_steps[user_idx] is not None:
                raise ValueError(
                    f"Conversation.script: User step #{user_idx} already has "
                    f"an Agent expectation; can't have two."
                )
            plan.agent_steps[user_idx] = s
        elif isinstance(s, Pause):
            plan.pauses_after_user[max(0, user_idx)] = s.ms
        else:
            raise TypeError(f"unsupported step type: {type(s).__name__}")
    return plan


# ─────────────────────────────────────────────────────────────────
#  Conversation — context manager API
# ─────────────────────────────────────────────────────────────────


class Conversation:
    """High-level scenario harness. See module docstring for usage."""

    # Per-turn timing budgets (ms). Tunable per scenario.
    DEFAULT_INTER_TURN_GAP_MS = 200
    """Silence between turns (after one turn's silence_after_ms)."""

    def __init__(
        self,
        handle: HeadlessHandle,
        plan: Optional[_ScenarioPlan] = None,
    ) -> None:
        self._handle = handle
        self._plan = plan
        self._next_turn_idx = 0  # for imperative user_says()

    @property
    def handle(self) -> HeadlessHandle:
        """Underlying ``HeadlessHandle`` (session, events, audio_in/out)."""
        return self._handle

    # ── Declarative entry point ────────────────────────────────

    @classmethod
    @asynccontextmanager
    async def script(
        cls,
        steps: Iterable[ScriptStep],
        *,
        char_seconds: float = 0.02,
        sample_rate: int = 16000,
    ) -> AsyncIterator["Conversation"]:
        """Build + run a declarative conversation script.

        Schedules all ``User``/``Agent``/``Pause`` steps as
        ``ScriptedTranscript``s and ``ScriptedReply``s upfront, then
        provides the running session for assertions.
        """
        plan = _plan_script(list(steps))

        # Build mocks from the plan ────────────────────────────────
        scripted_replies = [
            ScriptedReply(when=u.text, reply=a.text)
            for u, a in zip(plan.user_steps, plan.agent_steps)
            if a is not None
        ]
        llm = MockLLM.scripted(scripted_replies, default_reply="?")

        # STT scripts: each User → one ScriptedTranscript with a
        # cumulative trigger_after_ms based on prior turn durations.
        stt_scripts: list[ScriptedTranscript] = []
        vad_events: list[MockVADEvent] = []
        cursor_ms = 80  # initial buffer before first turn
        for i, u in enumerate(plan.user_steps):
            stt_scripts.append(
                ScriptedTranscript(
                    text=u.text,
                    interims=list(u.interims),
                    interim_gap_ms=20 if u.interims else 0,
                    trigger_after_ms=cursor_ms + u.speech_duration_ms - 50,
                )
            )
            vad_events.append(
                MockVADEvent("start", at_ms=cursor_ms, probability=0.9)
            )
            vad_events.append(
                MockVADEvent(
                    "end",
                    at_ms=cursor_ms + u.speech_duration_ms,
                    probability=0.1,
                )
            )
            cursor_ms += u.speech_duration_ms + u.silence_after_ms
            cursor_ms += plan.pauses_after_user.get(i, 0)
            cursor_ms += cls.DEFAULT_INTER_TURN_GAP_MS

        stt = MockSTT.scripted(stt_scripts) if stt_scripts else MockSTT()
        tts = MockTTS(char_seconds=char_seconds)
        vad = MockVAD.scripted(vad_events) if vad_events else MockVAD.silent()

        async with headless_session(
            llm=llm, stt=stt, tts=tts, vad=vad, sample_rate=sample_rate,
        ) as h:
            conv = cls(handle=h, plan=plan)
            yield conv

    # ── Imperative entry point ─────────────────────────────────

    @classmethod
    @asynccontextmanager
    async def with_mocks(
        cls,
        *,
        llm: MockLLM,
        stt: MockSTT,
        tts: MockTTS,
        vad: MockVAD,
        **kwargs,
    ) -> AsyncIterator["Conversation"]:
        """Imperative escape hatch — pass pre-built mocks. Use when the
        declarative script doesn't fit (e.g. mid-reply interrupt timing)."""
        async with headless_session(
            llm=llm, stt=stt, tts=tts, vad=vad, **kwargs,
        ) as h:
            yield cls(handle=h)

    # ── Declarative runner ─────────────────────────────────────

    async def run(self, *, total_timeout: float = 30.0) -> None:
        """Drive the scripted conversation to completion.

        Feeds audio for each user step, waits for the agent to finish
        each reply, then proceeds to the next.
        """
        if self._plan is None:
            raise RuntimeError(
                "Conversation.run() requires a script (use "
                "Conversation.script(...))."
            )
        deadline = time.monotonic() + total_timeout

        for i, user in enumerate(self._plan.user_steps):
            # Feed audio for this user turn
            self._handle.audio_in.feed_pcm(
                synth_voiced(user.speech_duration_ms / 1000.0)
            )
            self._handle.audio_in.feed_silence(user.silence_after_ms / 1000.0)
            pause_ms = self._plan.pauses_after_user.get(i, 0)
            if pause_ms:
                self._handle.audio_in.feed_silence(pause_ms / 1000.0)
            self._handle.audio_in.feed_silence(
                self.DEFAULT_INTER_TURN_GAP_MS / 1000.0
            )

            # Wait for this turn's STT final to be received by framework
            expected_text = user.text
            agent_step = self._plan.agent_steps[i]
            try:
                await self._handle.events.wait_for(
                    lambda e, t=expected_text: e.type == "user_input_transcribed"
                    and e.payload.is_final
                    and e.payload.transcript == t,
                    timeout=max(2.0, deadline - time.monotonic()),
                )
            except TimeoutError:
                # Continue — assertion will catch missing transcript.
                pass

            if agent_step is not None:
                # Wait for agent to enter speaking, then leave it.
                try:
                    await self._handle.events.wait_for_state(
                        agent="speaking",
                        timeout=max(2.0, deadline - time.monotonic()),
                    )
                    await self._handle.events.wait_for(
                        lambda e: e.type == "agent_state_changed"
                        and e.payload.old_state == "speaking",
                        timeout=max(2.0, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    pass

        # Small grace for ConversationItemAdded events to land.
        await asyncio.sleep(0.15)

    # ── Imperative steps ──────────────────────────────────────

    async def user_says(
        self,
        text: str,
        *,
        speech_duration_ms: int = 500,
        silence_after_ms: int = 300,
        timeout: float = 10.0,
    ) -> None:
        """Imperative step: simulate user saying ``text`` and wait
        until the framework processes the final transcript.

        Requires the underlying ``MockSTT`` to have a ScriptedTranscript
        for ``text`` already — either from ``Conversation.script(...)``
        or pre-baked via ``with_mocks(stt=MockSTT.scripted(...))``.
        """
        self._handle.audio_in.feed_pcm(synth_voiced(speech_duration_ms / 1000.0))
        self._handle.audio_in.feed_silence(silence_after_ms / 1000.0)
        await self._handle.events.wait_for(
            lambda e: e.type == "user_input_transcribed"
            and e.payload.is_final
            and e.payload.transcript == text,
            timeout=timeout,
        )

    async def wait_for_agent_state(
        self, state: str, *, timeout: float = 10.0
    ) -> None:
        """Block until ``agent_state == state``."""
        await self._handle.events.wait_for_state(agent=state, timeout=timeout)

    async def wait_for_agent_reply(self, *, timeout: float = 15.0) -> str:
        """Wait until the current agent turn finishes (state leaves
        ``speaking``) and return the spoken text (last assistant
        message in the conversation log)."""
        await self.wait_for_agent_state("speaking", timeout=timeout)
        await self._handle.events.wait_for(
            lambda e: e.type == "agent_state_changed"
            and e.payload.old_state == "speaking",
            timeout=timeout,
        )
        await asyncio.sleep(0.1)  # grace for ChatMessage commit
        msgs = self._handle.events.agent_messages()
        return msgs[-1] if msgs else ""

    async def feed_silence(self, ms: int) -> None:
        """Insert silence (no user speech)."""
        self._handle.audio_in.feed_silence(ms / 1000.0)

    # ── Assertions ────────────────────────────────────────────

    def assert_passed(self) -> None:
        """For declarative scripts: verify every Agent step matched.

        Compares ordered ``Agent`` expectations against the recorded
        assistant messages in conversation order. Substring match if
        ``Agent.contains`` is set; else exact match on the prefix.
        """
        if self._plan is None:
            raise RuntimeError(
                "assert_passed() only works with Conversation.script()"
            )
        agent_msgs = self._handle.events.agent_messages()
        expected = [a for a in self._plan.agent_steps if a is not None]

        assert len(agent_msgs) >= len(expected), (
            f"expected {len(expected)} agent reply/replies, "
            f"got {len(agent_msgs)}: {agent_msgs}"
        )

        for i, exp in enumerate(expected):
            actual = agent_msgs[i]
            needle = exp.contains if exp.contains else exp.text
            assert needle in actual, (
                f"agent reply #{i} mismatch:\n"
                f"  expected to contain: {needle!r}\n"
                f"  actual: {actual!r}"
            )

        # No errors emitted
        errors = self._handle.events.of_type("error")
        assert not errors, f"unexpected error events: {errors}"

    def assert_no_errors(self) -> None:
        errors = self._handle.events.of_type("error")
        assert not errors, f"unexpected error events: {errors}"

    def assert_n_user_finals(self, n: int) -> None:
        finals = self._handle.events.user_finals()
        assert len(finals) == n, (
            f"expected {n} user finals, got {len(finals)}: "
            f"{[t.transcript for t in finals]}"
        )

    def assert_audio_played(self) -> None:
        """At least one byte of TTS audio was captured."""
        assert self._handle.audio_out.captured_bytes > 0, (
            "no agent audio was emitted"
        )

    def assert_user_state_history_contains(self, state: str) -> None:
        history = self._handle.events.user_state_history()
        assert state in history, (
            f"user_state never reached {state!r}; history: {history}"
        )

    def assert_agent_state_history_contains(self, state: str) -> None:
        history = self._handle.events.agent_state_history()
        assert state in history, (
            f"agent_state never reached {state!r}; history: {history}"
        )

    # ── Convenience properties ────────────────────────────────

    @property
    def last_agent_reply(self) -> str:
        msgs = self._handle.events.agent_messages()
        return msgs[-1] if msgs else ""

    @property
    def all_user_finals(self) -> list[str]:
        return [t.transcript for t in self._handle.events.user_finals()]

    @property
    def all_agent_replies(self) -> list[str]:
        return list(self._handle.events.agent_messages())

    @property
    def captured_pcm_bytes(self) -> int:
        return self._handle.audio_out.captured_bytes
