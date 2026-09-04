"""Product contracts for learned EOT and full-duplex turn taking.

These tests deliberately cross the boundaries that the smaller policy tests do
not: the bundled ONNX EOT model supplies the score, the production turn runtime
decides the interruption, and the interruption owner determines whether the
utterance is allowed to reach the LLM.

They are fail-first regression tests for the September 2026 turn-taking review.
Do not replace the model output with hand-authored ``eot_scores``: doing so would
hide the coupling this suite is intended to detect.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from livekit.agents.llm import ChatContext

from eidolon.livekit.agent.full_duplex.agent_builder import build_full_duplex_agent
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output import DuckingStats
from eidolon.livekit.agent.session.interruption_orchestrator import (
    InterruptionOrchestrator,
)
from eidolon.livekit.agent.session.semantic_interrupt import SemanticInterruptHandler
from eidolon.livekit.agent.turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
)
from eidolon.livekit.common.config import TurnPolicyConfig
from eidolon.livekit.plugins.eot import ChineseModel
from eidolon.livekit.tests._harness.audio import synth_voiced
from eidolon.livekit.tests._harness.headless import headless_session
from eidolon.livekit.tests._harness.mocks import (
    MockLLM,
    MockSTT,
    MockTTS,
    MockVAD,
    MockVADEvent,
    ScriptedTranscript,
)


_KNOWN_EOT_MODEL_GAP = pytest.mark.xfail(
    strict=True,
    reason="learned EOT sends complete Chinese turns to LiveKit's long-delay band",
)
_KNOWN_INTERRUPT_SEMANTICS_GAP = pytest.mark.xfail(
    strict=True,
    reason="production has no non-lexical interruption intent implementation",
)
_KNOWN_ENDPOINTING_GAP = pytest.mark.xfail(
    strict=True,
    reason="Eidolon endpointing bounds are not bound to LiveKit Agent options",
)
_KNOWN_SINGLE_OWNER_GAP = pytest.mark.xfail(
    strict=True,
    reason="ducked transcripts currently run both EOT and turn-runtime policies",
)
_KNOWN_TERMINAL_HOLD_GAP = pytest.mark.xfail(
    strict=True,
    reason="a final transcript can currently retain the full six-second hold budget",
)


@pytest.fixture(scope="module")
def eot_model() -> ChineseModel:
    return ChineseModel()


@_KNOWN_EOT_MODEL_GAP
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "不要讲了",
        "换个话题吧",
        "不是，我说的是明天",
        "就这样吧",
    ],
)
async def test_complete_utterance_clears_framework_eot_threshold(
    eot_model: ChineseModel,
    text: str,
) -> None:
    """Complete Chinese turns must not fall into LiveKit's long-delay band."""

    chat = ChatContext()
    chat.add_message(role="user", content=[text])

    score = await eot_model.predict_end_of_turn(chat)
    threshold = await eot_model.unlikely_threshold("zh-CN")

    assert threshold is not None
    assert score >= threshold, (
        f"complete utterance {text!r} scored {score:.3f} below the "
        f"framework EOT threshold {threshold:.3f}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["我觉得", "因为这个", "然后", "还有就是"])
async def test_incomplete_utterance_stays_below_framework_eot_threshold(
    eot_model: ChineseModel,
    text: str,
) -> None:
    """Incomplete prefixes must retain the user's floor."""

    chat = ChatContext()
    chat.add_message(role="user", content=[text])

    score = await eot_model.predict_end_of_turn(chat)
    threshold = await eot_model.unlikely_threshold("zh-CN")

    assert threshold is not None
    assert score < threshold


@_KNOWN_ENDPOINTING_GAP
@pytest.mark.asyncio
async def test_livekit_session_commits_complete_turn_within_hard_slo() -> None:
    """Exercise the real LiveKit endpointing task instead of a policy helper."""

    model = ChineseModel()
    async with headless_session(
        llm=MockLLM.scripted([("就这样吧", "好的")]),
        stt=MockSTT.scripted(
            [ScriptedTranscript(text="就这样吧", trigger_after_ms=80)]
        ),
        tts=MockTTS(char_seconds=0.01),
        vad=MockVAD.scripted(
            [
                MockVADEvent("start", at_ms=10, probability=0.9),
                MockVADEvent("end", at_ms=180, probability=0.1),
            ]
        ),
        extra_agent_kwargs={"turn_handling": {"turn_detection": model}},
    ) as handle:
        handle.audio_in.feed_pcm(synth_voiced(0.2))
        handle.audio_in.feed_silence(0.2)

        speech_started = await handle.events.wait_for(
            lambda event: event.type == "user_state_changed"
            and event.payload.new_state == "speaking",
            timeout=2.0,
        )
        speech_stopped = await handle.events.wait_for(
            lambda event: event.type == "user_state_changed"
            and event.payload.new_state == "listening"
            and event.timestamp > speech_started.timestamp,
            timeout=2.0,
        )
        thinking = await handle.events.wait_for(
            lambda event: event.type == "agent_state_changed"
            and event.payload.new_state == "thinking"
            and event.timestamp > speech_stopped.timestamp,
            timeout=5.0,
        )

        speech_stop_to_thinking = thinking.timestamp - speech_stopped.timestamp
        assert speech_stop_to_thinking <= 1.5
        assert len(handle.events.user_messages()) == 1


@dataclass(frozen=True)
class _PlaybackCase:
    text: str
    action: Action
    intent: InterruptIntent
    continue_to_llm: bool


_PLAYBACK_CASES = (
    pytest.param(
        _PlaybackCase("停一下", Action.CANCEL, InterruptIntent.HARD_STOP, False),
        marks=_KNOWN_INTERRUPT_SEMANTICS_GAP,
        id="hard_stop_short",
    ),
    pytest.param(
        _PlaybackCase("不要讲了", Action.CANCEL, InterruptIntent.HARD_STOP, False),
        marks=_KNOWN_INTERRUPT_SEMANTICS_GAP,
        id="hard_stop_phrase",
    ),
    pytest.param(
        _PlaybackCase("换个话题吧", Action.CANCEL, InterruptIntent.TOPIC_SWITCH, True),
        marks=_KNOWN_INTERRUPT_SEMANTICS_GAP,
        id="topic_switch",
    ),
    pytest.param(
        _PlaybackCase("不是，我说的是明天", Action.CANCEL, InterruptIntent.CORRECTION, True),
        marks=_KNOWN_INTERRUPT_SEMANTICS_GAP,
        id="correction",
    ),
    pytest.param(
        _PlaybackCase("好的", Action.ROLLBACK, InterruptIntent.BACKCHANNEL, False),
        marks=_KNOWN_INTERRUPT_SEMANTICS_GAP,
        id="backchannel",
    ),
    pytest.param(
        _PlaybackCase(
            "那你现在能帮我做什么",
            Action.CANCEL,
            InterruptIntent.NORMAL_INTERRUPT,
            True,
        ),
        id="normal_interrupt",
    ),
)


@pytest.mark.parametrize("case", _PLAYBACK_CASES)
def test_playback_interruption_semantics_do_not_depend_on_eot_completeness(
    eot_model: ChineseModel,
    case: _PlaybackCase,
) -> None:
    """EOT completeness is a feature, not the interruption intent itself."""

    eot_model.reset()
    eot_model.update_vad(False)
    score = eot_model.update_asr(case.text, is_final=True)
    runtime = TurnPolicyRuntime(TurnPolicyConfig())
    decision = runtime.decide_from_transcript(
        case.text,
        score,
        vad_active=False,
        agent_speaking=True,
        is_final=True,
    )

    verdicts = []
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        no_evidence_timeout_sec=0.8,
        min_speech_sec=0.25,
        on_terminal_verdict=verdicts.append,
    )
    owner.start_candidate(timeline=TurnTimeline(f"contract-{case.intent.value}"))
    owner.note_transcript(case.text, is_final=True)
    owner.note_turn_policy_decision(
        decision,
        transcript=case.text,
        vad_active=False,
        eot_score=score,
    )
    if decision.action is Action.CANCEL:
        owner.resolve(action="cancel", reason="contract_test")
    elif decision.action is Action.ROLLBACK:
        owner.resolve(action="rollback", reason="contract_test")

    actual_intent = decision.intent or InterruptIntent.UNCERTAIN
    actual_continue = verdicts[-1].continue_to_llm if verdicts else None
    assert (decision.action, actual_intent, actual_continue) == (
        case.action,
        case.intent,
        case.continue_to_llm,
    ), (
        f"text={case.text!r} score={score:.3f} reason={decision.reason!r} "
        f"verdict={verdicts[-1] if verdicts else None!r}"
    )


@_KNOWN_TERMINAL_HOLD_GAP
def test_final_transcript_hold_cannot_extend_duck_to_six_seconds() -> None:
    """A terminal transcript has no future revision worth six seconds of silence."""

    class _Clock:
        now = 10.0

        def __call__(self) -> float:
            return self.now

    clock = _Clock()
    owner = InterruptionOrchestrator(
        evidence_timeout_sec=6.0,
        no_evidence_timeout_sec=0.8,
        min_speech_sec=0.25,
        clock=clock,
    )
    owner.start_candidate(timeline=TurnTimeline("terminal-hold"))
    owner.note_transcript("已经识别完成", is_final=True)
    owner.note_turn_policy_decision(
        Decision(
            action=Action.HOLD,
            reason="semantic_score_wait",
            intent=InterruptIntent.UNCERTAIN,
        ),
        transcript="已经识别完成",
        vad_active=False,
        eot_score=0.3,
    )
    clock.now += 0.4

    assert owner.defer_false_resume_after_speech_end(
        transcript="已经识别完成",
        duck_suspended=True,
    )
    remaining = owner.hold_remaining_sec()

    assert remaining is not None
    assert remaining <= 0.9


@_KNOWN_SINGLE_OWNER_GAP
def test_duck_active_transcript_has_one_interruption_decision_owner() -> None:
    """The turn runtime owns ducked decisions; EOT must not run a second policy."""

    model = MagicMock()
    model.current_eot_score = 0.65
    model.hard_interrupt_score_threshold = 0.9
    runtime = MagicMock()
    runtime.decide_from_transcript.return_value = Decision(
        action=Action.HOLD,
        reason="awaiting_more_evidence",
        intent=InterruptIntent.UNCERTAIN,
    )
    apply_decision = MagicMock()
    handler = SemanticInterruptHandler(
        get_eot_model=lambda: model,
        turn_runtime=runtime,
        get_timeline=lambda: None,
        get_duck_active=lambda: True,
        get_duck_stats=lambda: DuckingStats(suspend_ms=10.0),
        get_vad_active=lambda: True,
        soft_interrupt_active=lambda: False,
        soft_interrupt_timeout=lambda: 0.5,
        apply_decision=apply_decision,
        interrupt_current_turn=MagicMock(),
        enter_soft_interrupt=MagicMock(),
    )

    handler.run("我还想问一个问题", is_final=False)

    model.should_interrupt.assert_not_called()
    runtime.decide_from_transcript.assert_called_once()
    apply_decision.assert_called_once()


@_KNOWN_ENDPOINTING_GAP
def test_agent_explicitly_binds_eot_endpointing_band(
    eot_model: ChineseModel,
) -> None:
    """Do not silently fall back to LiveKit's unrelated 0.5s/3.0s defaults."""

    pipeline = SimpleNamespace(
        _instructions="test",
        _factory=SimpleNamespace(
            stt=SimpleNamespace(stt=None),
            llm=SimpleNamespace(llm=None),
            tts=SimpleNamespace(tts=None),
            vad=None,
        ),
        _turn_detection=lambda: eot_model,
    )

    agent = build_full_duplex_agent(pipeline)
    expected_fast = eot_model.get_dynamic_silence_threshold(
        "",
        p_complete=1.0,
        is_final=False,
    )
    expected_deep = eot_model.get_dynamic_silence_threshold(
        "",
        p_complete=0.0,
        is_final=False,
    )

    assert agent.min_endpointing_delay == pytest.approx(expected_fast)
    assert agent.max_endpointing_delay == pytest.approx(expected_deep)
