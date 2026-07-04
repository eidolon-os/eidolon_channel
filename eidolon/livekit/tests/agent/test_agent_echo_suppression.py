"""Content-based echo suppression (②).

Energy can't tell residual-echo spikes from real near-end speech (measured: echo
residual p99 ~1166 RMS > real speech p50 ~67 on the cleaned device output). So
during agent playback we reject a transcript that is the agent's OWN current
speech echoed back — matched by CONTENT against what the agent is saying, not by
energy. The agent's in-flight text is available via the TTS plugin's
``current_pushed_text``.

These guard the pure matcher; the call site (``_on_user_transcribed``) gates it on
agent-playback and drops matched transcripts before they start a user turn.
"""

from __future__ import annotations

from eidolon.livekit.agent.session.transcript_echo import TranscriptEchoGate


def _gate(agent_text: str) -> TranscriptEchoGate:
    # Stub the agent-text source (live impl reads TTS current_pushed_text).
    return TranscriptEchoGate(get_agent_text=lambda: agent_text)


# Real fragments observed on-device: the agent said
# "听起来你好像有两层意思——一个是想让我…你先说前面那个：你希望我叫你什么？"
# and STT transcribed the echo as "一个是。" / "说前面那个。" → must be dropped.
AGENT = "听起来你好像有两层意思——一个是想让我帮你，你先说前面那个：你希望我叫你什么？"


def test_echo_fragment_substring_is_detected() -> None:
    gate = _gate(AGENT)
    assert gate.is_echo("说前面那个") is True
    assert gate.is_echo("一个是") is True


def test_echo_ignores_punctuation_and_spaces() -> None:
    gate = _gate(AGENT)
    assert gate.is_echo("说前面那个。") is True
    assert gate.is_echo(" 你希望我叫你什么 ") is True


def test_real_user_turn_not_flagged_as_echo() -> None:
    gate = _gate(AGENT)
    # Substantive user input the agent did not say → not echo.
    assert gate.is_echo("查一下明天的天气") is False
    assert gate.is_echo("帮我换个话题") is False


def test_one_character_backchannel_is_not_echo() -> None:
    gate = _gate("你好！我是你的 AI 助手，请问有什么可以帮你的？")
    assert gate.is_echo("好") is False


def test_echo_min_chars_is_configurable() -> None:
    gate = TranscriptEchoGate(
        get_agent_text=lambda: "你好！我是你的 AI 助手，请问有什么可以帮你的？",
        min_normalized_chars=1,
    )
    assert gate.is_echo("好") is True


def test_no_agent_text_means_not_echo() -> None:
    # Agent not speaking (no in-flight TTS text) → nothing to echo.
    gate = _gate("")
    assert gate.is_echo("说前面那个") is False


def test_empty_transcript_is_not_echo() -> None:
    gate = _gate(AGENT)
    assert gate.is_echo("") is False
    assert gate.is_echo("   ") is False


def test_echo_gate_can_use_welcome_text_from_assistant_ledger() -> None:
    from eidolon.livekit.agent.session.assistant_speech import AssistantSpeechLedger

    ledger = AssistantSpeechLedger()
    ledger.record("你好！我是你的 AI 助手，请问有什么可以帮你的？", source="welcome")
    gate = TranscriptEchoGate(
        get_agent_text=lambda: ledger.current_or_recent_text(max_age_ms=3000)
    )

    assert gate.is_echo("我是你的 AI 助手") is True


def test_assistant_ledger_does_not_return_stale_text() -> None:
    from eidolon.livekit.agent.session.assistant_speech import AssistantSpeechLedger

    now = 100.0
    ledger = AssistantSpeechLedger(clock=lambda: now)
    ledger.record("你好！我是你的 AI 助手，请问有什么可以帮你的？", source="welcome")
    now = 104.0
    gate = TranscriptEchoGate(
        get_agent_text=lambda: ledger.current_or_recent_text(max_age_ms=3000)
    )

    assert gate.is_echo("我是你的 AI 助手") is False
