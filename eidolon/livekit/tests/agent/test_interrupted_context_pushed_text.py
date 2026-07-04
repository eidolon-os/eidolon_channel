"""G21 (2026-05-18): interrupted context primary source = TTS in-flight text.

The previous implementation (G2/G6) walked ``session.history`` for the
most-recent assistant message. Problem: ``session.history`` is only
updated AFTER a speech_handle winds down (livekit-agents 1.5
``_play_text_step``: ``add_message(role='assistant', ...)`` after
``audio_output.wait_for_playout()`` returns). When we snapshot the
context inside ``FullDuplexInterruptionEffects.cancel_and_interrupt`` — which fires BEFORE the
speech handle finishes — history still contains the PREVIOUS turn's
assistant message, so the LLM hint would say "you just said: <wrong
old text>".

Production manifestation: 4-turn test log showed welcome message
captured as interrupted context for round-3 cancel, because the
in-progress round-3 reply hadn't reached history yet.

G21 fix: prefer ``tts_plugin.current_pushed_text`` (BailianTTS /
SenseTimeTTS both expose this via weakref to the active synth stream,
reading the framework's accumulator on ``SynthesizeStream._pushed_text``).
History fallback is opt-in only because real-room logs showed it can capture
the previous assistant turn when the active speech handle has not reached
history yet.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.context_ledger import FullDuplexContextLedger
from eidolon.livekit.agent.output.ducking import OutputDuckingController


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def _interrupted_context(ledger: FullDuplexContextLedger) -> dict | None:
    return ledger.last_context


def _make_ledger(
    *,
    history_messages: list,
    tts_pushed_text: str | None,
    played_sec: float | None = 1.0,
    history_fallback_enabled: bool = False,
    assistant_text: str = "",
):
    """Build a full-duplex context ledger configured for snapshot tests.

    Args:
        history_messages: what session.history.messages() returns
        tts_pushed_text: what tts_plugin.current_pushed_text returns
            (None → property absent → fallback path triggered)
    """
    session = MagicMock()
    session.history.messages = MagicMock(return_value=history_messages)

    # tts factory stub: if tts_pushed_text is None, the plugin has no
    # current_pushed_text attribute (e.g. third-party TTS) → triggers fallback
    tts_plugin = MagicMock()
    if tts_pushed_text is None:
        del tts_plugin.current_pushed_text
    else:
        tts_plugin.current_pushed_text = tts_pushed_text
    factory = MagicMock()
    factory.tts.tts = tts_plugin

    cfg = SimpleNamespace(
        interrupted_context_enabled=True,
        interrupted_context_history_fallback_enabled=history_fallback_enabled,
    )

    ducking = OutputDuckingController()
    if played_sec is None:
        ducking.mixer = None
    else:
        mixer = MagicMock()
        type(mixer).played_seconds = property(lambda self: played_sec)
        ducking.mixer = mixer

    ledger = FullDuplexContextLedger(
        get_session=lambda: session,
        get_factory=lambda: factory,
        get_duck_mixer=lambda: ducking.mixer,
        get_config=lambda: cfg,
        get_timeline=lambda: None,
        get_assistant_text=lambda: assistant_text,
    )
    return SimpleNamespace(ledger=ledger, session=session)


# ---------------------------------------------------------------------------
# Primary path: in-flight TTS text wins over stale history
# ---------------------------------------------------------------------------


def test_prefers_tts_in_flight_over_history() -> None:
    """G21 core regression: when TTS reports in-flight text, use it even
    if history has older assistant messages."""
    runtime = _make_ledger(
        history_messages=[
            _msg("assistant", "previous turn reply"),  # stale
            _msg("user", "new question"),
        ],
        tts_pushed_text="正在合成中的当前回复",  # the right answer
    )

    runtime.ledger.snapshot()

    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["text"] == "正在合成中的当前回复"
    assert ctx["source"] == "tts_in_flight"
    assert ctx["played_seconds"] == 1.0


def test_tts_in_flight_captures_chinese_correctly() -> None:
    """Smoke: long Chinese in-flight text is preserved verbatim
    (no truncation at the 80-char log cutoff)."""
    long_text = "你好，今天天气不错，适合出去散步，记得带伞。" * 4  # > 80 chars
    runtime = _make_ledger(
        history_messages=[],
        tts_pushed_text=long_text,
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger)["text"] == long_text


# ---------------------------------------------------------------------------
# Fallback path: empty in-flight → history only when explicitly enabled
# ---------------------------------------------------------------------------


def test_skips_history_fallback_by_default_when_tts_empty() -> None:
    """No in-flight text should not capture stale session history by default."""
    runtime = _make_ledger(
        history_messages=[
            _msg("user", "hello"),
            _msg("assistant", "history fallback reply"),
        ],
        tts_pushed_text="",  # empty → fallback
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger) is None
    runtime.session.history.messages.assert_not_called()


def test_context_ledger_uses_assistant_speech_ledger_before_history() -> None:
    runtime = _make_ledger(
        history_messages=[_msg("assistant", "stale history reply")],
        tts_pushed_text="",
        assistant_text="你好！我是你的 AI 助手，请问有什么可以帮你的？",
    )

    runtime.ledger.snapshot()

    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["text"] == "你好！我是你的 AI 助手，请问有什么可以帮你的？"
    assert ctx["source"] == "assistant_speech_ledger"
    runtime.session.history.messages.assert_not_called()


def test_falls_back_to_history_when_tts_empty_and_enabled() -> None:
    """History fallback remains available for controlled integrations."""
    runtime = _make_ledger(
        history_messages=[
            _msg("user", "hello"),
            _msg("assistant", "history fallback reply"),
        ],
        tts_pushed_text="",
        history_fallback_enabled=True,
    )

    runtime.ledger.snapshot()

    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["text"] == "history fallback reply"
    assert ctx["source"] == "session_history_fallback"


def test_falls_back_to_history_when_tts_whitespace_only() -> None:
    """Whitespace-only in-flight text counts as 'no real text' → fallback."""
    runtime = _make_ledger(
        history_messages=[
            _msg("assistant", "history reply"),
        ],
        tts_pushed_text="   \n\t  ",
        history_fallback_enabled=True,
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger)["text"] == "history reply"
    assert _interrupted_context(runtime.ledger)["source"] == "session_history_fallback"


def test_falls_back_when_tts_plugin_lacks_property() -> None:
    """3rd-party TTS plugin without current_pushed_text → use history."""
    runtime = _make_ledger(
        history_messages=[
            _msg("assistant", "from history"),
        ],
        tts_pushed_text=None,  # property absent
        history_fallback_enabled=True,
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger)["text"] == "from history"
    assert _interrupted_context(runtime.ledger)["source"] == "session_history_fallback"


# ---------------------------------------------------------------------------
# Both empty → no snapshot
# ---------------------------------------------------------------------------


def test_no_snapshot_when_both_sources_empty() -> None:
    """No in-flight + no assistant in history → no context captured."""
    runtime = _make_ledger(
        history_messages=[
            _msg("user", "only user msgs"),
        ],
        tts_pushed_text="",
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger) is None


# ---------------------------------------------------------------------------
# played_seconds preserved on both paths
# ---------------------------------------------------------------------------


def test_played_seconds_recorded_on_primary_path() -> None:
    runtime = _make_ledger(
        history_messages=[],
        tts_pushed_text="in-flight",
        played_sec=2.5,
    )
    runtime.ledger.snapshot()
    assert _interrupted_context(runtime.ledger)["played_seconds"] == 2.5


def test_played_seconds_recorded_on_fallback_path() -> None:
    runtime = _make_ledger(
        history_messages=[_msg("assistant", "h")],
        tts_pushed_text="",
        played_sec=0.7,
        history_fallback_enabled=True,
    )
    runtime.ledger.snapshot()
    assert _interrupted_context(runtime.ledger)["played_seconds"] == 0.7


# ---------------------------------------------------------------------------
# Plugin-side: BailianTTS / SenseTimeTTS expose the property correctly
# ---------------------------------------------------------------------------


def test_bailian_tts_current_pushed_text_empty_when_no_stream() -> None:
    """A fresh BailianTTS instance with no active stream returns empty string,
    not None or AttributeError."""
    from eidolon.livekit.plugins.tts.bailian.tts import BailianTTS
    from eidolon.livekit.plugins.tts.bailian.config import BailianTTSConfig

    cfg = BailianTTSConfig(
        api_url="wss://test.example.com",
        api_key="test-key",
        model="cosyvoice-test",
        voice="test-voice",
    )
    tts = BailianTTS(config=cfg)
    assert tts.current_pushed_text == ""


def test_sensetime_tts_current_pushed_text_empty_when_no_stream() -> None:
    """A fresh SenseTimeTTS instance with no active stream returns empty."""
    from eidolon.livekit.plugins.tts.sensetime.tts import SenseTimeTTS
    from eidolon.livekit.plugins.tts.sensetime.config import SenseTimeTTSConfig

    cfg = SenseTimeTTSConfig(
        api_url="wss://test.example.com",
        api_key="test-key",
        model="senseaudio-test",
        voice="test-voice",
    )
    tts = SenseTimeTTS(config=cfg)
    assert tts.current_pushed_text == ""
