"""G21 (2026-05-18): interrupted context primary source = TTS in-flight text.

The previous implementation (G2/G6) walked ``session.history`` for the
most-recent assistant message. Problem: ``session.history`` is only
updated AFTER a speech_handle winds down (livekit-agents 1.5
``_play_text_step``: ``add_message(role='assistant', ...)`` after
``audio_output.wait_for_playout()`` returns). When we snapshot the
context inside ``_duck_cancel_and_interrupt`` — which fires BEFORE the
speech handle finishes — history still contains the PREVIOUS turn's
assistant message, so the LLM hint would say "you just said: <wrong
old text>".

Production manifestation: 4-turn test log showed welcome message
captured as interrupted context for round-3 cancel, because the
in-progress round-3 reply hadn't reached history yet.

G21 fix: prefer ``tts_plugin.current_pushed_text`` (BailianTTS /
SenseTimeTTS both expose this via weakref to the active synth stream,
reading the framework's accumulator on ``SynthesizeStream._pushed_text``).
Fall back to history only when the TTS plugin doesn't have the property.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def _make_pipeline(
    *,
    history_messages: list,
    tts_pushed_text: str | None,
    played_sec: float | None = 1.0,
):
    """Build a stub StreamingPipeline configured for context-snapshot.

    Args:
        history_messages: what session.history.messages() returns
        tts_pushed_text: what tts_plugin.current_pushed_text returns
            (None → property absent → fallback path triggered)
    """
    from eidolon.livekit.agent.streaming import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)

    # session.history.messages() stub
    pipeline._session = MagicMock()
    pipeline._session.history.messages = MagicMock(return_value=history_messages)

    # tts factory stub: if tts_pushed_text is None, the plugin has no
    # current_pushed_text attribute (e.g. third-party TTS) → triggers fallback
    tts_plugin = MagicMock()
    if tts_pushed_text is None:
        del tts_plugin.current_pushed_text
    else:
        tts_plugin.current_pushed_text = tts_pushed_text
    pipeline._factory = MagicMock()
    pipeline._factory.tts.tts = tts_plugin

    # eot config
    cfg = SimpleNamespace(interrupted_context_enabled=True)
    eot = SimpleNamespace(_config=cfg)
    pipeline._get_eot_model = MagicMock(return_value=eot)

    # duck mixer played_seconds
    if played_sec is None:
        pipeline._duck_mixer = None
    else:
        mixer = MagicMock()
        type(mixer).played_seconds = property(lambda self: played_sec)
        pipeline._duck_mixer = mixer

    pipeline._last_interrupted_context = None

    return pipeline


# ---------------------------------------------------------------------------
# Primary path: in-flight TTS text wins over stale history
# ---------------------------------------------------------------------------


def test_prefers_tts_in_flight_over_history() -> None:
    """G21 core regression: when TTS reports in-flight text, use it even
    if history has older assistant messages."""
    pipeline = _make_pipeline(
        history_messages=[
            _msg("assistant", "previous turn reply"),  # stale
            _msg("user", "new question"),
        ],
        tts_pushed_text="正在合成中的当前回复",  # the right answer
    )

    pipeline._snapshot_interrupted_context()

    ctx = pipeline._last_interrupted_context
    assert ctx is not None
    assert ctx["text"] == "正在合成中的当前回复"
    assert ctx["source"] == "tts_in_flight"
    assert ctx["played_seconds"] == 1.0


def test_tts_in_flight_captures_chinese_correctly() -> None:
    """Smoke: long Chinese in-flight text is preserved verbatim
    (no truncation at the 80-char log cutoff)."""
    long_text = "你好，今天天气不错，适合出去散步，记得带伞。" * 4  # > 80 chars
    pipeline = _make_pipeline(
        history_messages=[],
        tts_pushed_text=long_text,
    )

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context["text"] == long_text


# ---------------------------------------------------------------------------
# Fallback path: empty in-flight → history wins
# ---------------------------------------------------------------------------


def test_falls_back_to_history_when_tts_empty() -> None:
    """No in-flight text (synth not started yet, or already drained) →
    fall back to session.history."""
    pipeline = _make_pipeline(
        history_messages=[
            _msg("user", "hello"),
            _msg("assistant", "history fallback reply"),
        ],
        tts_pushed_text="",  # empty → fallback
    )

    pipeline._snapshot_interrupted_context()

    ctx = pipeline._last_interrupted_context
    assert ctx is not None
    assert ctx["text"] == "history fallback reply"
    assert ctx["source"] == "session_history_fallback"


def test_falls_back_to_history_when_tts_whitespace_only() -> None:
    """Whitespace-only in-flight text counts as 'no real text' → fallback."""
    pipeline = _make_pipeline(
        history_messages=[
            _msg("assistant", "history reply"),
        ],
        tts_pushed_text="   \n\t  ",
    )

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context["text"] == "history reply"
    assert pipeline._last_interrupted_context["source"] == "session_history_fallback"


def test_falls_back_when_tts_plugin_lacks_property() -> None:
    """3rd-party TTS plugin without current_pushed_text → use history."""
    pipeline = _make_pipeline(
        history_messages=[
            _msg("assistant", "from history"),
        ],
        tts_pushed_text=None,  # property absent
    )

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context["text"] == "from history"
    assert pipeline._last_interrupted_context["source"] == "session_history_fallback"


# ---------------------------------------------------------------------------
# Both empty → no snapshot
# ---------------------------------------------------------------------------


def test_no_snapshot_when_both_sources_empty() -> None:
    """No in-flight + no assistant in history → no context captured."""
    pipeline = _make_pipeline(
        history_messages=[
            _msg("user", "only user msgs"),
        ],
        tts_pushed_text="",
    )

    pipeline._snapshot_interrupted_context()

    assert pipeline._last_interrupted_context is None


# ---------------------------------------------------------------------------
# played_seconds preserved on both paths
# ---------------------------------------------------------------------------


def test_played_seconds_recorded_on_primary_path() -> None:
    pipeline = _make_pipeline(
        history_messages=[],
        tts_pushed_text="in-flight",
        played_sec=2.5,
    )
    pipeline._snapshot_interrupted_context()
    assert pipeline._last_interrupted_context["played_seconds"] == 2.5


def test_played_seconds_recorded_on_fallback_path() -> None:
    pipeline = _make_pipeline(
        history_messages=[_msg("assistant", "h")],
        tts_pushed_text="",
        played_sec=0.7,
    )
    pipeline._snapshot_interrupted_context()
    assert pipeline._last_interrupted_context["played_seconds"] == 0.7


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
