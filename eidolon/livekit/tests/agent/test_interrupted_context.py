"""Unit tests for interrupted context ledger snapshot correctness.

The framework's `ChatContext.messages` is a METHOD, not a property. Pre-G2
code accessed `.messages` without calling it, then tried to `reversed()` a
method object → TypeError at runtime. The bug was latent because the only
caller is the EOT-cancel path, which never fired in the prior 0.55 VAD-gate
regime.

Regression assertions:
  * `FullDuplexContextLedger.snapshot()` survives normal use (no TypeError).
  * It picks the last assistant message with non-empty text.
  * It tolerates a session.history that contains only user messages.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.full_duplex.context_ledger import FullDuplexContextLedger
from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.output.ducking import OutputDuckingController


def _interrupted_context(ledger: FullDuplexContextLedger) -> dict | None:
    return ledger.last_context


def _make_ledger_with_history(messages: list) -> SimpleNamespace:
    """Build a ledger whose ``session.history.messages()`` returns messages."""

    session = MagicMock()
    session.history.messages = MagicMock(return_value=messages)
    ducking = OutputDuckingController()
    ducking.mixer = None
    cfg = SimpleNamespace(
        interrupted_context_enabled=True,
        interrupted_context_history_fallback_enabled=True,
    )
    ledger = FullDuplexContextLedger(
        get_session=lambda: session,
        get_factory=lambda: None,
        get_duck_mixer=lambda: ducking.mixer,
        get_config=lambda: cfg,
        get_timeline=lambda: None,
    )
    return SimpleNamespace(ledger=ledger, session=session)


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def test_picks_last_assistant_message() -> None:
    """G2 happy path: walking history in reverse finds the most-recent
    assistant turn with non-empty text."""
    runtime = _make_ledger_with_history(
        [
            _msg("user", "hi"),
            _msg("assistant", "old reply"),
            _msg("user", "follow-up"),
            _msg("assistant", "newer reply"),
        ]
    )

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger) is not None
    assert _interrupted_context(runtime.ledger)["text"] == "newer reply"


def test_method_call_not_attribute_access() -> None:
    """G2 regression: ``messages`` must be called as a method, not a property.

    Asserts that .messages() is the call site (any future regression that
    reverts to attribute access will fail this — MagicMock's method tracker
    would not record a call)."""
    runtime = _make_ledger_with_history([
        _msg("assistant", "hello"),
    ])

    runtime.ledger.snapshot()

    runtime.session.history.messages.assert_called_once_with()


def test_skips_empty_text_messages() -> None:
    """Empty text_content should be skipped — we want the last assistant
    turn that actually said something."""
    runtime = _make_ledger_with_history([
        _msg("assistant", "non-empty"),
        _msg("assistant", ""),
    ])

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger) is not None
    assert _interrupted_context(runtime.ledger)["text"] == "non-empty"


def test_no_assistant_messages_leaves_context_none() -> None:
    """Pure-user history → nothing to snapshot, interrupted context stays None."""
    runtime = _make_ledger_with_history([
        _msg("user", "first"),
        _msg("user", "second"),
    ])

    runtime.ledger.snapshot()

    assert _interrupted_context(runtime.ledger) is None


def test_disabled_config_is_noop() -> None:
    """If ``interrupted_context_enabled=False`` the function returns
    immediately without touching session.history."""
    session = MagicMock()
    cfg = SimpleNamespace(interrupted_context_enabled=False)
    ledger = FullDuplexContextLedger(
        get_session=lambda: session,
        get_factory=lambda: None,
        get_duck_mixer=lambda: None,
        get_config=lambda: cfg,
        get_timeline=lambda: None,
    )

    ledger.snapshot()

    session.history.messages.assert_not_called()
    assert _interrupted_context(ledger) is None


# ---------------------------------------------------------------------------
# G6 (2026-05-17): played_seconds enrichment
# ---------------------------------------------------------------------------


def _make_ledger_with_duck_mixer(
    *, played_seconds: float | None, timeline: TurnTimeline | None = None
) -> SimpleNamespace:
    """Ledger with a stub duck mixer reporting ``played_seconds``."""

    session = MagicMock()
    session.history.messages = MagicMock(
        return_value=[_msg("assistant", "你好世界，今天天气不错")]
    )
    factory = SimpleNamespace(
        tts=SimpleNamespace(
            tts=SimpleNamespace(current_pushed_text="你好世界，今天天气不错")
        )
    )

    cfg = SimpleNamespace(interrupted_context_enabled=True)

    ducking = OutputDuckingController()
    if played_seconds is None:
        ducking.mixer = None
    else:
        mixer = MagicMock()
        type(mixer).played_seconds = property(lambda self: played_seconds)
        ducking.mixer = mixer

    ledger = FullDuplexContextLedger(
        get_session=lambda: session,
        get_factory=lambda: factory,
        get_duck_mixer=lambda: ducking.mixer,
        get_config=lambda: cfg,
        get_timeline=lambda: timeline,
    )
    return SimpleNamespace(ledger=ledger, timeline=timeline)


def test_snapshot_records_played_seconds_when_duck_mixer_present() -> None:
    """G6: when DuckingMixer is available, played_seconds gets captured."""
    runtime = _make_ledger_with_duck_mixer(played_seconds=1.5)
    runtime.ledger.snapshot()
    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["played_seconds"] == 1.5
    assert ctx["text"] == "你好世界，今天天气不错"


def test_snapshot_records_context_preview_on_timeline() -> None:
    timeline = TurnTimeline("turn-interrupted")
    runtime = _make_ledger_with_duck_mixer(played_seconds=1.5, timeline=timeline)

    runtime.ledger.snapshot()

    attr = timeline.attrs["interrupted_context"]
    assert attr["source"] == "tts_in_flight"
    assert attr["played_seconds"] == 1.5
    assert attr["text_preview"] == "你好世界，今天天气不错"


def test_snapshot_records_none_played_seconds_when_no_duck_mixer() -> None:
    """G6: without DuckingMixer (e.g. tests / non-streaming pipelines),
    played_seconds is None, not crash."""
    runtime = _make_ledger_with_duck_mixer(played_seconds=None)
    runtime.ledger.snapshot()
    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["played_seconds"] is None
    assert ctx["text"] == "你好世界，今天天气不错"


def test_snapshot_played_seconds_zero_is_preserved() -> None:
    """G6: 0.0 played-seconds (cancel fired before any audio) is meaningful
    and must not be collapsed to None — the LLM hint differentiates."""
    runtime = _make_ledger_with_duck_mixer(played_seconds=0.0)
    runtime.ledger.snapshot()
    ctx = _interrupted_context(runtime.ledger)
    assert ctx is not None
    assert ctx["played_seconds"] == 0.0
