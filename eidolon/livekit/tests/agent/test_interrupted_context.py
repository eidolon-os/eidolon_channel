"""Unit tests for G2: `_snapshot_interrupted_context` API correctness.

The framework's `ChatContext.messages` is a METHOD, not a property. Pre-G2
code accessed `.messages` without calling it, then tried to `reversed()` a
method object → TypeError at runtime. The bug was latent because the only
caller is the EOT-cancel path, which never fired in the prior 0.55 VAD-gate
regime.

Regression assertions:
  * `_snapshot_interrupted_context` survives normal use (no TypeError).
  * It picks the last assistant message with non-empty text.
  * It tolerates a session.history that contains only user messages.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from eidolon.livekit.agent.full_duplex import StreamingPipeline


def _interrupted_context(pipeline: "StreamingPipeline") -> dict | None:
    return pipeline._interrupted_context.last_context


def _make_pipeline_with_history(messages: list) -> "StreamingPipeline":
    """Build a minimally-initialised StreamingPipeline whose
    ``_session.history.messages()`` returns the supplied list."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    # Bypass full __init__ — _snapshot_interrupted_context only reads
    # ``self._session`` and ``self._get_eot_model()._config``.
    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.history.messages = MagicMock(return_value=messages)
    # G6 (2026-05-17): _snapshot_interrupted_context now consults
    # self._duck_mixer.played_seconds when available; set None for these
    # legacy tests that pre-date G6.
    pipeline._duck_mixer = None

    # Stub _get_eot_model() → cfg with history fallback enabled for these
    # legacy method-call regression tests.
    cfg = SimpleNamespace(
        interrupted_context_enabled=True,
        interrupted_context_history_fallback_enabled=True,
    )
    eot = SimpleNamespace(_config=cfg)
    pipeline._get_eot_model = MagicMock(return_value=eot)

    return pipeline


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def test_picks_last_assistant_message() -> None:
    """G2 happy path: walking history in reverse finds the most-recent
    assistant turn with non-empty text."""
    pipeline = _make_pipeline_with_history([
        _msg("user", "hi"),
        _msg("assistant", "old reply"),
        _msg("user", "follow-up"),
        _msg("assistant", "newer reply"),
    ])

    pipeline._snapshot_interrupted_context()

    assert _interrupted_context(pipeline) is not None
    assert _interrupted_context(pipeline)["text"] == "newer reply"


def test_method_call_not_attribute_access() -> None:
    """G2 regression: ``messages`` must be called as a method, not a property.

    Asserts that .messages() is the call site (any future regression that
    reverts to attribute access will fail this — MagicMock's method tracker
    would not record a call)."""
    pipeline = _make_pipeline_with_history([
        _msg("assistant", "hello"),
    ])

    pipeline._snapshot_interrupted_context()

    pipeline._session.history.messages.assert_called_once_with()


def test_skips_empty_text_messages() -> None:
    """Empty text_content should be skipped — we want the last assistant
    turn that actually said something."""
    pipeline = _make_pipeline_with_history([
        _msg("assistant", "non-empty"),
        _msg("assistant", ""),
    ])

    pipeline._snapshot_interrupted_context()

    assert _interrupted_context(pipeline) is not None
    assert _interrupted_context(pipeline)["text"] == "non-empty"


def test_no_assistant_messages_leaves_context_none() -> None:
    """Pure-user history → nothing to snapshot, interrupted context stays None."""
    pipeline = _make_pipeline_with_history([
        _msg("user", "first"),
        _msg("user", "second"),
    ])

    pipeline._snapshot_interrupted_context()

    assert _interrupted_context(pipeline) is None


def test_disabled_config_is_noop() -> None:
    """If ``interrupted_context_enabled=False`` the function returns
    immediately without touching session.history."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    cfg = SimpleNamespace(interrupted_context_enabled=False)
    pipeline._get_eot_model = MagicMock(return_value=SimpleNamespace(_config=cfg))

    pipeline._snapshot_interrupted_context()

    pipeline._session.history.messages.assert_not_called()
    assert _interrupted_context(pipeline) is None


# ---------------------------------------------------------------------------
# G6 (2026-05-17): played_seconds enrichment
# ---------------------------------------------------------------------------


def _make_pipeline_with_duck_mixer(
    *, played_seconds: float | None
) -> "StreamingPipeline":
    """Pipeline with a stub duck mixer reporting ``played_seconds``."""
    from eidolon.livekit.agent.full_duplex import StreamingPipeline

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._session = MagicMock()
    pipeline._session.history.messages = MagicMock(
        return_value=[_msg("assistant", "你好世界，今天天气不错")]
    )
    pipeline._factory = SimpleNamespace(
        tts=SimpleNamespace(
            tts=SimpleNamespace(current_pushed_text="你好世界，今天天气不错")
        )
    )

    cfg = SimpleNamespace(interrupted_context_enabled=True)
    pipeline._get_eot_model = MagicMock(
        return_value=SimpleNamespace(_config=cfg)
    )

    if played_seconds is None:
        pipeline._duck_mixer = None
    else:
        mixer = MagicMock()
        type(mixer).played_seconds = property(lambda self: played_seconds)
        pipeline._duck_mixer = mixer

    return pipeline


def test_snapshot_records_played_seconds_when_duck_mixer_present() -> None:
    """G6: when DuckingMixer is available, played_seconds gets captured."""
    pipeline = _make_pipeline_with_duck_mixer(played_seconds=1.5)
    pipeline._snapshot_interrupted_context()
    ctx = _interrupted_context(pipeline)
    assert ctx is not None
    assert ctx["played_seconds"] == 1.5
    assert ctx["text"] == "你好世界，今天天气不错"


def test_snapshot_records_context_preview_on_timeline() -> None:
    from eidolon.livekit.agent.observability import TurnTimeline

    pipeline = _make_pipeline_with_duck_mixer(played_seconds=1.5)
    pipeline._timeline = TurnTimeline("turn-interrupted")

    pipeline._snapshot_interrupted_context()

    attr = pipeline._timeline.attrs["interrupted_context"]
    assert attr["source"] == "tts_in_flight"
    assert attr["played_seconds"] == 1.5
    assert attr["text_preview"] == "你好世界，今天天气不错"


def test_snapshot_records_none_played_seconds_when_no_duck_mixer() -> None:
    """G6: without DuckingMixer (e.g. tests / non-streaming pipelines),
    played_seconds is None, not crash."""
    pipeline = _make_pipeline_with_duck_mixer(played_seconds=None)
    pipeline._snapshot_interrupted_context()
    ctx = _interrupted_context(pipeline)
    assert ctx is not None
    assert ctx["played_seconds"] is None
    assert ctx["text"] == "你好世界，今天天气不错"


def test_snapshot_played_seconds_zero_is_preserved() -> None:
    """G6: 0.0 played-seconds (cancel fired before any audio) is meaningful
    and must not be collapsed to None — the LLM hint differentiates."""
    pipeline = _make_pipeline_with_duck_mixer(played_seconds=0.0)
    pipeline._snapshot_interrupted_context()
    ctx = _interrupted_context(pipeline)
    assert ctx is not None
    assert ctx["played_seconds"] == 0.0
