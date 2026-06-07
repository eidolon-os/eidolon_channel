"""InterruptedContextManager tests."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon.livekit.agent.context import InterruptedContextManager


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, text_content=text)


def test_interrupted_context_manager_prefers_tts_in_flight_text() -> None:
    manager = InterruptedContextManager()
    session = MagicMock()
    session.history.messages = MagicMock(
        return_value=[_msg("assistant", "history reply")]
    )
    factory = SimpleNamespace(
        tts=SimpleNamespace(tts=SimpleNamespace(current_pushed_text="tts reply"))
    )
    duck_mixer = SimpleNamespace(played_seconds=1.2)
    cfg = SimpleNamespace(interrupted_context_enabled=True)

    manager.snapshot(
        session=session,
        factory=factory,
        duck_mixer=duck_mixer,
        config=cfg,
    )

    assert manager.last_context is not None
    assert manager.last_context["text"] == "tts reply"
    assert manager.last_context["source"] == "tts_in_flight"
    assert manager.last_context["played_seconds"] == 1.2


def test_interrupted_context_manager_injects_and_clears_hint() -> None:
    manager = InterruptedContextManager()
    manager.last_context = {
        "text": "刚才的回答",
        "timestamp": time.monotonic(),
        "played_seconds": 0.0,
        "source": "tts_in_flight",
    }
    session = MagicMock()
    cfg = SimpleNamespace(interrupted_context_max_age_sec=999999.0)

    manager.inject(session=session, config=cfg)

    session.history.insert.assert_called_once()
    hint = session.history.insert.call_args.args[0]
    assert hint.role == "system"
    assert "刚才的回答" in hint.content[0]
    assert "用户几乎没听到任何内容" in hint.content[0]
    assert manager.last_context is None
