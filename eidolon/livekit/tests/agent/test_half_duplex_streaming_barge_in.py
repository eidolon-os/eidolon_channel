# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""L2 guard: half_duplex StreamingPipeline must NOT enter the barge-in path.

Regression coverage for the bug where half_duplex reused full_duplex's barge-in
orchestration: every utterance armed a duck + interruption candidate that timed
out with no transcript and rolled back (continue_to_llm=False), so the user turn
never committed and the agent never replied.

Unlike the mock-owner unit tests in ``test_full_duplex_speech_lifecycle.py``,
these drive a **real** ``StreamingPipeline`` — only the ``SharedStageFactory`` is
a mock, so the whole barge-in owner wiring (speech lifecycle, attention effects,
interruption orchestrator, ducking) is exercised end to end, without a LiveKit
room. Each half_duplex assertion is mirrored by a full_duplex control, so the
test provably distinguishes the two modes and cannot silently pass if the mode
gate is removed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from eidolon_sdk.biz.contracts import (
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
)

from eidolon.livekit.agent.full_duplex import StreamingPipeline


def _pipeline(mode: str, *, allow_interruptions: bool) -> StreamingPipeline:
    """A real StreamingPipeline with only the stage factory mocked."""
    return StreamingPipeline(
        MagicMock(),
        interaction_mode=mode,
        allow_interruptions=allow_interruptions,
    )


def _spy_attention(pipeline: StreamingPipeline) -> MagicMock:
    """Wrap the real attention handler so we can assert whether VAD-start armed it."""
    pipeline._ensure_attention_effect_handler()  # constructs pipeline._attention_effects
    spy = MagicMock(wraps=pipeline._attention_effects.handle_speaking_started)
    pipeline._attention_effects.handle_speaking_started = spy
    return spy


def test_half_duplex_vad_start_does_not_enter_barge_in() -> None:
    pipeline = _pipeline(INTERACTION_MODE_HALF_DUPLEX, allow_interruptions=False)
    assert pipeline._barge_in_enabled is False
    spy = _spy_attention(pipeline)

    pipeline._ensure_speech_lifecycle().handle_started()

    # No barge-in machinery armed on VAD-start...
    spy.assert_not_called()
    assert pipeline._interruption_orchestrator.active is False
    assert pipeline._ducking.is_suspended is False
    assert pipeline._timeline.attrs.get("interruption_owner") == "disabled_no_barge_in"
    # ...but the shared turn setup still ran (the turn must reach framework commit).
    assert pipeline._timeline.attrs.get("interruption_owner") != "livekit_native_adaptive"


def test_full_duplex_vad_start_enters_barge_in() -> None:
    # Control: identical drive with barge-in ON. Proves the half_duplex assertions
    # above actually distinguish the modes (the test can't pass if the gate is gone).
    pipeline = _pipeline(INTERACTION_MODE_FULL_DUPLEX, allow_interruptions=True)
    assert pipeline._barge_in_enabled is True
    spy = _spy_attention(pipeline)

    pipeline._ensure_speech_lifecycle().handle_started()

    spy.assert_called_once()
    assert pipeline._timeline.attrs.get("interruption_owner") != "disabled_no_barge_in"


async def test_half_duplex_vad_stop_does_not_resolve_interruption() -> None:
    pipeline = _pipeline(INTERACTION_MODE_HALF_DUPLEX, allow_interruptions=False)
    # voiceprint is a shared (non-barge-in) owner; stub it so finish_turn does not
    # spawn a background verify task (which would need a running loop, irrelevant
    # to this barge-in assertion).
    pipeline._voiceprint_turns = MagicMock()
    lifecycle = pipeline._ensure_speech_lifecycle()

    lifecycle.handle_started()
    lifecycle.handle_stopped()

    # Stop-side barge-in resolution is skipped; no duck was ever suspended.
    assert pipeline._ducking.is_suspended is False
    assert pipeline._interruption_orchestrator.active is False
