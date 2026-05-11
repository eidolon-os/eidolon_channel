"""Shared types for the voice agent pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable


class PipelineState(Enum):
    """Current state of the pipeline."""

    IDLE = auto()
    PROCESSING_AUDIO = auto()  #: STT is transcribing
    GENERATING = auto()         #: LLM is generating response
    SPEAKING = auto()           #: TTS is synthesizing audio


# ---------------------------------------------------------------------------
# Pipeline callbacks (Null Object pattern — every field defaults to no-op)
# ---------------------------------------------------------------------------


def _noop() -> None: ...
def _noop_str(_t: str) -> None: ...
def _noop_exc(_e: Exception) -> None: ...


@dataclass
class PipelineCallbacks:
    """Callbacks invoked by the pipeline to report events to the client.

    All fields default to no-op functions, so callers can always invoke
    them directly without ``None`` guards. To handle a specific event,
    pass a callable when constructing :class:`PipelineCallbacks`.

    Example::

        cbs = PipelineCallbacks(
            on_user_message=lambda text: print(f"user said: {text}"),
        )
        # Other callbacks are no-ops; safe to call:
        cbs.on_agent_started_speaking()
    """

    on_user_started_speaking: Callable[[], None] = _noop
    on_user_ended_speaking: Callable[[], None] = _noop
    on_user_message: Callable[[str], None] = _noop_str
    on_agent_started_speaking: Callable[[], None] = _noop
    on_agent_message: Callable[[str], None] = _noop_str
    on_agent_ended_speaking: Callable[[], None] = _noop
    on_agent_response_done: Callable[[], None] = _noop
    on_error: Callable[[Exception], None] = _noop_exc
    on_duck_started: Callable[[], None] = _noop
    on_duck_resolved: Callable[[str], None] = _noop_str


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def generate_turn_id() -> str:
    """Generate a unique ID for a new user turn."""
    return uuid.uuid4().hex[:16]
