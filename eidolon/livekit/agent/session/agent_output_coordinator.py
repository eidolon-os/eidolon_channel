"""Coordinate observable agent-output progress for one user turn.

This module is intentionally side-effect-light.  It does not control LiveKit or
audio playback; it normalizes framework/provider events into one timeline field
so silent-output failures are visible without grepping three logs.
"""

from __future__ import annotations

from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline


class AgentOutputCoordinator:
    """Track brain/TTS/audio progress and own the active response timeline.

    A full-duplex session can have two turns alive at once: the committed turn
    whose response is still playing, and a new user candidate attempting to
    interrupt it.  Provider and playback events must remain attached to the
    former until that response reaches a terminal outcome.  Keeping that
    ownership here avoids coupling output identity to the mutable speech
    candidate pointer.
    """

    def __init__(self) -> None:
        self._active_timeline: TurnTimeline | None = None

    @property
    def active_timeline(self) -> TurnTimeline | None:
        return self._active_timeline

    def claim(self, timeline: TurnTimeline) -> TurnTimeline | None:
        """Bind output events to ``timeline`` and return any displaced owner."""

        previous = self._active_timeline
        self._active_timeline = timeline
        return previous if previous is not timeline else None

    def release(self, timeline: TurnTimeline) -> bool:
        """Release ``timeline`` iff it still owns the response lifecycle."""

        if self._active_timeline is not timeline:
            return False
        self._active_timeline = None
        return True

    def record_brain_event(
        self,
        timeline: TurnTimeline,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        name = str(event.get("event") or "")
        output = _agent_output(timeline)
        output["last_event"] = name
        if name == "brain_request_started":
            output.update({"phase": "brain_requested", "outcome": "pending"})
        elif name == "brain_state":
            state = str(event.get("state") or "")
            output["brain_state"] = state
            if state == "thinking":
                output["phase"] = "brain_thinking"
            elif state == "speaking":
                if "brain_first_delta_at" not in timeline.timestamps:
                    output["phase"] = "brain_speaking_without_delta"
                    output["risk"] = "awaiting_first_delta"
                else:
                    output["phase"] = "brain_speaking"
        elif name == "brain_first_delta":
            output.update(
                {
                    "phase": "llm_text_streaming",
                    "first_delta_seen": True,
                    "risk": "",
                }
            )
        elif name == "brain_done":
            if "brain_first_delta_at" not in timeline.timestamps:
                output.update(
                    {
                        "phase": "silent_failure",
                        "outcome": "brain_done_without_delta",
                        "silent_failure": True,
                    }
                )
            else:
                output.update({"outcome": "brain_done"})
        elif name == "brain_cancelled":
            output.update({"phase": "cancelled", "outcome": "brain_cancelled"})
        elif name == "brain_error":
            output.update(
                {
                    "phase": "silent_failure"
                    if "brain_first_delta_at" not in timeline.timestamps
                    else "brain_error_after_delta",
                    "outcome": "brain_error",
                    "silent_failure": "brain_first_delta_at"
                    not in timeline.timestamps,
                    "error_code": event.get("code", ""),
                    "error_message": event.get("message", ""),
                }
            )
        timeline.set_attr("agent_output", output)
        return output

    def record_tts_event(
        self,
        timeline: TurnTimeline,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        name = str(event.get("event") or "")
        output = _agent_output(timeline)
        output["last_event"] = name
        if name == "tts_first_text_sent":
            output.update({"phase": "tts_text_sent"})
        elif name == "tts_provider_first_audio":
            output.update({"phase": "tts_audio_ready"})
        timeline.set_attr("agent_output", output)
        return output

    def record_llm_error(
        self,
        timeline: TurnTimeline,
        event: Any,
    ) -> dict[str, Any]:
        output = _agent_output(timeline)
        recoverable = bool(getattr(event, "recoverable", False))
        error = getattr(event, "error", None)
        output.update(
            {
                "last_event": "llm_error",
                "llm_error_recoverable": recoverable,
                "error_message": str(error or ""),
            }
        )
        if not recoverable and "brain_first_delta_at" not in timeline.timestamps:
            output.update(
                {
                    "phase": "silent_failure",
                    "outcome": "llm_error_without_delta",
                    "silent_failure": True,
                }
            )
        timeline.set_attr("agent_output", output)
        return output

    def record_first_delta_timeout(
        self,
        timeline: TurnTimeline,
        *,
        timeout_sec: float,
    ) -> dict[str, Any]:
        output = _agent_output(timeline)
        output.update(
            {
                "phase": "first_delta_timeout",
                "risk": "awaiting_first_delta",
                "first_delta_timeout_sec": timeout_sec,
            }
        )
        if output.get("outcome") in (None, "", "pending"):
            output["outcome"] = "pending"
        timeline.set_attr("agent_output", output)
        return output

    def record_agent_state(
        self,
        timeline: TurnTimeline,
        *,
        old_state: str,
        new_state: str,
    ) -> dict[str, Any]:
        output = _agent_output(timeline)
        output["framework_agent_state"] = new_state
        if new_state == "speaking":
            if "tts_provider_first_audio_at" not in timeline.timestamps:
                output.update(
                    {
                        "phase": "framework_speaking_without_provider_audio",
                        "risk": "awaiting_tts_provider_audio",
                    }
                )
            else:
                output.update({"phase": "framework_speaking"})
        elif old_state == "speaking" and new_state in ("idle", "listening"):
            output.update({"outcome": output.get("outcome") or "playback_done"})
        timeline.set_attr("agent_output", output)
        return output


def _agent_output(timeline: TurnTimeline) -> dict[str, Any]:
    return dict(timeline.attrs.get("agent_output") or {})
