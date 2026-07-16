"""Provider event observers for a LiveKit streaming session."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.common.config import ObservabilityConfig
from eidolon.livekit.agent.observability import TurnTimeline
from .agent_output_coordinator import AgentOutputCoordinator

logger = logging.getLogger("agent.session.provider_events")


class ProviderEventObserver:
    """Bridge provider-native timing events into the active turn timeline."""

    def __init__(
        self,
        *,
        factory: Any,
        get_timeline: Callable[[], TurnTimeline | None],
        flush_timeline: Callable[[TurnTimeline, str], None] | None = None,
        append_timeline_snapshot: Callable[[TurnTimeline, str], None] | None = None,
        publish_milestone: Callable[[TurnTimeline, str, str], None] | None = None,
        first_delta_timeout_sec: float | None = None,
        stt_pending_event_window_sec: float | None = None,
        stt_pending_event_preroll_sec: float | None = None,
        stt_pending_event_max_count: int | None = None,
    ) -> None:
        self._factory = factory
        self._get_timeline = get_timeline
        self._flush_timeline = flush_timeline
        self._append_timeline_snapshot = append_timeline_snapshot
        self._publish_milestone = publish_milestone
        observability_defaults = ObservabilityConfig()
        if first_delta_timeout_sec is None:
            first_delta_timeout_sec = observability_defaults.llm_first_delta_timeout_ms / 1000.0
        self._first_delta_timeout_sec = first_delta_timeout_sec
        if stt_pending_event_window_sec is None:
            stt_pending_event_window_sec = (
                observability_defaults.stt_pending_provider_event_window_ms / 1000.0
            )
        if stt_pending_event_preroll_sec is None:
            stt_pending_event_preroll_sec = (
                observability_defaults.stt_pending_provider_event_preroll_ms / 1000.0
            )
        if stt_pending_event_max_count is None:
            stt_pending_event_max_count = (
                observability_defaults.stt_pending_provider_event_max_count
            )
        self._stt_pending_event_window_sec = max(0.0, float(stt_pending_event_window_sec))
        self._stt_pending_event_preroll_sec = max(0.0, float(stt_pending_event_preroll_sec))
        self._stt_pending_event_max_count = max(1, int(stt_pending_event_max_count))
        self._first_delta_watchdog: asyncio.Task | None = None
        self.llm_metrics_observer_installed = False
        self.brain_provider_observer_installed = False
        self.stt_provider_observer_installed = False
        self.tts_provider_observer_installed = False
        self.pending_stt_provider_events: list[dict[str, Any]] = []
        self.agent_output = AgentOutputCoordinator()

    def cancel_output_watchdog(self) -> None:
        self._cancel_first_delta_watchdog()

    def install_all(self) -> None:
        self.install_llm_metrics_observer()
        self.install_brain_provider_event_observer()
        self.install_stt_provider_event_observer()
        self.install_tts_provider_event_observer()

    def install_llm_metrics_observer(self) -> None:
        """Bridge LiveKit LLM metrics into the active turn timeline."""
        if self.llm_metrics_observer_installed:
            return
        llm_plugin = getattr(getattr(self._factory, "llm", None), "llm", None)
        if llm_plugin is None or not hasattr(llm_plugin, "on"):
            return

        def _on_metrics_collected(metrics: Any) -> None:
            timeline = self._get_timeline()
            if timeline is None:
                return
            if not self._should_record_llm_metrics(timeline, metrics):
                logger.debug(
                    "[ProviderEventObserver] ignored LLM metrics outside current turn "
                    "timeline=%s request_id=%s",
                    timeline.turn_id,
                    getattr(metrics, "request_id", ""),
                )
                return
            ttft = getattr(metrics, "ttft", None)
            duration = getattr(metrics, "duration", None)
            if ttft is not None and ttft >= 0:
                timeline.mark_after("llm_first_delta_at", "llm_started_at", ttft)
                self._cancel_first_delta_watchdog()
            timeline.set_attr(
                "llm_metrics",
                {
                    "request_id": getattr(metrics, "request_id", ""),
                    "ttft_ms": ttft * 1000 if ttft is not None else None,
                    "duration_ms": duration * 1000 if duration is not None else None,
                    "completion_tokens": getattr(metrics, "completion_tokens", 0),
                    "prompt_tokens": getattr(metrics, "prompt_tokens", 0),
                    "total_tokens": getattr(metrics, "total_tokens", 0),
                    "cancelled": getattr(metrics, "cancelled", False),
                },
            )

        llm_plugin.on("metrics_collected", _on_metrics_collected)
        llm_plugin.on("error", self._on_llm_error)
        self.llm_metrics_observer_installed = True

    def _on_llm_error(self, event: Any) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        if not self._timeline_has_reply_context(timeline):
            logger.debug(
                "[ProviderEventObserver] ignored LLM error outside current turn timeline=%s",
                timeline.turn_id,
            )
            return
        timestamp = getattr(event, "timestamp", None)
        if isinstance(timestamp, (int, float)):
            timeline.mark_at("llm_error_at", float(timestamp))
        else:
            timeline.mark("llm_error_at")
        timeline.set_attr(
            "llm_error",
            {
                "label": getattr(event, "label", ""),
                "recoverable": bool(getattr(event, "recoverable", False)),
                "error": str(getattr(event, "error", "") or ""),
            },
        )
        output = self.agent_output.record_llm_error(timeline, event)
        self._emit_milestone(timeline, "llm_error", "livekit_llm_error")
        self._cancel_first_delta_watchdog()
        if bool(output.get("silent_failure")):
            self._flush_silent_output_if_terminal(timeline, output)

    def install_brain_provider_event_observer(self) -> None:
        """Bridge provider-native brain RPC timing events into the timeline."""
        if self.brain_provider_observer_installed:
            return
        llm_plugin = getattr(getattr(self._factory, "llm", None), "llm", None)
        if llm_plugin is None or not hasattr(llm_plugin, "on"):
            return

        mark_by_event = {
            "brain_request_started": "brain_request_started_at",
            "brain_request_sent": "brain_request_sent_at",
            "brain_first_delta": "brain_first_delta_at",
            "brain_done": "brain_done_at",
            "brain_cancelled": "brain_cancelled_at",
            "brain_state": None,
            "brain_error": "brain_error_at",
        }

        def _on_provider_event(event: Any) -> None:
            timeline = self._get_timeline()
            if timeline is None or not isinstance(event, dict):
                return
            mark = mark_by_event.get(str(event.get("event") or ""))
            event_name = str(event.get("event") or "")
            if event_name == "brain_state":
                state = str(event.get("state") or "")
                if state == "thinking":
                    mark = "brain_state_thinking_at"
                elif state == "speaking":
                    mark = "brain_state_speaking_at"
            if mark is None and event_name != "brain_state":
                return
            if not self._should_record_brain_event(timeline, event_name, event):
                logger.debug(
                    "[ProviderEventObserver] ignored brain provider event outside "
                    "current turn timeline=%s event=%s turn_id=%s request_id=%s",
                    timeline.turn_id,
                    event_name,
                    event.get("turn_id", ""),
                    event.get("request_id", ""),
                )
                return
            if mark is not None:
                timestamp = event.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    timeline.mark_at(mark, float(timestamp))
                    if mark == "brain_first_delta_at":
                        timeline.mark_at("llm_first_delta_at", float(timestamp))
                        self._cancel_first_delta_watchdog()
                else:
                    timeline.mark(mark)
                    if mark == "brain_first_delta_at":
                        timeline.mark("llm_first_delta_at")
                        self._cancel_first_delta_watchdog()
            brain_rpc = dict(timeline.attrs.get("brain_rpc") or {})
            brain_rpc.update(
                {
                    "provider": event.get("provider", ""),
                    "turn_id": event.get("turn_id", brain_rpc.get("turn_id", "")),
                    "request_id": event.get(
                        "request_id",
                        brain_rpc.get("request_id", ""),
                    ),
                    "conversation_id": event.get(
                        "conversation_id",
                        brain_rpc.get("conversation_id", ""),
                    ),
                    "last_event": event.get("event", ""),
                }
            )
            if "user_text_source" in event:
                brain_rpc["user_text_source"] = event.get("user_text_source")
            if "text_overridden" in event:
                brain_rpc["text_overridden"] = bool(event.get("text_overridden"))
            if "text_chars" in event:
                brain_rpc["text_chars"] = event.get("text_chars")
            if "framework_text_chars" in event:
                brain_rpc["framework_text_chars"] = event.get("framework_text_chars")
            if "attempt" in event:
                brain_rpc["attempt"] = event.get("attempt")
            timeline.set_attr("brain_rpc", brain_rpc)
            output = self.agent_output.record_brain_event(timeline, event)
            if event_name in {
                "brain_request_sent",
                "brain_first_delta",
                "brain_done",
                "brain_cancelled",
                "brain_error",
            }:
                self._emit_milestone(timeline, event_name, event_name)
            if event_name == "brain_request_sent":
                self._arm_first_delta_watchdog(timeline)
            elif event_name in {"brain_done", "brain_cancelled", "brain_error"}:
                self._cancel_first_delta_watchdog()
            self._flush_silent_output_if_terminal(timeline, output)

        llm_plugin.on("provider_event", _on_provider_event)
        self.brain_provider_observer_installed = True

    def install_tts_provider_event_observer(self) -> None:
        """Bridge provider-native TTS streaming timing events into timeline."""
        if self.tts_provider_observer_installed:
            return
        tts_plugin = getattr(getattr(self._factory, "tts", None), "tts", None)
        if tts_plugin is None or not hasattr(tts_plugin, "on"):
            return

        mark_by_event = {
            "tts_stream_started": "tts_stream_started_at",
            "tts_connection_acquired": "tts_connection_acquired_at",
            "tts_request_started": "tts_request_started_at",
            "tts_first_text_sent": "tts_first_text_sent_at",
            "tts_provider_first_audio": "tts_provider_first_audio_at",
        }

        def _on_provider_event(event: Any) -> None:
            timeline = self._get_timeline()
            if timeline is None or not isinstance(event, dict):
                return
            mark = mark_by_event.get(str(event.get("event") or ""))
            if mark is None:
                return
            if not self._should_record_tts_event(timeline):
                logger.debug(
                    "[ProviderEventObserver] ignored TTS provider event outside "
                    "current turn timeline=%s event=%s",
                    timeline.turn_id,
                    event.get("event", ""),
                )
                return
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timeline.mark_at(mark, float(timestamp))
            else:
                timeline.mark(mark)
            tts_stream = dict(timeline.attrs.get("tts_stream") or {})
            tts_stream.update(
                {
                    "provider": event.get("provider", ""),
                    "model": event.get("model", tts_stream.get("model", "")),
                    "last_event": event.get("event", ""),
                }
            )
            timeline.set_attr("tts_stream", tts_stream)
            self.agent_output.record_tts_event(timeline, event)
            if mark == "tts_provider_first_audio_at":
                self._emit_milestone(
                    timeline,
                    "tts_provider_first_audio",
                    "tts_provider_first_audio",
                )
            if mark in {"tts_first_text_sent_at", "tts_provider_first_audio_at"}:
                self._cancel_first_delta_watchdog()

        tts_plugin.on("provider_event", _on_provider_event)
        self.tts_provider_observer_installed = True

    def _emit_milestone(self, timeline: TurnTimeline, milestone: str, reason: str) -> None:
        if self._publish_milestone is not None:
            self._publish_milestone(timeline, milestone, reason)

    def _should_record_llm_metrics(self, timeline: TurnTimeline, metrics: Any) -> bool:
        request_id = str(getattr(metrics, "request_id", "") or "")
        if request_id and not self._event_identity_matches_brain_rpc(
            timeline,
            request_id=request_id,
        ):
            return False
        return self._timeline_has_reply_context(timeline)

    def _should_record_brain_event(
        self,
        timeline: TurnTimeline,
        event_name: str,
        event: dict[str, Any],
    ) -> bool:
        if event_name == "brain_request_started":
            return self._timeline_has_reply_context(timeline)
        if event_name == "brain_request_sent":
            return self._timeline_has_reply_context(timeline) and (
                self._event_identity_matches_brain_rpc(
                    timeline,
                    turn_id=str(event.get("turn_id") or ""),
                    request_id=str(event.get("request_id") or ""),
                )
                or self._is_retry_attempt_after_terminal_brain_event(
                    timeline,
                    event,
                )
            )

        event_turn_id = str(event.get("turn_id") or "")
        event_request_id = str(event.get("request_id") or "")
        if event_turn_id or event_request_id:
            if not self._brain_rpc_has_identity(timeline):
                return False
            return self._event_identity_matches_brain_rpc(
                timeline,
                turn_id=event_turn_id,
                request_id=event_request_id,
            )
        return self._timeline_has_reply_context(timeline)

    @staticmethod
    def _is_retry_attempt_after_terminal_brain_event(
        timeline: TurnTimeline,
        event: dict[str, Any],
    ) -> bool:
        brain_rpc = timeline.attrs.get("brain_rpc")
        if not isinstance(brain_rpc, dict):
            return False
        if str(brain_rpc.get("last_event") or "") not in {
            "brain_error",
            "brain_cancelled",
        }:
            return False
        known_attempt = brain_rpc.get("attempt")
        event_attempt = event.get("attempt")
        if not isinstance(known_attempt, (int, float)):
            return False
        if not isinstance(event_attempt, (int, float)):
            return False
        return float(event_attempt) > float(known_attempt)

    def _should_record_tts_event(self, timeline: TurnTimeline) -> bool:
        return self._timeline_has_reply_context(timeline)

    @staticmethod
    def _timeline_has_reply_context(timeline: TurnTimeline) -> bool:
        attrs = timeline.attrs
        if any(
            key in attrs
            for key in (
                "canonical_user_text",
                "framework_commit_request",
                "framework_completed_turn",
                "brain_rpc",
                "tts_stream",
            )
        ):
            return True
        return any(
            mark in timeline.timestamps
            for mark in (
                "turn_committed_at",
                "llm_started_at",
                "brain_request_started_at",
                "brain_request_sent_at",
                "brain_first_delta_at",
            )
        )

    @staticmethod
    def _brain_rpc_has_identity(timeline: TurnTimeline) -> bool:
        brain_rpc = timeline.attrs.get("brain_rpc")
        if not isinstance(brain_rpc, dict):
            return False
        return bool(brain_rpc.get("turn_id") or brain_rpc.get("request_id"))

    @staticmethod
    def _event_identity_matches_brain_rpc(
        timeline: TurnTimeline,
        *,
        turn_id: str = "",
        request_id: str = "",
    ) -> bool:
        brain_rpc = timeline.attrs.get("brain_rpc")
        if not isinstance(brain_rpc, dict):
            return True
        known_turn_id = str(brain_rpc.get("turn_id") or "")
        known_request_id = str(brain_rpc.get("request_id") or "")
        if turn_id and known_turn_id and turn_id != known_turn_id:
            return False
        if request_id and known_request_id and request_id != known_request_id:
            return False
        return True

    def _arm_first_delta_watchdog(self, timeline: TurnTimeline) -> None:
        if self._append_timeline_snapshot is None:
            return
        if self._first_delta_timeout_sec <= 0:
            return
        if "brain_first_delta_at" in timeline.timestamps:
            return
        if self._first_delta_watchdog is not None and not self._first_delta_watchdog.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        turn_id = timeline.turn_id
        timeout_sec = self._first_delta_timeout_sec
        self._first_delta_watchdog = loop.create_task(
            self._first_delta_timeout_after(turn_id, timeout_sec)
        )

    def _cancel_first_delta_watchdog(self) -> None:
        task = self._first_delta_watchdog
        if task is not None and not task.done():
            task.cancel()
        self._first_delta_watchdog = None

    async def _first_delta_timeout_after(self, turn_id: str, timeout_sec: float) -> None:
        try:
            await asyncio.sleep(timeout_sec)
        except asyncio.CancelledError:
            return
        timeline = self._get_timeline()
        if timeline is None or timeline.turn_id != turn_id:
            return
        if "brain_first_delta_at" in timeline.timestamps:
            return
        if any(
            mark in timeline.timestamps
            for mark in (
                "brain_done_at",
                "brain_cancelled_at",
                "llm_error_at",
                "tts_first_text_sent_at",
                "tts_provider_first_audio_at",
            )
        ):
            return
        timeline.mark("llm_first_delta_timeout_at")
        self.agent_output.record_first_delta_timeout(
            timeline,
            timeout_sec=timeout_sec,
        )
        if self._append_timeline_snapshot is not None:
            self._append_timeline_snapshot(
                timeline,
                "agent_output_first_delta_timeout",
            )

    def _flush_silent_output_if_terminal(
        self,
        timeline: TurnTimeline,
        output: dict[str, Any],
    ) -> None:
        if self._flush_timeline is None:
            return
        if not bool(output.get("silent_failure")):
            return
        outcome = str(output.get("outcome") or "")
        if outcome not in {"brain_done_without_delta", "llm_error_without_delta"}:
            return
        self._flush_timeline(timeline, f"agent_output_{outcome}")

    def install_stt_provider_event_observer(self) -> None:
        """Bridge provider-native STT streaming timing events into timeline."""
        if self.stt_provider_observer_installed:
            return
        stt_plugin = getattr(getattr(self._factory, "stt", None), "stt", None)
        if stt_plugin is None or not hasattr(stt_plugin, "on"):
            return

        def _on_provider_event(event: Any) -> None:
            if not isinstance(event, dict):
                return
            if self._get_timeline() is None:
                self.remember_pending_stt_provider_event(event)
                return
            self.record_stt_provider_event(event)

        stt_plugin.on("provider_event", _on_provider_event)
        self.stt_provider_observer_installed = True

    def remember_pending_stt_provider_event(self, event: dict[str, Any]) -> None:
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return
        pending = self.pending_stt_provider_events
        pending.append(dict(event))
        cutoff = float(timestamp) - self._stt_pending_event_window_sec
        self.pending_stt_provider_events = [
            item
            for item in pending[-self._stt_pending_event_max_count:]
            if isinstance(item.get("timestamp"), (int, float))
            and float(item["timestamp"]) >= cutoff
        ]

    def apply_pending_stt_provider_events(self) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        speech_started_at = timeline.timestamps.get("speech_started_at")
        if speech_started_at is None:
            return
        pending = list(self.pending_stt_provider_events)
        self.pending_stt_provider_events = []
        for event in pending:
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, (int, float)):
                continue
            if float(timestamp) < speech_started_at - self._stt_pending_event_preroll_sec:
                continue
            self.record_stt_provider_event(event)

    def record_stt_provider_event(self, event: dict[str, Any]) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        mark_by_event = {
            "stt_stream_started": "stt_stream_started_at",
            "stt_ws_connected": "stt_ws_connected_at",
            "stt_first_audio_sent": "stt_stream_first_audio_sent_at",
            "stt_turn_first_audio_sent": "stt_first_audio_sent_at",
            "stt_flush_sent": "stt_flush_sent_at",
            "stt_provider_first_partial": "stt_provider_first_partial_at",
            "stt_provider_final": "stt_provider_final_at",
        }
        mark = mark_by_event.get(str(event.get("event") or ""))
        if mark is None:
            return
        event_turn_id = event.get("turn_id")
        if event_turn_id and event_turn_id != timeline.turn_id:
            return
        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timeline.mark_at(mark, float(timestamp))
        else:
            timeline.mark(mark)
        stt_stream = dict(timeline.attrs.get("stt_stream") or {})
        stt_stream.update(
            {
                "provider": event.get("provider", ""),
                "model": event.get("model", stt_stream.get("model", "")),
                "stream_id": event.get("stream_id", stt_stream.get("stream_id", "")),
                "language": event.get("language", stt_stream.get("language", "")),
                "last_event": event.get("event", ""),
            }
        )
        text_preview = event.get("text_preview")
        if isinstance(text_preview, str) and text_preview:
            stt_stream["last_text_preview"] = text_preview
        timeline.set_attr("stt_stream", stt_stream)

    def observe_stt_turn_audio(self) -> None:
        timeline = self._get_timeline()
        if timeline is None:
            return
        speech_started_at = timeline.timestamps.get("speech_started_at")
        if speech_started_at is None:
            return
        stt_stage = getattr(self._factory, "stt", None)
        stt_plugin = getattr(stt_stage, "_stt", None) or getattr(stt_stage, "stt", None)
        observer = getattr(stt_plugin, "observe_next_audio_for_turn", None)
        if not callable(observer):
            return
        try:
            observed = bool(
                observer(
                    turn_id=timeline.turn_id,
                    speech_started_at=speech_started_at,
                )
            )
            timeline.set_attr("stt_turn_audio_observer_installed", observed)
        except Exception:
            logger.exception("[ProviderEventObserver] failed to arm STT turn-audio observer")
