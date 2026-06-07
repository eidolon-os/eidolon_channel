"""Provider event observers for a LiveKit streaming session."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from eidolon.livekit.agent.observability import TurnTimeline

logger = logging.getLogger("agent.session.provider_events")


class ProviderEventObserver:
    """Bridge provider-native timing events into the active turn timeline."""

    def __init__(
        self,
        *,
        factory: Any,
        get_timeline: Callable[[], TurnTimeline | None],
    ) -> None:
        self._factory = factory
        self._get_timeline = get_timeline
        self.llm_metrics_observer_installed = False
        self.brain_provider_observer_installed = False
        self.stt_provider_observer_installed = False
        self.tts_provider_observer_installed = False
        self.pending_stt_provider_events: list[dict[str, Any]] = []

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
            ttft = getattr(metrics, "ttft", None)
            duration = getattr(metrics, "duration", None)
            if ttft is not None and ttft >= 0:
                timeline.mark_after("llm_first_delta_at", "llm_started_at", ttft)
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
        self.llm_metrics_observer_installed = True

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
        }

        def _on_provider_event(event: Any) -> None:
            timeline = self._get_timeline()
            if timeline is None or not isinstance(event, dict):
                return
            mark = mark_by_event.get(str(event.get("event") or ""))
            if mark is None:
                return
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)):
                timeline.mark_at(mark, float(timestamp))
                if mark == "brain_first_delta_at":
                    timeline.mark_at("llm_first_delta_at", float(timestamp))
            else:
                timeline.mark(mark)
                if mark == "brain_first_delta_at":
                    timeline.mark("llm_first_delta_at")
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
            timeline.set_attr("brain_rpc", brain_rpc)

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
            "tts_request_started": "tts_request_started_at",
            "tts_provider_first_audio": "tts_provider_first_audio_at",
        }

        def _on_provider_event(event: Any) -> None:
            timeline = self._get_timeline()
            if timeline is None or not isinstance(event, dict):
                return
            mark = mark_by_event.get(str(event.get("event") or ""))
            if mark is None:
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

        tts_plugin.on("provider_event", _on_provider_event)
        self.tts_provider_observer_installed = True

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
        cutoff = float(timestamp) - 2.0
        self.pending_stt_provider_events = [
            item
            for item in pending[-32:]
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
            if float(timestamp) < speech_started_at - 0.5:
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
            logger.exception(
                "[ProviderEventObserver] failed to arm STT turn-audio observer"
            )
