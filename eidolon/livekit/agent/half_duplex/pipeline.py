"""Half-duplex PTT pipeline.

This pipeline is intentionally separate from ``StreamingPipeline``. It uses
LiveKit as the room media/data engine, but owns PTT turn boundaries itself:
client ``ptt=true`` opens a server-side audio segment, ``ptt=false`` closes it,
then the closed segment is transcribed once and passed to AgentSession as text.

No streaming STT transcript event, EOT score, or full-duplex interruption owner
participates in the PTT commit decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable

from eidolon_sdk.biz.contracts import (
    COMPANION_UI_STATE_TOPIC,
    CONTROL_OP_PLAYBACK_STOP,
    CONTROL_OP_PTT_TURN_STATUS,
    CONTROL_TOPIC,
    INTERACTION_MODE_PTT,
    SESSION_END_ERROR,
    SESSION_END_USER_LEFT,
    SESSION_INTENT_USER_INITIATED,
    WIRE_SCHEMA_VERSION,
)

from eidolon.livekit.agent.observability import TurnTimeline
from eidolon.livekit.agent.runtime.interaction_mode import (
    resolve_idle_policy,
    resolve_welcome_text,
)
from eidolon.livekit.agent.shared.pipeline import BasePipeline
from eidolon.livekit.agent.shared.types import (
    PipelineCallbacks,
    PipelineState,
    generate_turn_id,
)
from eidolon.livekit.agent.session.client_control import (
    append_client_control_event,
    build_client_control_event,
    build_session_client_control_envelope,
)
from eidolon.livekit.agent.session.agent_output_coordinator import AgentOutputCoordinator
from eidolon.livekit.agent.session.idle import IdleWatchdog
from eidolon.livekit.agent.session.provider_events import ProviderEventObserver
from eidolon.livekit.agent.session.room_data import RoomDataHandler
from eidolon.livekit.common.config import ObservabilityConfig, TurnPolicyConfig

from .control import (
    PTT_OUTCOME_COMMITTED,
    PTT_OUTCOME_FINALIZING,
    PTT_OUTCOME_RECORDING,
    build_ptt_turn_status_payload,
    ptt_rejected_outcome,
    should_drop_pending_ptt_control_event,
)
from .ptt_segment import PttAudioSegmentConfig, PttAudioSegmentRecorder
from .ptt_transcriber import PttSegmentTranscriber, PttSegmentTranscriberConfig
from .ptt_turn_controller import HalfDuplexPttTurnController, PttSegmentTurnResult

if TYPE_CHECKING:
    from livekit.agents.voice import AgentSession
    from livekit.rtc import Room

logger = logging.getLogger("agent.half_duplex")


class HalfDuplexPttPipeline(BasePipeline):
    """Room-level half-duplex PTT pipeline using complete audio segments."""

    def __init__(
        self,
        factory: Any,
        *,
        instructions: str = "",
        welcome_message: str | None = None,
        audio_sample_rate: int = 16_000,
        turn_policy: TurnPolicyConfig | None = None,
        observability: ObservabilityConfig | None = None,
        callbacks: PipelineCallbacks | None = None,
        session_intent: str = SESSION_INTENT_USER_INITIATED,
        on_session_started: Callable[[], Awaitable[None]] | None = None,
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        on_idle_disconnect: Callable[[], Any] | None = None,
        on_session_closed: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        self._instructions = instructions
        self._welcome_message = welcome_message
        self._audio_sample_rate = audio_sample_rate
        self._turn_policy = turn_policy or TurnPolicyConfig()
        self._session_intent = session_intent
        self._observability = observability or ObservabilityConfig()
        self._on_session_started = on_session_started
        self._on_session_end = on_session_end
        self._on_idle_disconnect = on_idle_disconnect
        self._on_session_closed = on_session_closed
        self._close_error: Any | None = None
        self._session: AgentSession | None = None
        self._session_closed_event: asyncio.Event = asyncio.Event()
        self._room_disconnected_event: asyncio.Event = asyncio.Event()
        self._timeline: TurnTimeline | None = None
        self._flushed_timeline_ids: set[str] = set()
        self._agent_output = AgentOutputCoordinator()
        self._provider_events = ProviderEventObserver(
            factory=factory,
            get_timeline=lambda: self._timeline,
            get_output_timeline=lambda: self._agent_output.active_timeline,
            agent_output=self._agent_output,
            flush_timeline=self._flush_output_timeline,
            first_delta_timeout_sec=(
                self._observability.llm_first_delta_timeout_ms / 1000.0
            ),
        )
        self._pending_client_control_events: list[dict[str, str]] = []
        self._room_data = RoomDataHandler(get_timeline=lambda: self._timeline)
        self._last_ptt_held = False
        self._track_tasks: set[asyncio.Task[Any]] = set()
        self._observed_audio_track_ids: set[int] = set()
        self._turn_tasks: set[asyncio.Task[Any]] = set()
        self._ptt_controller = self._build_ptt_controller()
        idle_policy = resolve_idle_policy(
            session_intent=session_intent,
            idle_config=self._turn_policy.idle,
        )
        self._idle_timeout_sec: float = idle_policy.timeout_sec
        self._idle_end_reason = idle_policy.end_reason
        self._idle_disconnect_grace_sec: float = self._turn_policy.idle.disconnect_grace_ms / 1000.0
        self._idle_disconnect_started = False
        self._idle_watchdog_controller = self._build_idle_watchdog()

    def _welcome_on_enter_text(self) -> str | None:
        return resolve_welcome_text(
            session_intent=self._session_intent,
            welcome_message=self._welcome_message,
        )

    async def run(self, room: Room) -> None:
        from livekit.agents.voice import Agent, AgentSession
        from livekit.agents.voice.room_io import AudioOutputOptions, RoomOptions

        logger.info("[HalfDuplexPttPipeline] starting room=%s", room.name)
        self._room = room
        self._started = True
        self._install_room_observers(room)
        await self._warmup_stages()
        self._provider_events.install_all()

        pipeline = self

        class PttAgent(Agent):
            async def on_enter(self) -> None:
                welcome = pipeline._welcome_on_enter_text()
                if welcome is None:
                    logger.info(
                        "[HalfDuplexPttPipeline] welcome suppressed intent=%s",
                        pipeline._session_intent,
                    )
                    return
                logger.info(
                    "[HalfDuplexPttPipeline] welcome on_enter room=%s",
                    getattr(pipeline._room, "name", ""),
                )
                self.session.say(welcome, allow_interruptions=True)

        session = AgentSession()
        self._session = session
        session.on("agent_state_changed", self._on_agent_state_changed)
        session.on("error", self._on_session_error)
        session.on("close", self._on_session_close)

        await session.start(
            agent=PttAgent(
                instructions=self._instructions,
                llm=self._factory.llm.llm,
                tts=self._factory.tts.tts,
            ),
            room=room,
            room_options=RoomOptions(
                audio_input=False,
                audio_output=AudioOutputOptions(sample_rate=self._audio_sample_rate),
                text_output=True,
            ),
        )
        if self._on_session_started is not None:
            await self._on_session_started()
        self._publish_companion_ui_state("listening", "session_started")
        logger.info("[HalfDuplexPttPipeline] session started")
        self._start_idle_watchdog()

        session_wait = asyncio.create_task(self._session_closed_event.wait())
        room_wait = asyncio.create_task(self._room_disconnected_event.wait())
        try:
            done, pending = await asyncio.wait(
                {session_wait, room_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
            if session_wait in done and not self._idle_disconnect_started:
                await self._end_serving_on_close()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        logger.info("[HalfDuplexPttPipeline] shutting down")
        self._stop_idle_watchdog()
        self._provider_events.cancel_output_watchdog()
        for task in list(self._track_tasks):
            task.cancel()
        for task in list(self._turn_tasks):
            task.cancel()
        if self._track_tasks:
            await asyncio.gather(*self._track_tasks, return_exceptions=True)
        if self._turn_tasks:
            await asyncio.gather(*self._turn_tasks, return_exceptions=True)
        self._track_tasks.clear()
        self._turn_tasks.clear()
        output_timeline = self._agent_output.active_timeline
        if output_timeline is not None:
            self._flush_output_timeline(output_timeline, "ptt_pipeline_shutdown")
        session = self._session
        if session is not None:
            close = getattr(session, "aclose", None)
            if callable(close):
                await close()
        await self._shutdown_stages()
        close_factory = getattr(self._factory, "aclose", None)
        if callable(close_factory):
            await close_factory()
        await super().shutdown()

    def _build_ptt_controller(self) -> HalfDuplexPttTurnController:
        ptt = self._turn_policy.ptt
        recorder = PttAudioSegmentRecorder(
            config=PttAudioSegmentConfig(
                sample_rate=self._audio_sample_rate,
                max_duration_sec=ptt.segment_max_audio_ms / 1000.0,
            )
        )
        transcriber = PttSegmentTranscriber(
            self._factory.stt,
            config=PttSegmentTranscriberConfig(
                strategy=ptt.segment_stt_strategy,
                min_audio_duration_sec=ptt.segment_min_audio_ms / 1000.0,
                min_rms_ppm=ptt.segment_min_rms_ppm,
            ),
        )
        return HalfDuplexPttTurnController(
            recorder=recorder,
            transcriber=transcriber,
            agent_output_active=self._agent_output_active_for_ptt,
            preempt_agent_output=self._preempt_agent_output_for_ptt,
            tap_to_stop_max_audio_sec=ptt.segment_tap_to_stop_max_audio_ms / 1000.0,
        )

    def _build_idle_watchdog(self) -> IdleWatchdog:
        return IdleWatchdog(
            timeout_sec=self._idle_timeout_sec,
            get_session=lambda: self._session,
            get_room=lambda: getattr(self, "_room", None),
            get_timeline=lambda: self._timeline,
            session_closed_event=self._session_closed_event,
            on_idle_disconnect=(
                self._disconnect_idle_room if self._on_idle_disconnect is not None else None
            ),
            on_session_end=self._on_session_end,
            disconnect_grace_sec=self._idle_disconnect_grace_sec,
            idle_end_reason=self._idle_end_reason,
            is_busy=self._idle_busy,
        )

    def _mark_activity(self) -> None:
        self._idle_watchdog_controller.mark_activity()

    def _start_idle_watchdog(self) -> None:
        self._idle_watchdog_controller.start()

    def _stop_idle_watchdog(self) -> None:
        self._idle_watchdog_controller.stop()

    def _idle_busy(self) -> bool:
        ptt_controller = getattr(self, "_ptt_controller", None)
        if ptt_controller is not None and getattr(ptt_controller, "state", "idle") != "idle":
            return True
        if self._state in {PipelineState.GENERATING, PipelineState.SPEAKING}:
            return True
        session = self._session
        return bool(
            session is not None
            and (
                getattr(session, "agent_state", None) in ("thinking", "speaking")
                or getattr(session, "user_state", None) == "speaking"
            )
        )

    async def _disconnect_idle_room(self) -> None:
        self._idle_disconnect_started = True
        cb = self._on_idle_disconnect
        if cb is None:
            return
        result = cb()
        if hasattr(result, "__await__"):
            await result

    def _install_room_observers(self, room: Room) -> None:
        self._room_data.install(room, on_packet=self._on_room_packet)

        @room.on("track_subscribed")
        def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            del publication
            self._maybe_start_audio_stream(track, participant)

        @room.on("disconnected")
        def _on_disconnected(reason: Any) -> None:
            logger.info("[HalfDuplexPttPipeline] room disconnected reason=%s", reason)
            self._room_disconnected_event.set()

        self._observe_existing_audio_tracks(room)

    def _observe_existing_audio_tracks(self, room: Room) -> None:
        for participant in getattr(room, "remote_participants", {}).values():
            publications = getattr(participant, "track_publications", {})
            for publication in publications.values():
                if not bool(getattr(publication, "subscribed", False)):
                    continue
                track = getattr(publication, "track", None)
                if track is not None:
                    self._maybe_start_audio_stream(track, participant)

    def _maybe_start_audio_stream(self, track: Any, participant: Any) -> None:
        try:
            from livekit import rtc

            audio_kind = getattr(rtc.TrackKind, "KIND_AUDIO", 1)
            if getattr(track, "kind", None) != audio_kind:
                return
            track_id = id(track)
            if track_id in self._observed_audio_track_ids:
                return
            self._observed_audio_track_ids.add(track_id)
            stream = rtc.AudioStream(
                track,
                sample_rate=self._audio_sample_rate,
                num_channels=1,
                capacity=32,
            )
        except Exception:
            logger.exception(
                "[HalfDuplexPttPipeline] failed to start audio stream participant=%s",
                getattr(participant, "identity", ""),
            )
            return
        task = asyncio.create_task(self._consume_audio_stream(stream, participant))
        self._track_tasks.add(task)

        def _discard(done: asyncio.Task[Any]) -> None:
            self._track_tasks.discard(done)
            self._observed_audio_track_ids.discard(track_id)

        task.add_done_callback(_discard)
        logger.info(
            "[HalfDuplexPttPipeline] observing audio track participant=%s",
            getattr(participant, "identity", ""),
        )

    async def _consume_audio_stream(self, stream: Any, participant: Any) -> None:
        try:
            async for event in stream:
                frame = getattr(event, "frame", None)
                if frame is not None:
                    self._ptt_controller.push_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[HalfDuplexPttPipeline] audio stream failed participant=%s",
                getattr(participant, "identity", ""),
            )
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.debug("[HalfDuplexPttPipeline] audio stream close failed", exc_info=True)

    def _on_room_packet(self, packet: Any) -> None:
        participant = getattr(packet, "participant", None)
        state = self._room_data.latest_client_audio_state(
            participant_identity=getattr(participant, "identity", None)
        )
        if state is None:
            return
        held = bool(state.ptt)
        was_held = self._last_ptt_held
        if held and not was_held:
            self._mark_activity()
            result = self._ptt_controller.press()
            if result.action == "reject":
                self._publish_ptt_turn_status(
                    ptt_rejected_outcome(result.reason),
                    result.reason,
                )
                logger.info(
                    "[HalfDuplexPttPipeline] PTT press rejected reason=%s state=%s",
                    result.reason,
                    result.state,
                )
                return
            self._last_ptt_held = True
            self._start_ptt_timeline(result)
            self._publish_ptt_turn_status(PTT_OUTCOME_RECORDING, result.reason)
            logger.info(
                "[HalfDuplexPttPipeline] PTT pressed preempted=%s",
                result.preempted_agent_output,
            )
            return
        if was_held and not held:
            self._mark_activity()
            self._last_ptt_held = False
            self._mark_ptt_released()
            self._publish_ptt_turn_status(PTT_OUTCOME_FINALIZING, "released")
            task = asyncio.create_task(self._finalize_ptt_turn())
            self._turn_tasks.add(task)
            task.add_done_callback(self._turn_tasks.discard)

    async def _finalize_ptt_turn(self) -> None:
        try:
            result = await self._ptt_controller.release()
        except Exception:
            logger.exception("[HalfDuplexPttPipeline] PTT segment transcription failed")
            self._publish_ptt_turn_status(
                ptt_rejected_outcome("transcription_error"),
                "transcription_error",
            )
            self._flush_ptt_timeline(
                terminal={"action": "reject", "reason": "transcription_error"},
                reason="ptt_segment_transcription_error",
            )
            return
        if result.action == "reject":
            self._publish_ptt_turn_status(
                ptt_rejected_outcome(result.reason),
                result.reason,
            )
            self._record_ptt_result(result)
            return
        if result.action != "commit":
            return
        session = self._session
        if session is None:
            self._publish_ptt_turn_status(ptt_rejected_outcome("no_session"), "no_session")
            self._flush_ptt_timeline(
                terminal={"action": "reject", "reason": "no_session"},
                reason="ptt_segment_no_session",
            )
            return
        self._callbacks.on_user_message(result.transcript)
        self._mark_activity()
        self._publish_ptt_turn_status(
            PTT_OUTCOME_COMMITTED,
            result.reason,
            transcript=result.transcript,
        )
        self._record_ptt_result(result)
        try:
            session.generate_reply(user_input=result.transcript, input_modality="audio")
        except Exception:
            output_timeline = self._agent_output.active_timeline
            if output_timeline is not None:
                output_timeline.mark("brain_error_at")
                output_timeline.set_attr("ptt_generate_reply_error", True)
                self._flush_output_timeline(output_timeline, "ptt_generate_reply_error")
            raise

    def _record_ptt_result(self, result: PttSegmentTurnResult) -> None:
        logger.info(
            "[HalfDuplexPttPipeline] PTT result action=%s reason=%s mode=%s "
            "latency_ms=%.1f audio_ms=%.1f effective_audio_ms=%.1f "
            "leading_silence_ms=%.1f rms_ppm=%d text=%r",
            result.action,
            result.reason,
            result.stt_mode,
            result.stt_latency_ms,
            result.audio_duration_sec * 1000,
            result.audio_effective_duration_sec * 1000,
            result.audio_leading_silence_sec * 1000,
            result.audio_rms_ppm,
            result.transcript[:80],
        )
        terminal_action = "commit" if result.action == "commit" else "reject"
        if terminal_action == "commit":
            timeline = self._timeline
            if timeline is None:
                return
            terminal = {
                "action": terminal_action,
                "reason": result.reason,
                "transcript_preview": result.transcript[:120],
            }
            self._apply_ptt_terminal(timeline, terminal=terminal, result=result)
            timeline.set_attr("timeline_flush_reason", "ptt_segment_commit_pending_output")
            self._timeline = None
            self._pending_client_control_events = []
            displaced = self._agent_output.claim(timeline)
            if displaced is not None:
                self._flush_output_timeline(displaced, "ptt_output_superseded")
            return
        self._flush_ptt_timeline(
            terminal={
                "action": terminal_action,
                "reason": result.reason,
                "transcript_preview": result.transcript[:120],
            },
            result=result,
            reason=f"ptt_segment_{terminal_action}",
        )

    def _start_ptt_timeline(self, result: PttSegmentTurnResult) -> None:
        timeline = TurnTimeline(generate_turn_id())
        self._timeline = timeline
        now = time.monotonic()
        room = getattr(self, "_room", None)
        timeline.set_attr("room_name", getattr(room, "name", "") if room else "")
        timeline.set_attr("pipeline", "half_duplex_ptt_segment")
        timeline.set_attr("interaction_mode", INTERACTION_MODE_PTT)
        timeline.set_attr("ptt_turn_owner", "segment")
        timeline.set_attr(
            "ptt_segment",
            {
                "state": "recording",
                "preempted_agent_output": result.preempted_agent_output,
            },
        )
        self._apply_pending_client_control_events(timeline)
        timeline.mark_at("speech_started_at", now)
        if result.preempted_agent_output:
            timeline.mark_at("interrupt_started_at", now)
            timeline.mark_at("interrupt_resolved_at", now)
            timeline.record_decision(
                action="cancel",
                reason="explicit_client_ptt",
                rollback_drop_buffered=True,
                intent="hard_stop",
                intent_source="client_audio_state.ptt",
                intent_confidence=1.0,
                source="client_audio_state",
                resolved_reason="explicit_client_ptt",
            )

    def _mark_ptt_released(self) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        timeline.mark("speech_stopped_at")
        ptt_segment = dict(timeline.attrs.get("ptt_segment") or {})
        ptt_segment["state"] = "finalizing"
        timeline.set_attr("ptt_segment", ptt_segment)

    def _flush_ptt_timeline(
        self,
        *,
        terminal: dict[str, object],
        reason: str,
        result: PttSegmentTurnResult | None = None,
    ) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        self._apply_ptt_terminal(timeline, terminal=terminal, result=result)
        self._append_ptt_timeline(timeline, reason)
        self._timeline = None
        self._pending_client_control_events = []

    def _apply_ptt_terminal(
        self,
        timeline: TurnTimeline,
        *,
        terminal: dict[str, object],
        result: PttSegmentTurnResult | None,
    ) -> None:
        now = time.monotonic()
        ptt_segment = dict(timeline.attrs.get("ptt_segment") or {})
        if result is not None:
            ptt_segment.update(
                {
                    "state": "idle",
                    "terminal": terminal,
                    "stt_mode": result.stt_mode,
                    "stt_latency_ms": result.stt_latency_ms,
                    "audio_duration_sec": result.audio_duration_sec,
                    "audio_effective_duration_sec": result.audio_effective_duration_sec,
                    "audio_leading_silence_sec": result.audio_leading_silence_sec,
                    "audio_rms_ppm": result.audio_rms_ppm,
                    "preempted_agent_output": result.preempted_agent_output,
                    "transcript_preview": result.transcript[:120],
                }
            )
            if result.stt_mode != "none":
                stt_started = max(0.0, now - (result.stt_latency_ms / 1000.0))
                timeline.mark_at("stt_stream_started_at", stt_started)
                timeline.mark_at("stt_provider_final_at", now)
                timeline.mark_at("transcript_final_at", now)
                if result.transcript:
                    timeline.mark_at("transcript_interim_first_at", now)
            if result.action == "commit":
                timeline.mark_at("turn_committed_at", now)
        else:
            ptt_segment.update({"state": "idle", "terminal": terminal})
        timeline.set_attr("ptt_segment", ptt_segment)
        timeline.set_attr("ptt_segment_terminal", terminal)

    def _append_ptt_timeline(self, timeline: TurnTimeline, reason: str) -> None:
        if timeline.turn_id in self._flushed_timeline_ids:
            return
        timeline.set_attr("timeline_flush_reason", reason)
        timeline.append_debug_jsonl(self._observability.timeline_debug_path)
        self._flushed_timeline_ids.add(timeline.turn_id)

    def _flush_output_timeline(self, timeline: TurnTimeline, reason: str) -> None:
        if "tts_provider_first_audio_at" in timeline.timestamps:
            timeline.mark("agent_audio_playback_done_at")
        self._append_ptt_timeline(timeline, reason)
        self._agent_output.release(timeline)

    @staticmethod
    def _output_timeline_started(timeline: TurnTimeline) -> bool:
        return any(
            mark in timeline.timestamps
            for mark in (
                "llm_started_at",
                "brain_request_started_at",
                "brain_request_sent_at",
                "brain_first_delta_at",
                "brain_error_at",
                "brain_cancelled_at",
                "tts_stream_started_at",
            )
        )

    def _agent_output_active_for_ptt(self) -> bool:
        return self._state in {PipelineState.GENERATING, PipelineState.SPEAKING}

    def _preempt_agent_output_for_ptt(self) -> None:
        session = self._session
        if session is not None:
            try:
                fut = session.interrupt(force=True)
                fut.add_done_callback(self._log_interrupt_failure)
            except Exception:
                logger.debug("[HalfDuplexPttPipeline] session interrupt failed", exc_info=True)
        self._publish_client_control(CONTROL_OP_PLAYBACK_STOP, reason="explicit_client_ptt")

    @staticmethod
    def _log_interrupt_failure(done: asyncio.Future[None]) -> None:
        try:
            done.result()
        except Exception:
            logger.debug("[HalfDuplexPttPipeline] session interrupt future failed", exc_info=True)

    def _on_agent_state_changed(self, event: Any) -> None:
        super()._on_agent_state_changed(event)
        new = str(getattr(event, "new_state", "") or "")
        if new:
            self._mark_activity()
        if new == "thinking":
            output_timeline = self._agent_output.active_timeline
            if output_timeline is not None:
                output_timeline.mark("llm_started_at")
            self._publish_companion_ui_state("thinking", "agent_state:thinking")
        elif new == "speaking":
            self._publish_companion_ui_state("speaking", "agent_state:speaking")
        elif new in ("idle", "listening"):
            self._publish_companion_ui_state("listening", f"agent_state:{new}")
            output_timeline = self._agent_output.active_timeline
            if output_timeline is not None and self._output_timeline_started(output_timeline):
                self._flush_output_timeline(
                    output_timeline,
                    f"ptt_agent_state_{new}",
                )

    def _on_session_close(self, event: Any) -> None:
        reason = getattr(event, "reason", None)
        error = getattr(event, "error", None)
        self._close_error = error
        logger.info(
            "[HalfDuplexPttPipeline] session close event reason=%s error=%s",
            reason,
            error,
        )
        self._session_closed_event.set()

    async def _end_serving_on_close(self) -> None:
        on_end = self._on_session_end
        if on_end is not None:
            reason = SESSION_END_ERROR if self._close_error else SESSION_END_USER_LEFT
            try:
                await on_end(reason)
            except Exception:
                logger.exception(
                    "[HalfDuplexPttPipeline] session_end on close failed (reason=%s)",
                    reason,
                )
        cb = self._on_session_closed
        if cb is None:
            return
        try:
            result = cb()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.exception("[HalfDuplexPttPipeline] on_session_closed failed")

    def _publish_ptt_turn_status(
        self,
        outcome: str,
        reason: str,
        *,
        transcript: str = "",
    ) -> None:
        self._publish_client_control(
            CONTROL_OP_PTT_TURN_STATUS,
            reason=reason,
            payload=build_ptt_turn_status_payload(
                outcome,
                reason,
                transcript=transcript,
            ),
        )

    def _publish_companion_ui_state(self, state: str, reason: str) -> None:
        payload = {
            "schema_v": WIRE_SCHEMA_VERSION,
            "type": COMPANION_UI_STATE_TOPIC,
            "state": state,
            "reason": reason,
            "ts_ms": int(time.time() * 1000),
        }
        self._publish_data(COMPANION_UI_STATE_TOPIC, payload)

    def _publish_client_control(
        self,
        op: str,
        *,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        turn_id = self._timeline.turn_id if self._timeline is not None else ""
        envelope = build_session_client_control_envelope(
            op=op,
            reason=reason,
            payload=payload,
            turn_id=turn_id,
        )
        control_payload = envelope["payload"]
        if not self._publish_data(CONTROL_TOPIC, envelope):
            return
        logger.info(
            "[HalfDuplexPttPipeline] queued client control op=%s reason=%s "
            "outcome=%s turn_id=%s topic=%s",
            op,
            reason,
            control_payload.get("outcome"),
            turn_id,
            CONTROL_TOPIC,
        )
        self._record_client_control_event(op=op, reason=reason)

    def _publish_data(self, topic: str, payload: dict[str, object]) -> bool:
        room = getattr(self, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            if topic == CONTROL_TOPIC:
                logger.warning(
                    "[HalfDuplexPttPipeline] skipped client control op=%s "
                    "reason=%s because local participant is unavailable",
                    payload.get("op"),
                    (payload.get("payload") or {}).get("reason")
                    if isinstance(payload.get("payload"), dict)
                    else None,
                )
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if topic == CONTROL_TOPIC:
                logger.warning(
                    "[HalfDuplexPttPipeline] skipped client control op=%s "
                    "reason=%s because no running event loop is available",
                    payload.get("op"),
                    (payload.get("payload") or {}).get("reason")
                    if isinstance(payload.get("payload"), dict)
                    else None,
                )
            return False

        async def _send() -> None:
            await local.publish_data(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=topic,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.warning(
                    "[HalfDuplexPttPipeline] failed to publish data topic=%s op=%s",
                    topic,
                    payload.get("op"),
                    exc_info=True,
                )
                return
            if topic == CONTROL_TOPIC:
                logger.info(
                    "[HalfDuplexPttPipeline] published client control op=%s "
                    "reason=%s outcome=%s topic=%s",
                    payload.get("op"),
                    (payload.get("payload") or {}).get("reason")
                    if isinstance(payload.get("payload"), dict)
                    else None,
                    (payload.get("payload") or {}).get("outcome")
                    if isinstance(payload.get("payload"), dict)
                    else None,
                    topic,
                )

        task.add_done_callback(_log_failure)
        return True

    def _record_client_control_event(self, *, op: str, reason: str) -> None:
        timeline = self._timeline
        turn_id = timeline.turn_id if timeline is not None else ""
        event = build_client_control_event(op=op, reason=reason, turn_id=turn_id)
        if timeline is None:
            if should_drop_pending_ptt_control_event(
                op=op,
                reason=reason,
                turn_id=turn_id,
            ):
                return
            self._pending_client_control_events = append_client_control_event(
                self._pending_client_control_events,
                event,
            )
            return
        events = list(timeline.attrs.get("client_control_events") or ())
        timeline.set_attr(
            "client_control_events",
            append_client_control_event(events, event),
        )

    def _apply_pending_client_control_events(self, timeline: TurnTimeline) -> None:
        if not self._pending_client_control_events:
            return
        events = list(timeline.attrs.get("client_control_events") or ())
        for event in self._pending_client_control_events:
            fixed = dict(event)
            fixed["turn_id"] = timeline.turn_id
            events = append_client_control_event(events, fixed)
        timeline.set_attr("client_control_events", events)
        self._pending_client_control_events = []
