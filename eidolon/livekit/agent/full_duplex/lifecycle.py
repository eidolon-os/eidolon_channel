"""Full-duplex AgentSession run/start/shutdown lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from eidolon_sdk.biz.contracts import (
    SESSION_END_ERROR,
    SESSION_END_USER_LEFT,
)

from ..integration import framework_patches
from ..pipeline.base import BasePipeline
from ..pipeline.types import generate_turn_id

if TYPE_CHECKING:
    from livekit.rtc import Room

    from .pipeline import StreamingPipeline

logger = logging.getLogger("agent")


class FullDuplexSessionLifecycle:
    """Own AgentSession startup, background consumers, and shutdown ordering."""

    def __init__(self, pipeline: StreamingPipeline) -> None:
        self._pipeline = pipeline

    async def run(self, room: Room) -> None:
        """Start the full-duplex pipeline and block until the session closes."""
        from livekit.agents.voice import AgentSession
        from livekit.agents.voice.room_io import AudioOutputOptions, RoomOptions

        pipeline = self._pipeline
        logger.info("[StreamingPipeline] starting room=%s", room.name)
        pipeline._room = room
        pipeline._started = True

        agent = pipeline._build_agent()
        session = AgentSession(
            turn_handling=pipeline._build_turn_handling(),
            aec_warmup_duration=pipeline._aec_warmup_duration,
        )
        pipeline._session = session

        session.on("user_state_changed", pipeline._on_user_state_changed)
        session.on("agent_state_changed", pipeline._on_agent_state_changed)
        session.on("user_input_transcribed", pipeline._on_user_transcribed)
        session.on("error", pipeline._on_session_error)
        session.on("close", pipeline._on_session_close)

        pipeline._session_signals.register_vad_inference_callback()

        await pipeline._warmup_stages()
        if pipeline._filler is not None:
            await pipeline._filler.warmup()

        logger.info("[StreamingPipeline] calling session.start()...")
        pipeline._ensure_room_data_bridge().install(room)
        pipeline._voiceprint_turns.install(room)
        await session.start(
            agent=agent,
            room=room,
            room_options=RoomOptions(
                audio_output=AudioOutputOptions(
                    sample_rate=pipeline._audio_sample_rate,
                ),
            ),
        )
        pipeline._publish_companion_ui_state("listening", "session_started")
        if pipeline._uses_livekit_native_adaptive_interruption():
            logger.info(
                "[StreamingPipeline] LiveKit native adaptive interruption owner "
                "enabled; channel audio-activity patch skipped"
            )
        else:
            framework_patches.disable_audio_activity_interruption(session)

        pipeline._ensure_output_flow().install_duck_mixer(session)
        if pipeline._filler is not None and session.output.audio is not None:
            target_sr = session.output.audio.sample_rate
            logger.info(
                "[StreamingPipeline] preparing filler clips for output @ %d Hz",
                target_sr,
            )
            pipeline._filler.prepare_for_output(target_sr)

        pipeline._get_eot_model().start_session(room.name or generate_turn_id())
        logger.info("[StreamingPipeline] session started")

        pipeline._start_idle_watchdog()
        self._start_proactive_consumer()

        try:
            await pipeline._session_closed_event.wait()
            logger.info("[StreamingPipeline] session closed event received, exiting run()")
            await self._delete_room_on_close()
        except asyncio.CancelledError:
            logger.info("[StreamingPipeline] cancelled")
            raise
        finally:
            await self.shutdown()

    async def _delete_room_on_close(self) -> None:
        """Prompt room teardown on session close before provider shutdown drain."""
        pipeline = self._pipeline
        on_end = getattr(pipeline, "_on_session_end", None)
        if on_end is not None:
            reason = (
                SESSION_END_ERROR
                if getattr(pipeline, "_close_error", None)
                else SESSION_END_USER_LEFT
            )
            try:
                await on_end(reason)
            except Exception:
                logger.exception(
                    "[StreamingPipeline] session_end on close failed (reason=%s)",
                    reason,
                )
        cb = getattr(pipeline, "_on_session_closed", None)
        if cb is None:
            return
        try:
            await cb()
        except Exception:
            logger.exception("[StreamingPipeline] on_session_closed (prompt room delete) failed")

    def _start_proactive_consumer(self) -> None:
        """Spawn the background proactive-report stream (best-effort)."""
        pipeline = self._pipeline
        if pipeline._proactive_task is not None and not pipeline._proactive_task.done():
            return
        pipeline._proactive_task = asyncio.create_task(
            self._run_proactive_consumer(),
            name="eidolon-proactive-consumer",
        )

    async def _run_proactive_consumer(self) -> None:
        """Open the proactive stream against the brain and keep it running."""
        pipeline = self._pipeline
        llm_plugin = getattr(getattr(pipeline._factory, "llm", None), "llm", None)
        opener = getattr(llm_plugin, "open_proactive_subscriber", None)
        if opener is None:
            logger.info(
                "[StreamingPipeline] proactive consumer disabled "
                "(LLM backend has no proactive stream)"
            )
            return
        try:
            subscriber = await opener(on_event=self._on_proactive_report)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[StreamingPipeline] failed to open proactive stream")
            return
        pipeline._proactive_subscriber = subscriber
        try:
            await subscriber.run()
        except asyncio.CancelledError:
            raise
        finally:
            await subscriber.aclose()
            pipeline._proactive_subscriber = None

    async def _on_proactive_report(self, report: Any) -> None:
        """Speak a proactive brain report into the room via TTS."""
        pipeline = self._pipeline
        text = (getattr(report, "text", "") or "").strip()
        if not text:
            return
        session = pipeline._session
        if session is None:
            logger.info(
                "[StreamingPipeline] dropping proactive report (session closed) intent=%s",
                getattr(report, "intent", ""),
            )
            return
        logger.info(
            "[StreamingPipeline] proactive report intent=%s chars=%d - speaking",
            getattr(report, "intent", ""),
            len(text),
        )
        pipeline._mark_activity()
        session.say(text, allow_interruptions=True)

    def _stop_proactive_consumer(self) -> None:
        pipeline = self._pipeline
        task = pipeline._proactive_task
        if task is not None and not task.done():
            task.cancel()
        pipeline._proactive_task = None

    async def shutdown(self) -> None:
        """Gracefully shut down the full-duplex session."""
        pipeline = self._pipeline
        logger.info("[StreamingPipeline] shutting down")
        self._stop_proactive_consumer()
        if hasattr(pipeline, "_interruption_effects"):
            pipeline._interruption_effects.cancel_soft_interrupt()
            pipeline._interruption_effects.cancel_stable_signal_timer()
        pipeline._ensure_turn_completion().cancel_pending_voiceprint_commits("shutdown")
        if hasattr(pipeline, "_provider_events"):
            pipeline._provider_events.cancel_output_watchdog()
        pipeline._ducking.cancel_timeout()
        pipeline._stop_idle_watchdog()
        if hasattr(pipeline, "_voiceprint_turns"):
            await pipeline._voiceprint_turns.aclose()
        if pipeline._session is not None:
            try:
                await pipeline._session.aclose()
            except Exception:
                logger.exception("[StreamingPipeline] error shutting down session")
            pipeline._session = None
        await pipeline._shutdown_stages()
        await BasePipeline.shutdown(pipeline)
