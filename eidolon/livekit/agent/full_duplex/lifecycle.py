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
from ..runtime.resolver import wait_for_runtime_participant_identity
from ..shared.pipeline import BasePipeline
from ..shared.types import generate_turn_id
from .state_machine import FullDuplexPhase

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

        participant_identity = await wait_for_runtime_participant_identity(room)
        pipeline._runtime_participant_identity = participant_identity
        await pipeline._ensure_turn_event_sink().start(room)
        logger.info(
            "[StreamingPipeline] binding RoomIO to runtime participant=%s",
            participant_identity,
        )

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
        session.on("close", self._on_session_close)

        room.on("disconnected", self._on_room_disconnected)

        pipeline._session_signals.register_vad_inference_callback()

        await pipeline._warmup_stages()
        if pipeline._filler is not None:
            await pipeline._filler.warmup()

        logger.info("[StreamingPipeline] calling session.start()...")
        pipeline._ensure_room_data_bridge().install(room)
        pipeline._voiceprint_turns.install(room)

        # Digital-human video avatar (per-session). When enabled, start the avatar
        # worker and route TTS audio to it via DataStreamAudioOutput instead of the
        # room; the framework then leaves our pre-set output.audio in place
        # (agent_session sets room audio_output=False when output.audio is set) and
        # the barge-in duck mixer wraps the DataStream sink unchanged. On failure we
        # fall back to audio-only so a flaky avatar service never drops the call.
        room_audio_output: AudioOutputOptions | bool = AudioOutputOptions(
            sample_rate=pipeline._audio_sample_rate,
        )
        if pipeline._avatar_enabled:
            if await self._start_avatar_worker(room, session):
                room_audio_output = False

        await session.start(
            agent=agent,
            room=room,
            room_options=RoomOptions(
                audio_output=room_audio_output,
                participant_identity=participant_identity,
            ),
        )
        if pipeline._on_session_started is not None:
            await pipeline._on_session_started()
        pipeline._publish_companion_ui_state("listening", "session_started")
        if pipeline._uses_livekit_native_adaptive_interruption():
            logger.info(
                "[StreamingPipeline] LiveKit native adaptive interruption owner "
                "enabled; channel audio-activity patch skipped"
            )
        else:
            framework_patches.disable_audio_activity_interruption(session)

        # Barge-in only: the ducking mixer is a removable interposer over the
        # output sink. half_duplex never barges in, so it is not installed there;
        # normal TTS then flows straight through session.output.audio unchanged.
        if pipeline._barge_in_enabled:
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
            if await self._await_conversation_end():
                logger.info("[StreamingPipeline] session closed event received, exiting run()")
                await self._end_serving_on_close()
            else:
                # Out of the room already: session_end would have nobody to
                # reach, and the serving contract is withdrawn by the job's
                # shutdown callback regardless of how the job got here.
                logger.info("[StreamingPipeline] channel dropped us, exiting run()")
        except asyncio.CancelledError:
            logger.info("[StreamingPipeline] cancelled")
            raise
        finally:
            await self.shutdown()

    async def _start_avatar_worker(self, room: Room, session: Any) -> bool:
        """Start the avatar worker and route session audio to it.

        Returns True if the worker joined and ``session.output.audio`` was set to
        a DataStreamAudioOutput; False (fall back to audio-only) on failure.
        """
        from livekit.agents.voice.avatar import DataStreamAudioOutput

        from eidolon.livekit.avatar import AvatarWorker, resolve_session_face_image

        pipeline = self._pipeline
        cfg = pipeline._avatar_config
        core = pipeline._core_config
        agent_identity = room.local_participant.identity

        # Seed the digital-human service with this companion's configured display
        # face (Ditto ``cond_image``). Best-effort and gated: unconfigured or
        # unresolvable → None → the service's own default avatar, so audio-only
        # and legacy sessions are unaffected.
        face_image: bytes | None = None
        if getattr(cfg, "cond_image_enabled", True):
            runtime_services = getattr(pipeline._factory, "runtime_services", None)
            try:
                face_image = await resolve_session_face_image(
                    room,
                    context_resolver=getattr(
                        pipeline._factory,
                        "runtime_context_resolver",
                        None,
                    ),
                    runtime_client=(
                        runtime_services.runtime if runtime_services is not None else None
                    ),
                )
            except Exception:
                logger.exception(
                    "[StreamingPipeline] avatar face resolution failed; using default avatar"
                )

        worker = AvatarWorker(
            cfg,
            livekit_url=core.livekit_url,
            api_key=core.api_key,
            api_secret=core.api_secret,
            room_name=room.name,
            agent_identity=agent_identity,
            face_image=face_image,
        )
        try:
            avatar_identity = await worker.start()
        except Exception:
            logger.exception("[StreamingPipeline] avatar worker start failed")
            await worker.aclose()
            if not cfg.fallback_to_audio_on_failure:
                raise
            logger.warning("[StreamingPipeline] avatar unavailable; audio-only fallback")
            return False

        session.output.audio = DataStreamAudioOutput(
            room,
            destination_identity=avatar_identity,
            wait_playback_start=True,
            sample_rate=pipeline._audio_sample_rate,
        )
        pipeline._avatar_worker = worker
        logger.info(
            "[StreamingPipeline] avatar routing enabled → %s (audio to worker)",
            avatar_identity,
        )
        return True

    def _on_room_disconnected(self, reason: Any = None) -> None:
        """Being dropped from the channel is the end of the conversation.

        There is nothing left to listen to or speak into, so waiting for the
        framework to reach the same conclusion only prolongs a metered speech
        stream. Distinct from "reconnecting", which LiveKit reports separately —
        this event means the room session is over, not interrupted.
        """
        logger.info("[lifecycle][StreamingPipeline] room disconnected reason=%s", reason)
        self._pipeline._room_disconnected_event.set()

    async def _await_conversation_end(self) -> bool:
        """Wait for whichever ends this conversation first.

        Returns True when the AgentSession closed (the device is still reachable,
        so the close path may still speak to it) and False when the channel
        dropped us (it cannot).
        """
        pipeline = self._pipeline
        session_wait = asyncio.create_task(pipeline._session_closed_event.wait())
        room_wait = asyncio.create_task(pipeline._room_disconnected_event.wait())
        try:
            done, pending = await asyncio.wait(
                {session_wait, room_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            return session_wait in done
        finally:
            for task in (session_wait, room_wait):
                task.cancel()
            await asyncio.gather(session_wait, room_wait, return_exceptions=True)

    def _on_session_close(self, event: Any) -> None:
        """Wake run() so shutdown fires immediately on session close.

        The AgentSession emits this event from its ``_aclose_impl`` finalizer
        (e.g. when ``close_on_disconnect`` triggers after a participant leaves).
        We capture it here and signal ``_session_closed_event``; ``run()`` is
        awaiting that event and will proceed to ``shutdown()``.

        Also clean up the EOT model's per-session UserProfile so long-running
        daemons don't accumulate state across rooms. Defensive: catch and log;
        this must not block the close path.
        """
        pipeline = self._pipeline
        reason = getattr(event, "reason", None)
        error = getattr(event, "error", None)
        # Captured for _end_serving_on_close -> session_end reason: error
        # close -> "error", clean close -> "user_left".
        pipeline._close_reason = reason
        pipeline._close_error = error
        logger.info(
            "[StreamingPipeline] session close event received reason=%s error=%s",
            reason,
            error,
        )
        try:
            pipeline._get_eot_model().end_session()
        except Exception:
            logger.exception("[StreamingPipeline] eot_model.end_session failed (non-fatal)")
        duck_metrics = pipeline._ducking.get_metrics()
        if duck_metrics is not None:
            logger.info(
                "[StreamingPipeline] session duck metrics: %s",
                duck_metrics,
            )
        self._reject_open_user_turn_on_close(error=error)
        pipeline._finish_agent_output(
            "session_error_during_output" if error else "session_closed_during_output"
        )
        pipeline._append_timeline_debug("session_closed")
        pipeline._session_closed_event.set()

    def _reject_open_user_turn_on_close(self, *, error: Any) -> None:
        """Give every product candidate a durable terminal state before teardown."""

        pipeline = self._pipeline
        coordinator = getattr(pipeline, "_user_turns", None)
        candidate = getattr(coordinator, "active", None)
        if candidate is None or getattr(candidate, "state", None) != "open":
            return
        reason = "session_error" if error else "session_closed"
        decision = coordinator.reject_active(reason)
        if decision.action != "reject":
            return
        timeline = candidate.timeline
        pipeline._record_full_duplex_transition(
            FullDuplexPhase.USER_TURN_REJECTED,
            event="session_closed_with_open_user_turn",
            reason=reason,
            side_effect="irreversible",
            timeline=timeline,
        )
        pipeline._flush_turn_timeline(timeline, reason)

    async def _end_serving_on_close(self) -> None:
        """Give up this conversation promptly, before the provider shutdown drain.

        What "giving up" costs the device is the caller's business, not ours —
        the channel outlives the conversation, so this hook announces the end
        and hands teardown to whoever owns the serving contract.
        """
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
            logger.exception("[StreamingPipeline] on_session_closed (prompt teardown) failed")

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
        pipeline._ensure_turn_completion().reset_candidate_voiceprint_tasks()
        pipeline._ensure_turn_completion().cancel_completed_voiceprint_turn()
        if hasattr(pipeline, "_provider_events"):
            pipeline._provider_events.cancel_output_watchdog()
        pipeline._ducking.cancel_timeout()
        pipeline._stop_idle_watchdog()
        avatar_worker = getattr(pipeline, "_avatar_worker", None)
        if avatar_worker is not None:
            try:
                await avatar_worker.aclose()
            except Exception:
                logger.exception("[StreamingPipeline] error closing avatar worker")
            pipeline._avatar_worker = None
        if hasattr(pipeline, "_voiceprint_turns"):
            await pipeline._voiceprint_turns.aclose()
        if pipeline._session is not None:
            try:
                await pipeline._session.aclose()
            except Exception:
                logger.exception("[StreamingPipeline] error shutting down session")
            pipeline._session = None
        pipeline._finish_agent_output("session_shutdown_during_output")
        pipeline._append_timeline_debug("session_shutdown")
        await pipeline._ensure_turn_event_sink().close(
            reason="session_error" if getattr(pipeline, "_close_error", None) else "session_ended"
        )
        await pipeline._shutdown_stages()
        close_factory = getattr(pipeline._factory, "aclose", None)
        if callable(close_factory):
            await close_factory()
        await BasePipeline.shutdown(pipeline)
