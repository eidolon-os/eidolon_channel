"""StreamingPipeline — STREAMING mode via AgentSession + ChineseModel EOT.

============================================================================
ARCHITECTURE DECISION (ADR — boundary between this layer and AgentSession)
============================================================================

Most orchestration is delegated to ``livekit.agents.AgentSession``:
turn detection, endpointing, interruption staging, preemptive
generation, plugin lifecycle, room I/O, event emission. This class
IS NOT a replacement for AgentSession — it's a thin wrapper that:

  1. **Configures AgentSession** with our turn_handling /
     interruption / preemptive options (R8.9.b: must be on
     AgentSession constructor, not Agent constructor).

  2. **Manages plugin warmup/shutdown** that AgentSession doesn't
     own (see ``agent/pipeline/base.py`` ADR).

  3. **Bridges framework events to our extensions** —
     ``user_state_changed`` → STT user_away signal,
     ``agent_state_changed`` → soft-interrupt timer cancel,
     VAD inference probabilities → EOT state (R7 G6, framework
     doesn't expose this).

  4. **Implements soft/hard interrupt staging** — the framework has
     ``false_interruption_timeout`` for similar intent, but we need
     a state machine that can be cancelled by silence (the user
     paused, didn't actually interrupt) AND that integrates with
     our PolicyChain decisions. Our soft-interrupt is in addition
     to (not replacing) framework's.

  5. **Forces framework's auto-interrupt OFF** via
     ``_framework_patches.disable_audio_activity_interruption`` —
     so EOT PolicyChain is the sole authority. See
     ``_framework_patches.py`` for the rationale (no public API
     does this without breaking endpointing).

If you find yourself adding logic here that AgentSession already
handles, push back — likely the right shape is to USE the
AgentSession config rather than override.
============================================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents.voice import Agent as lk_Agent
    from livekit.agents.voice import AgentSession
    from livekit.rtc import Room

from . import _framework_patches
from .ducking import DuckingMixer
from .factory import SharedStageFactory
from .filler import FillerManager
from .pipeline.base import BasePipeline
from .pipeline.types import PipelineCallbacks, PipelineState, generate_turn_id

logger = logging.getLogger("agent")


# Module-level cache for the EOT model singleton.
# All StreamingPipeline instances share the same ChineseModel instance, which in turn
# shares the same EotManager singleton (and thus the same ONNX session).
_eot_model_cache: Any = None


def _get_shared_eot_model() -> Any:
    """Lazily create and cache the shared ChineseModel instance.

    The EotManager inside ChineseModel is a thread-safe singleton that holds
    the ONNX session, so all callers share the same model weights in memory.
    """
    global _eot_model_cache
    if _eot_model_cache is None:
        from eidolon.livekit.plugins.eot import ChineseModel

        logger.info("[StreamingPipeline] loading EOT model...")
        _eot_model_cache = ChineseModel()
        logger.info("[StreamingPipeline] EOT model loaded")
    return _eot_model_cache


class StreamingPipeline(BasePipeline):
    """
    STREAMING mode: real-time audio via LiveKit Room.

    Audio flows from Room participant stream -> VAD -> STT (AgentSession auto) ->
    ChineseModel EOT -> LLM -> TTS. Full AgentSession orchestration.

    Usage::

        factory = SharedStageFactory(config)
        pipeline = StreamingPipeline(
            factory,
            instructions="You are a helpful voice assistant.",
        )
        await pipeline.run(room)
    """

    def __init__(
        self,
        factory: SharedStageFactory,
        *,
        instructions: str = "",
        callbacks: PipelineCallbacks | None = None,
        allow_interruptions: bool = True,
        welcome_message: str = "",
        false_interruption_timeout: float | None = 6.0,
        audio_sample_rate: int = 16000,
        stt_commit_transcript_timeout: float = 5.0,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        self._instructions = instructions
        self._allow_interruptions = allow_interruptions
        # Round 8 R8.9: fixed welcome (instead of LLM-generated). LLM with
        # only a system prompt context tends to echo back instruction
        # templates, which the user heard as garbled "welcome".
        self._welcome_message = welcome_message
        # Round 8 R8.9: framework default 2.0s is too short for Chinese
        # STT (SenseAudio) which often takes 3-5s to deliver a final
        # transcript. The framework misclassifies real interrupts as
        # false and resumes the agent's speech mid-utterance. 6.0s gives
        # STT enough headroom.
        self._false_interruption_timeout = false_interruption_timeout
        self._audio_sample_rate = audio_sample_rate
        # F1 fix (2026-05-16): pass to session.commit_user_turn() so STT FINAL
        # has enough time to arrive before framework promotes the latest INTERIM
        # to a FINAL (which causes a doomed LLM call + cancel).
        self._stt_commit_transcript_timeout = stt_commit_transcript_timeout

        self._session: AgentSession | None = None
        # Set when AgentSession emits "close" event (e.g. participant disconnect).
        # run() awaits this instead of polling room.isconnected, so shutdown
        # fires within milliseconds of the framework deciding to close.
        self._session_closed_event: asyncio.Event = asyncio.Event()

        # EOT semantic interruption check state
        self._user_speaking_start_time: float | None = None
        # ``self._user_vad_active`` was removed in Round 8 cleanup —
        # framework's ``session.user_state == "speaking"`` is the
        # authoritative source. Reading our own mirror introduced
        # a race-prone edge case (user_state speaking → away skipping
        # listening would leave the mirror stale).
        self._latest_asr_text: str = ""

        # Two-stage interruption state: soft interrupt waits for confirmation before
        # hard-cutting. If user falls silent (VAD inactive) during the wait, the
        # interrupt is cancelled (false interruption). If the timeout fires, we upgrade
        # to a hard interrupt.
        #
        # NOTE: when ``EidolonEOTConfig.duck_enabled`` is True (default), the
        # primary interrupt mechanism is the DuckingMixer + early-resume
        # watcher (see ``_install_duck_mixer`` and ``_run_eot_check`` below).
        # The soft-interrupt path here is kept as a fallback for the rare
        # cases where EOT signals a cut but the mixer isn't installed
        # (e.g. duck_enabled=False, or audio output sink not yet attached).
        self._soft_interrupt_active: bool = False
        self._soft_interrupt_timer: asyncio.Task | None = None

        # Read timeout from EOT model config (can be overridden per-pipeline via arg).
        self._soft_interrupt_timeout: float = self._get_eot_model().soft_interrupt_timeout

        # DuckingMixer + suspend-window timeout fallback task. Both lazy.
        # Mixer is installed in ``run()`` after ``session.start()`` returns,
        # so the AudioOutput chain is fully assembled. The timeout task is
        # started on every VAD-start (user_state listening → speaking) and
        # cancelled the moment EOT decides to cancel/unduck.
        self._duck_mixer: DuckingMixer | None = None
        self._duck_timeout_task: asyncio.Task | None = None
        self._last_unduck_time: float = 0.0
        self._duck_suspend_start: float = 0.0

        # Filler word injection for latency masking.
        eot_cfg = self._get_eot_model()._config
        self._filler: FillerManager | None = None
        if eot_cfg.filler_enabled:
            self._filler = FillerManager(
                self._factory.tts,
                phrases=list(eot_cfg.filler_phrases),
            )

        # Interrupted content tracking — snapshot of agent text at
        # the moment of confirmed interrupt, injected as context into
        # the next LLM turn so the model can optionally reference it.
        self._last_interrupted_context: dict[str, Any] | None = None

        # Eagerly trigger EOT model loading so the ONNX session is ready before
        # the first user audio frame arrives. This avoids cold-start delay after
        # session.start() is called.
        _get_shared_eot_model()

    def _get_eot_model(self) -> Any:
        """Return the shared EOT model instance."""
        return _get_shared_eot_model()

    async def run(self, room: Room) -> None:
        """Start the streaming pipeline. Blocks until room disconnects."""
        import livekit.agents as la
        from livekit.agents.voice import Agent, AgentSession

        logger.info("[StreamingPipeline] starting room=%s", room.name)
        self._room = room
        self._started = True

        agent = self._build_agent()

        # Round 8 R8.9 (re-fix): the ``turn_handling`` config — including
        # ``false_interruption_timeout`` — must be passed to AGENTSESSION,
        # not to Agent. The framework's ``AgentActivity`` reads
        # ``session._opts.turn_handling.interruption`` from the
        # AgentSession-level options. The previous attempt put this on
        # Agent.turn_handling and it was silently ignored — production
        # logs showed the default 2.0s timeout still firing instead of
        # our configured 6.0s. Source of truth: ``agent_session.py:354
        # _resolve_interruption(turn_handling.get("interruption"))``.
        session = AgentSession(
            turn_handling={
                "interruption": {
                    "enabled": self._allow_interruptions,
                    "discard_audio_if_uninterruptible": True,
                    "false_interruption_timeout": self._false_interruption_timeout,
                },
                # Round 8 R8.12.c: disable preemptive generation entirely.
                # Production logs (2026-05-07) showed:
                #     "preemptive generation enabled but chat context or
                #      tools have changed after on_user_turn_completed"
                # The framework was firing LLM calls based on STT interim
                # transcripts; when later STT chunks changed the user's
                # message, the preemptive call was wasted (and sometimes
                # its TTS partial leaked through). For Chinese workloads
                # with frequent server-VAD-driven segmentation (R8.12.a),
                # context-changed is the common case, not the exception.
                # Trade-off: ~500-1000ms slower first audio, far more
                # consistent state.
                "preemptive_generation": {
                    "enabled": False,
                    "preemptive_tts": False,  # belt + suspenders
                },
            },
        )
        self._session = session

        session.on("user_state_changed", self._on_user_state_changed)
        session.on("agent_state_changed", self._on_agent_state_changed)
        session.on("user_input_transcribed", self._on_user_transcribed)
        session.on("error", self._on_session_error)
        session.on("close", self._on_session_close)

        # Round 7 G6: bridge per-frame VAD probability into EOT state. The
        # framework's AgentSession does NOT re-emit INFERENCE_DONE events
        # externally, so we register directly on the FireRed VAD's per-frame
        # callback hook. Policies can then read
        # state.recent_avg_vad_confidence() for confidence-gated decisions.
        self._register_vad_inference_callback()

        # Warm up persistent-connection stages (STT, TTS) before starting the
        # session. Stages without a warmup() are silently skipped.
        await self._warmup_stages()
        if self._filler is not None:
            await self._filler.warmup()

        logger.info("[StreamingPipeline] calling session.start()...")
        await session.start(
            agent=agent,
            room=room,
            room_input_options=la.RoomInputOptions(),
            room_output_options=la.RoomOutputOptions(
                transcription_enabled=True,
                audio_sample_rate=self._audio_sample_rate,
            ),
        )
        # Disable framework's built-in audio-activity auto-interrupt so EOT
        # PolicyChain (and the DuckingMixer below) is the sole authority on
        # interrupt decisions. See _framework_patches.disable_audio_activity_interruption
        # for the full rationale (no public API alternative — internal flags must be
        # patched). The patch sets BOTH the runtime flag AND the default-
        # value flag, so framework's restore logic on agent state transitions
        # doesn't undo us. No re-patch needed in _on_agent_state_changed.
        _framework_patches.disable_audio_activity_interruption(session)

        # Install the DuckingMixer between TTS frames and the RoomIO sink.
        # Must run AFTER session.start() because that's when the framework
        # assembles ``session.output.audio`` (RoomIO + TranscriptSynchronizer).
        # See ``_install_duck_mixer`` for state-machine details.
        self._install_duck_mixer(session)

        # Now that the audio output chain is assembled, resample and
        # envelope cached filler clips to match the chain's sample rate.
        # This must run AFTER _install_duck_mixer so session.output.audio
        # is in its final form (DuckingMixer→TranscriptSync→RoomIO).
        if self._filler is not None and session.output.audio is not None:
            target_sr = session.output.audio.sample_rate
            logger.info(
                "[StreamingPipeline] preparing filler clips for output @ %d Hz",
                target_sr,
            )
            self._filler.prepare_for_output(target_sr)

        # Initialize EOT session context for multi-turn awareness.
        self._get_eot_model().start_session(room.name or generate_turn_id())
        logger.info("[StreamingPipeline] session started")

        try:
            # Wait for AgentSession to close (e.g. participant disconnect →
            # framework auto-closes session via close_on_disconnect=True).
            # This replaces the old `while room.isconnected:` polling, which
            # didn't react to session-level close in time and required the
            # 30s entrypoint watchdog to force shutdown — leaving TTS
            # connections open and heartbeats firing for tens of seconds.
            await self._session_closed_event.wait()
            logger.info(
                "[StreamingPipeline] session closed event received, exiting run()"
            )
        except asyncio.CancelledError:
            logger.info("[StreamingPipeline] cancelled")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down the session."""
        logger.info("[StreamingPipeline] shutting down")
        # Cancel any pending soft interrupt / duck timeout before closing.
        self._cancel_soft_interrupt()
        self._cancel_duck_timeout()
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:
                logger.exception("[StreamingPipeline] error shutting down session")
            self._session = None
        # Tear down persistent-connection stages (STT, TTS, etc.).
        await self._shutdown_stages()
        await super().shutdown()

    def _build_agent(self) -> lk_Agent:
        """Build the LiveKit Agent."""
        from livekit.agents.voice import Agent

        # Bind welcome to a closure so the inner Agent class can read it
        # without us touching its constructor signature.
        welcome_message = self._welcome_message

        class VoiceAgent(Agent):
            async def on_enter(self) -> None:
                logger.info("[VoiceAgent] on_enter welcome=%r",
                            welcome_message[:30] if welcome_message else "")
                # Round 8 R8.9: use ``session.say(welcome)`` instead of
                # ``session.generate_reply()`` for the initial greeting.
                # generate_reply with no user message hands an empty
                # context to the LLM, which then frequently echoes the
                # system prompt template back as the "welcome". Fixed text
                # is faster (no LLM call), more deterministic, and avoids
                # leaking instruction text to users.
                if welcome_message:
                    self.session.say(welcome_message, allow_interruptions=True)
                # else: silent welcome — agent waits for user to speak first

        # Round 8 R8.9 (re-fix): turn_handling config (including
        # false_interruption_timeout, preemptive_generation) lives on
        # AGENTSESSION, not Agent. Putting it here was silently ignored.
        # Agent only carries per-agent override of ``turn_detection`` (the
        # EOT model instance, which is per-agent semantic).
        return VoiceAgent(
            instructions=self._instructions,
            stt=self._factory.stt.stt,
            llm=self._factory.llm.llm,
            tts=self._factory.tts.tts,
            vad=self._factory.vad.vad if self._factory.vad else None,
            turn_detection=self._get_eot_model(),
        )

    def _on_session_close(self, event: Any) -> None:
        """Wake run() so shutdown fires immediately on session close.

        The AgentSession emits this event from its ``_aclose_impl`` finalizer
        (e.g. when ``close_on_disconnect`` triggers after a participant leaves).
        We capture it here and signal ``_session_closed_event``; ``run()`` is
        awaiting that event and will proceed to ``shutdown()``.

        Round 8 P2.L8: also clean up the EOT model's per-session
        UserProfile so long-running daemons don't accumulate state
        across rooms. Defensive: catch and log — must not block the
        close path.
        """
        reason = getattr(event, "reason", None)
        error = getattr(event, "error", None)
        logger.info(
            "[StreamingPipeline] session close event received reason=%s error=%s",
            reason, error,
        )
        try:
            self._get_eot_model().end_session()
        except Exception:
            logger.exception(
                "[StreamingPipeline] eot_model.end_session failed (non-fatal)"
            )
        if self._duck_mixer is not None:
            metrics = self._duck_mixer.get_metrics()
            logger.info(
                "[StreamingPipeline] session duck metrics: %s", metrics,
            )
        self._session_closed_event.set()

    def _on_agent_state_changed(self, event: Any) -> None:
        """Forward agent state change + cancel pending soft interrupts.

        Round 8 cleanup: previously this also re-applied
        ``disable_audio_activity_interruption`` on every agent
        speech-start, because we only patched the runtime flag and
        framework restored it on each transition. Now ``_framework_patches``
        also patches the *default* flag, so the disable is permanent for
        the session and we no longer need this hot path.
        """
        super()._on_agent_state_changed(event)

        # If agent starts thinking or speaking (e.g. after a new user transcript), cancel
        # any pending soft interrupt timer. The timer was set for the PREVIOUS turn's
        # interrupt; it must not fire and cancel the NEW agent response.
        if event.new_state in ("thinking", "speaking"):
            if self._filler is not None:
                self._filler.cancel()
        if event.new_state in ("thinking", "speaking") and self._soft_interrupt_active:
            logger.info(
                "[StreamingPipeline] agent started %s, cancelling pending soft interrupt timer",
                event.new_state,
            )
            self._cancel_soft_interrupt()

    def _register_vad_inference_callback(self) -> None:
        """Round 7 G6: bridge per-frame VAD probability to EOT state.

        Wires ``FireredPvadVAD.register_inference_callback`` (only available
        on the FireRed plugin — duck-typed for compatibility) so each
        ``INFERENCE_DONE`` window updates ``state.probability_samples``.
        Policies can then call ``state.recent_avg_vad_confidence()`` to
        gate cuts on confidence.

        No-op if VAD is None or doesn't expose the callback hook (e.g.
        Silero VAD plugin doesn't have it).

        The callback is invoked from the VAD inference thread; we keep it
        dead-simple (just push a sample into a deque) — no awaits, no I/O.
        """
        try:
            vad_stage = self._factory.vad
            raw_vad = vad_stage.vad if vad_stage else None
            if raw_vad is None:
                return
            if not hasattr(raw_vad, "register_inference_callback"):
                return  # e.g. Silero plugin

            eot_model = self._get_eot_model()

            def _on_inference(probability: float, speaking: bool) -> None:
                # Inference thread → keep this lightweight.
                try:
                    eot_model.update_vad_probability(probability)
                except Exception:
                    # Don't let pipeline state errors stall VAD inference.
                    pass

            raw_vad.register_inference_callback(_on_inference)
            logger.info(
                "[StreamingPipeline] VAD inference callback registered "
                "(per-frame probability → EOT state)"
            )
        except Exception:
            logger.exception(
                "[StreamingPipeline] failed to register VAD inference callback"
            )

    def _signal_stt_user_away(self) -> None:
        """Forward user_state -> "away" to STT plugin if it supports it.

        Part of the Round 7 G11 state-sync bridge: lets STT plugins with
        active streams (e.g. SenseTimeSTT) abort immediately when the
        framework decides the user is gone, instead of waiting for the
        per-stream 30 s safety net to fire.

        Uses ``hasattr`` ducktyping so STT plugins without this hook
        (e.g. Bailian) are safely no-op'd.
        """
        try:
            stt = self._factory.stt._stt
            if hasattr(stt, "signal_user_away"):
                stt.signal_user_away()
        except Exception:
            logger.exception(
                "[StreamingPipeline] error signaling user_away to STT"
            )

    def _signal_stt_user_present(self) -> None:
        """Reverse of :meth:`_signal_stt_user_away` — user_state returned
        from "away" to "listening" / "speaking".
        """
        try:
            stt = self._factory.stt._stt
            if hasattr(stt, "signal_user_present"):
                stt.signal_user_present()
        except Exception:
            logger.exception(
                "[StreamingPipeline] error signaling user_present to STT"
            )

    def _on_user_state_changed(self, event: Any) -> None:
        try:
            old = event.old_state
            new = event.new_state
            logger.info("[StreamingPipeline] user_state: %s -> %s", old, new)

            # ------------------------------------------------------------
            # State-sync bridge (Round 7 G11)
            #
            # Translate framework's high-level user_state transitions into
            # plugin-level signals so plugins can react to authoritative
            # decisions without waiting for their per-stream timeouts.
            #
            # Most important case: user_state -> "away" means the framework
            # has determined no real user activity. Any in-flight STT stream
            # is processing noise/echo, so signal it to abort immediately
            # rather than burn 30 s of bandwidth + compute on the safety net.
            # ------------------------------------------------------------
            if new == "away":
                self._signal_stt_user_away()
            elif old == "away" and new in ("listening", "speaking"):
                self._signal_stt_user_present()

            if new == "speaking":
                self._callbacks.on_user_started_speaking()
                self._user_speaking_start_time = time.time()
                # Immediately clear stale text so EOT only sees text from THIS speech turn.
                self._latest_asr_text = ""

                # Feed VAD signal into EOT model so VADState reflects user activity.
                self._get_eot_model().update_vad(True)

                # Phase C: immediately fade agent output to silence and arm
                # the suspend-window fallback. EOT decisions in
                # ``_run_eot_check`` will resolve us out of SUSPENDED before
                # the timeout fires in the typical case.
                self._duck_and_arm_timeout()

                # If agent is speaking and interruptions are allowed, EOT check is
                # triggered synchronously in _on_user_transcribed as soon as STT
                # delivers the first transcript (INTERIM or FINAL) — no polling needed.

            elif old == "speaking" and new == "listening":
                self._user_speaking_start_time = None

                # Feed VAD silence signal into EOT model.
                eot_model = self._get_eot_model()
                eot_model.update_vad(False)

                # If a soft interrupt was pending and the user fell silent, this is a
                # false interruption — cancel the pending hard interrupt.
                if self._soft_interrupt_active:
                    logger.info(
                        "[StreamingPipeline] user fell silent during soft interrupt → "
                        "false interruption, cancelling"
                    )
                    self._cancel_soft_interrupt()

                # Phase C: if we're still SUSPENDED, the user finished
                # speaking without producing a strong-enough interrupt
                # signal — confirmed false interruption, resume agent
                # output smoothly.
                self._duck_unduck_if_suspended()

                self._callbacks.on_user_ended_speaking()
                if self._session is not None:
                    # Record turn BEFORE reset so dialogue history captures this turn.
                    if self._latest_asr_text:
                        eot_model.record_turn(
                            self._latest_asr_text,
                            is_complete=True,
                            eot_score=eot_model._current_eot_score,
                        )
                    # Reset per-turn state, then commit — this ordering ensures
                    # the framework sees clean state if it calls predict_end_of_turn.
                    eot_model.reset()
                    self._inject_interrupted_context()
                    # F1 fix (2026-05-16): framework default is 2.0s, too short
                    # for Bailian FunASR FINAL on long Chinese sentences. Pass our
                    # configured timeout so framework gives STT enough headroom
                    # before promoting INTERIM→FINAL and firing a doomed LLM call.
                    self._session.commit_user_turn(
                        transcript_timeout=self._stt_commit_transcript_timeout,
                    )
                    if (
                        self._filler is not None
                        and self._session.output.audio is not None
                    ):
                        self._filler.inject(self._session.output.audio)

                self._latest_asr_text = ""

        except Exception:
            logger.exception("[StreamingPipeline] error in _on_user_state_changed")

    def _on_user_transcribed(self, event: Any) -> None:
        """Handle user transcription events.

        Two responsibilities:
          1. Refresh EOT state (ASR text + conversation phase) on every
             transcript event — this is what activates Round 7 G5
             phase-aware threshold scaling. ``update_asr`` is wired
             here (Round 8 R8.5.c); without this call the phase detector
             never runs in production.
          2. While the agent is speaking, trigger the semantic EOT
             check synchronously (event-driven, replacing the old
             polling approach).
        """
        if event.transcript:
            self._latest_asr_text = event.transcript
            # Round 8 R8.5.c: drive phase tracking + ONNX-debounced
            # scoring on every ASR event (interim + final). The 200ms
            # debounce inside update_asr coexists with EotManager's 50ms
            # cache — both contribute to keeping CPU bounded under the
            # ~100ms FunASR interim cadence.
            try:
                self._get_eot_model().update_asr(
                    event.transcript, is_final=event.is_final
                )
            except Exception:
                logger.exception(
                    "[StreamingPipeline] eot_model.update_asr failed"
                )

        # Event-driven EOT check: react immediately when STT delivers text,
        # instead of polling for it. This ensures we analyze the CURRENT
        # speech turn's text, not a stale one from a previous turn.
        agent_is_speaking = self._state == PipelineState.SPEAKING
        if agent_is_speaking and self._allow_interruptions and event.transcript:
            self._run_eot_check(event.transcript, is_final=event.is_final)

        super()._on_user_transcribed(event)

    def _run_eot_check(self, text: str, is_final: bool = False) -> None:
        """
        Synchronous EOT semantic check triggered by each STT transcript event.

        Runs immediately on the text received from STT, without polling or delay.
        Two-stage flow:
        - Strong intent  → hard interrupt immediately
        - EOT says cut   → enter soft interrupt (wait for confirmation)
        - User silent    → cancel soft interrupt (false interruption)
        - Timeout        → upgrade to hard interrupt
        """
        if not text or not text.strip():
            return

        eot_model = self._get_eot_model()

        cfg = eot_model._config

        # Phase C duck-mixer path takes priority when a mixer is installed
        # AND we're in the suspend window (mixer state == SUSPENDED). The
        # mixer was armed on user_state listening→speaking; we now resolve
        # it based on EOT signal.
        duck_active = (
            self._duck_mixer is not None
            and self._duck_mixer.state == "SUSPENDED"
        )

        # Strong interrupt intent — confirmed cancel (real interrupt).
        if eot_model._turn_end_policy.is_strong_interrupt_intent(text):
            logger.info(
                "[StreamingPipeline] EOT: strong interrupt intent, text=%r",
                text[:50],
            )
            if duck_active:
                self._duck_cancel_and_interrupt()
            else:
                self._interrupt_current_turn()
            return

        # Compute score (always, for both duck-active and fallback paths).
        # vad_active reads from framework's authoritative user_state.
        vad_active = (
            self._session is not None
            and self._session.user_state == "speaking"
        )
        should_cut = eot_model.should_interrupt(
            text, vad_active=vad_active, is_final=is_final,
        )
        score = eot_model.current_eot_score

        # ──────────────────────────────────────────────────────────
        # Duck-active path: resolve SUSPENDED state based on score
        # ──────────────────────────────────────────────────────────
        if duck_active:
            # High score — confirm cancel (real interrupt).
            if score >= cfg.duck_early_cancel_score_threshold:
                suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
                logger.info(
                    "[StreamingPipeline] EOT(duck): score=%.2f ≥ %.2f → cancel "
                    "(suspend_ms=%.0f). text=%r",
                    score, cfg.duck_early_cancel_score_threshold,
                    suspend_ms, text[:80],
                )
                self._duck_cancel_and_interrupt()
                return
            # Low score — confirm false interrupt, resume.
            # Note: only treat 0 < score ≤ threshold as positively-low.
            # A score of exactly 0 means "no signal yet" (text too short
            # for ONNX inference), in which case we wait for more interim.
            if 0 < score <= cfg.duck_early_resume_score_threshold:
                suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
                logger.info(
                    "[StreamingPipeline] EOT(duck): score=%.2f ≤ %.2f → unduck "
                    "(false, suspend_ms=%.0f). text=%r",
                    score, cfg.duck_early_resume_score_threshold,
                    suspend_ms, text[:80],
                )
                self._duck_unduck_if_suspended(reason="eot_low_score")
                return
            # Mid-band — leave SUSPENDED; either next interim resolves
            # us or the suspend timeout fallback unducks.
            logger.info(
                "[StreamingPipeline] EOT(duck): score=%.2f mid-band, holding SUSPENDED. text=%r",
                score, text[:80],
            )
            return

        # ──────────────────────────────────────────────────────────
        # Fallback path (mixer not installed, or duck disabled):
        # the legacy two-stage soft/hard interrupt machinery.
        # ──────────────────────────────────────────────────────────

        # Already in soft interrupt: stay in the waiting state until timeout or silence.
        if self._soft_interrupt_active:
            logger.info(
                "[StreamingPipeline] EOT: already in soft interrupt, waiting. text=%r",
                text[:80],
            )
            return

        if should_cut:
            if score >= eot_model.hard_interrupt_score_threshold:
                logger.info(
                    "[StreamingPipeline] EOT: score=%.2f ≥ %.2f → hard interrupt "
                    "(skip soft stage). text=%r",
                    score,
                    eot_model.hard_interrupt_score_threshold,
                    text[:80],
                )
                self._interrupt_current_turn()
            else:
                logger.info(
                    "[StreamingPipeline] EOT: score=%.2f → soft interrupt "
                    "(timeout=%.2fs). text=%r",
                    score,
                    self._soft_interrupt_timeout,
                    text[:80],
                )
                self._enter_soft_interrupt()
        else:
            logger.info(
                "[StreamingPipeline] EOT: should_interrupt=False, text=%r",
                text[:80],
            )

    def _interrupt_current_turn(self) -> None:
        """Interrupt the currently in-progress agent turn via session.interrupt()."""
        if not self._allow_interruptions:
            return

        if self._session is not None:
            self._session.interrupt()

        # Reset VAD state so EOT doesn't carry stale state into the next turn.
        self._get_eot_model().update_vad(False)

        logger.info("[StreamingPipeline] turn interrupted")

    def _enter_soft_interrupt(self) -> None:
        """
        Enter soft interrupt: wait for confirmation before performing a hard interrupt.

        The soft interrupt timer is started. If it fires, we upgrade to a hard
        interrupt. If the user falls silent (VAD inactive) during the wait, we
        cancel the soft interrupt (false interruption).
        """
        self._soft_interrupt_active = True
        self._soft_interrupt_timer = asyncio.create_task(
            self._soft_interrupt_timeout_task()
        )
        logger.info("[StreamingPipeline] soft interrupt entered (timeout=%.1fs)", self._soft_interrupt_timeout)

    async def _soft_interrupt_timeout_task(self) -> None:
        """Timer task: fires after _soft_interrupt_timeout → upgrade to hard interrupt."""
        try:
            await asyncio.sleep(self._soft_interrupt_timeout)
            if self._soft_interrupt_active:
                logger.info(
                    "[StreamingPipeline] soft interrupt timeout → upgrading to hard interrupt"
                )
                self._cancel_soft_interrupt()
                self._interrupt_current_turn()
        except asyncio.CancelledError:
            pass  # Cancelled when soft interrupt is resolved (false interruption)

    def _cancel_soft_interrupt(self) -> None:
        """Cancel soft interrupt (detected as a false interruption)."""
        self._soft_interrupt_active = False
        if self._soft_interrupt_timer:
            self._soft_interrupt_timer.cancel()
            self._soft_interrupt_timer = None
        logger.info("[StreamingPipeline] soft interrupt cancelled (false interruption)")

    # ------------------------------------------------------------------
    # DuckingMixer integration
    # ------------------------------------------------------------------
    #
    # State machine driven by VAD + EOT events:
    #
    #   user_state listening → speaking
    #     → _duck_and_arm_timeout()
    #         mixer.duck()  (50 ms fade-out to silence)
    #         start _duck_timeout_task (default 0.5 s fallback)
    #
    #   _run_eot_check (per STT interim/final):
    #     strong_interrupt_intent OR score >= duck_early_cancel_score_threshold
    #       → _duck_cancel_and_interrupt()  (real interrupt, no resume)
    #     score <= duck_early_resume_score_threshold (and > 0)
    #       → _duck_unduck()  (false interrupt, smooth fade-in)
    #     mid-band → leave SUSPENDED, let timeout decide
    #
    #   _duck_timeout_task fires (no decision in window):
    #     → mixer.unduck()  (default to false-interrupt, conservative)
    #
    #   user_state speaking → listening (user actually finished):
    #     → if still SUSPENDED, mixer.unduck()  (false interrupt confirmed)
    #
    # Tunables: see EidolonEOTConfig "Ducking mixer + early-resume watcher"
    # block (duck_enabled, duck_fade_ms, duck_suspend_volume,
    # duck_suspend_timeout_sec, duck_early_cancel_score_threshold,
    # duck_early_resume_score_threshold).
    # ------------------------------------------------------------------

    def _install_duck_mixer(self, session: "AgentSession") -> None:
        """Wrap the session's audio output sink with a :class:`DuckingMixer`.

        Called once after ``session.start()``. If duck_enabled is False or
        the session has no audio sink (rare — only in headless tests),
        skips the install and the rest of this module's duck_* paths
        become no-ops.
        """
        cfg = self._get_eot_model()._config
        if not getattr(cfg, "duck_enabled", True):
            logger.info("[StreamingPipeline] duck_enabled=False — skipping DuckingMixer")
            return
        inner = session.output.audio
        if inner is None:
            logger.warning(
                "[StreamingPipeline] session.output.audio is None — "
                "DuckingMixer not installed (interrupt path falls back to "
                "soft/hard interrupt only)"
            )
            return
        mixer = DuckingMixer(
            inner,
            fade_ms=cfg.duck_fade_ms,
            fade_in_ms=cfg.duck_fade_in_ms,
            suspend_volume=cfg.duck_suspend_volume,
            buffer_max_sec=cfg.duck_buffer_max_sec,
        )
        session.output.audio = mixer
        self._duck_mixer = mixer
        logger.info(
            "[StreamingPipeline] DuckingMixer installed "
            "(fade_out=%dms fade_in=%dms suspend_vol=%.2f "
            "buffer_max=%.1fs cancel_thr=%.2f resume_thr=%.2f "
            "timeout=%.2fs cooldown=%.2fs)",
            cfg.duck_fade_ms,
            cfg.duck_fade_in_ms,
            cfg.duck_suspend_volume,
            cfg.duck_buffer_max_sec,
            cfg.duck_early_cancel_score_threshold,
            cfg.duck_early_resume_score_threshold,
            cfg.duck_suspend_timeout_sec,
            cfg.duck_cooldown_sec,
        )

    def _duck_and_arm_timeout(self) -> None:
        """Fade output to silence and arm the suspend-window fallback.

        Triggered on every ``user_state: listening → speaking``. Idempotent
        — reentrant calls during a still-active suspend just restart the
        ramp from current volume (no click) and reset the timeout deadline.

        Skips the duck if within ``duck_cooldown_sec`` of the last unduck
        to prevent "volume yo-yo" from rapid VAD toggling.
        """
        if self._duck_mixer is None:
            return
        # F3 (2026-05-16): skip duck when agent isn't actually speaking.
        # Previously, every ``user_state: listening → speaking`` armed a duck
        # cycle even when ``agent_state=listening`` (idle), wasting fade-out/
        # fade-in compute and producing misleading "duck NORMAL→SUSPENDED" log
        # noise. The duck only has work to do when the agent is mid-utterance.
        if self._state != PipelineState.SPEAKING:
            logger.debug(
                "[StreamingPipeline] duck skipped — agent not speaking (state=%s)",
                self._state.name if hasattr(self._state, "name") else self._state,
            )
            return
        cfg = self._get_eot_model()._config
        if self._filler is not None and self._filler.is_playing:
            logger.info("[StreamingPipeline] duck skipped — filler playing")
            return
        now = time.monotonic()
        if now - self._last_unduck_time < cfg.duck_cooldown_sec:
            logger.info(
                "[StreamingPipeline] duck skipped — within cooldown (%.2fs since last unduck)",
                now - self._last_unduck_time,
            )
            return
        # Cancel any prior timeout before re-arming.
        self._cancel_duck_timeout()
        self._duck_suspend_start = now
        self._duck_mixer.duck()
        self._callbacks.on_duck_started()
        vad_to_duck_ms = 0.0
        if self._user_speaking_start_time is not None:
            vad_to_duck_ms = (now - self._user_speaking_start_time) * 1000
        logger.info(
            "[StreamingPipeline] duck armed  vad→duck=%.1fms  "
            "timeout=%.2fs  cooldown=%.2fs",
            vad_to_duck_ms, cfg.duck_suspend_timeout_sec, cfg.duck_cooldown_sec,
        )
        self._duck_timeout_task = asyncio.create_task(
            self._duck_suspend_timeout_fallback(cfg.duck_suspend_timeout_sec)
        )

    async def _duck_suspend_timeout_fallback(self, timeout_sec: float) -> None:
        """Fallback: if no EOT decision arrives in ``timeout_sec``, unduck.

        Conservative choice — defaulting to unduck means a real interrupt
        that didn't produce STT text in time gets resumed. The alternative
        (default-cancel) would be more disruptive on noise / echo / dropped
        STT. Real interrupts almost always produce text within 300-500 ms
        on Bailian streaming, so the timeout normally fires only for
        genuine false positives.
        """
        try:
            await asyncio.sleep(timeout_sec)
            if self._duck_mixer is not None and self._duck_mixer.state == "SUSPENDED":
                suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
                buffered = self._duck_mixer.buffered_frames
                buffered_sec = self._duck_mixer.buffered_sec
                logger.info(
                    "[StreamingPipeline] duck resolved  reason=timeout  "
                    "action=unduck  suspend_ms=%.0f  "
                    "buffered=%d frames (%.3fs)  timeout=%.2fs",
                    suspend_ms, buffered, buffered_sec, timeout_sec,
                )
                self._duck_mixer.unduck()
                self._last_unduck_time = time.monotonic()
                self._callbacks.on_duck_resolved("timeout")
        except asyncio.CancelledError:
            pass

    def _cancel_duck_timeout(self) -> None:
        """Cancel the suspend-window fallback task if active. Safe to call any time."""
        if self._duck_timeout_task is not None and not self._duck_timeout_task.done():
            self._duck_timeout_task.cancel()
        self._duck_timeout_task = None

    def _duck_cancel_and_interrupt(self) -> None:
        """Confirm interrupt: discard buffer + cancel TTS generation."""
        self._cancel_duck_timeout()
        suspend_ms = 0.0
        buffered = 0
        buffered_sec = 0.0
        if self._duck_mixer is not None:
            suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
            buffered = self._duck_mixer.buffered_frames
            buffered_sec = self._duck_mixer.buffered_sec
        logger.info(
            "[StreamingPipeline] duck resolved  reason=eot_cancel  "
            "action=cancel  suspend_ms=%.0f  discarded=%d frames (%.3fs)",
            suspend_ms, buffered, buffered_sec,
        )
        self._snapshot_interrupted_context()
        if self._duck_mixer is not None:
            self._duck_mixer.cancel()
        self._callbacks.on_duck_resolved("cancel")
        self._interrupt_current_turn()

    def _duck_unduck_if_suspended(self, reason: str = "user_silent") -> None:
        """Resume the agent's TTS if we're SUSPENDED. No-op otherwise."""
        if self._duck_mixer is None:
            return
        self._cancel_duck_timeout()
        if self._duck_mixer.state == "SUSPENDED":
            suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
            buffered = self._duck_mixer.buffered_frames
            buffered_sec = self._duck_mixer.buffered_sec
            logger.info(
                "[StreamingPipeline] duck resolved  reason=%s  "
                "action=unduck  suspend_ms=%.0f  "
                "buffered=%d frames (%.3fs)",
                reason, suspend_ms, buffered, buffered_sec,
            )
            self._duck_mixer.unduck()
            self._last_unduck_time = time.monotonic()
            self._callbacks.on_duck_resolved("unduck")

    # ------------------------------------------------------------------
    # Interrupted content tracking (Phase 3)
    # ------------------------------------------------------------------

    def _snapshot_interrupted_context(self) -> None:
        """Capture the agent's last response text at the point of interruption."""
        cfg = self._get_eot_model()._config
        if not cfg.interrupted_context_enabled:
            return
        if self._session is None:
            return
        try:
            messages = self._session.history.messages
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.text_content:
                    self._last_interrupted_context = {
                        "text": msg.text_content,
                        "timestamp": time.monotonic(),
                    }
                    logger.info(
                        "[StreamingPipeline] interrupted context captured: %r",
                        msg.text_content[:80],
                    )
                    return
        except Exception:
            logger.warning(
                "[StreamingPipeline] failed to capture interrupted context",
                exc_info=True,
            )

    def _inject_interrupted_context(self) -> None:
        """Inject interrupted context into the conversation history.

        Called before ``commit_user_turn()`` so the LLM sees the context
        when generating its next response.
        """
        if self._last_interrupted_context is None:
            return
        if self._session is None:
            return
        cfg = self._get_eot_model()._config
        age = time.monotonic() - self._last_interrupted_context["timestamp"]
        if age > cfg.interrupted_context_max_age_sec:
            logger.info(
                "[StreamingPipeline] interrupted context expired (age=%.1fs)",
                age,
            )
            self._last_interrupted_context = None
            return

        interrupted_text = self._last_interrupted_context["text"]
        self._last_interrupted_context = None

        try:
            from livekit.agents.llm import ChatMessage

            hint = ChatMessage.create(
                text=(
                    f"[系统提示] 你刚才说到「{interrupted_text[:200]}」时被用户打断了。"
                    "如果用户的新问题与之前话题相关，你可以自然地衔接回去；"
                    "如果无关，直接回答新问题即可。不要提及这条系统提示。"
                ),
                role="system",
            )
            self._session.history.insert(hint)
            logger.info(
                "[StreamingPipeline] injected interrupted context hint (%d chars)",
                len(interrupted_text),
            )
        except Exception:
            logger.warning(
                "[StreamingPipeline] failed to inject interrupted context",
                exc_info=True,
            )
