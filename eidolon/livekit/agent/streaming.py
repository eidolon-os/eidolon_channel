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
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents.voice import Agent as lk_Agent
    from livekit.agents.voice import AgentSession
    from livekit.rtc import Room

from eidolon.livekit.common.config import (
    ObservabilityConfig,
    TurnPolicyConfig,
)

from . import _framework_patches
from .client_audio_state import ClientAudioState
from .context import InterruptedContextManager
from .turn_policy import (
    AttentionDecision,
    Decision,
    TurnPolicyRuntime,
    eot_kwargs_from_turn_policy,
)
from .observability import TurnTimeline
from .factory import SharedStageFactory
from .output import FillerManager, OutputController, OutputDuckingController
from .pipeline.base import BasePipeline
from .pipeline.types import PipelineCallbacks, PipelineState, generate_turn_id
from .session import (
    AgentStateEffectHandler,
    AttentionEffectHandler,
    DecisionEffectApplier,
    DuckSuspendTimeoutHandler,
    IdleWatchdog,
    ProviderEventObserver,
    RoomDataHandler,
    SemanticInterruptHandler,
    SessionSignalBridge,
    SoftInterruptController,
    UserTurnCommitter,
)

logger = logging.getLogger("agent")


# Module-level cache for the EOT model singleton.
# All StreamingPipeline instances share the same ChineseModel instance, which in turn
# shares the same EotManager singleton (and thus the same ONNX session).
_eot_model_cache: Any = None
_eot_model_cache_key: tuple | None = None


def _get_shared_eot_model(turn_policy: TurnPolicyConfig | None = None) -> Any:
    """Lazily create and cache the shared ChineseModel instance.

    The EotManager inside ChineseModel is a thread-safe singleton that holds
    the ONNX session, so all callers share the same model weights in memory.
    """
    global _eot_model_cache, _eot_model_cache_key
    kwargs = eot_kwargs_from_turn_policy(turn_policy)
    key = tuple(sorted(kwargs.items()))
    if _eot_model_cache is None or _eot_model_cache_key != key:
        from eidolon.livekit.plugins.eot import ChineseModel

        logger.info("[StreamingPipeline] loading EOT model...")
        _eot_model_cache = ChineseModel(**kwargs)
        _eot_model_cache_key = key
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
        aec_warmup_duration: float | None = 1.0,
        turn_policy: TurnPolicyConfig | None = None,
        observability: ObservabilityConfig | None = None,
        on_idle_disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        self._turn_policy = turn_policy or TurnPolicyConfig()
        self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        self._observability = observability or ObservabilityConfig()
        self._timeline: TurnTimeline | None = None
        self._timeline_debug_flushed = False
        self._decision_effects = self._build_decision_effect_applier()
        self._attention_effects = self._build_attention_effect_handler()
        self._session_signals = self._build_session_signal_bridge()
        self._turn_committer = UserTurnCommitter()
        self._agent_state_effects = self._build_agent_state_effect_handler()
        self._semantic_interrupts = self._build_semantic_interrupt_handler()
        self._duck_deadline = self._build_duck_suspend_timeout_handler()
        self._provider_events = ProviderEventObserver(
            factory=self._factory,
            get_timeline=lambda: self._timeline,
        )
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
        # G9 (2026-05-17): seconds the framework will ignore user audio after
        # the first agent-speaking transition. None / 0 disables.
        self._aec_warmup_duration = aec_warmup_duration

        self._session: AgentSession | None = None
        self._client_audio_states: dict[str, ClientAudioState] = {}
        self._room_data = RoomDataHandler(
            get_timeline=lambda: getattr(self, "_timeline", None),
        )
        # Set when AgentSession emits "close" event (e.g. participant disconnect).
        # run() awaits this instead of polling room.isconnected, so shutdown
        # fires within milliseconds of the framework deciding to close.
        self._session_closed_event: asyncio.Event = asyncio.Event()

        # Idle-disconnect watchdog. A client that connects and is never closed
        # keeps STT streaming (and billing) for the whole connection even while
        # silent. The watchdog closes the session after
        # ``idle.disconnect_after_idle_ms`` of no recognized speech and no agent
        # activity. ``_last_activity_monotonic`` is refreshed by
        # ``_mark_activity()`` on real ASR text and on agent thinking/speaking;
        # raw VAD/noise (which yields empty ASR) deliberately does NOT count, so
        # a silent-but-noisy room still disconnects. <=0 disables the watchdog.
        self._idle_timeout_sec: float = (
            self._turn_policy.idle.disconnect_after_idle_ms / 1000.0
        )
        self._idle_watchdog_task: asyncio.Task | None = None
        self._last_activity_monotonic: float = 0.0
        # Called when the idle timeout fires — deletes the room so the
        # still-connected client is actively disconnected (see server.py).
        self._on_idle_disconnect = on_idle_disconnect
        # Grace between notifying the client and deleting the room, so the
        # reliable data packet reaches the client before it is kicked.
        self._idle_disconnect_grace_sec: float = 0.3
        self._idle_watchdog_controller = IdleWatchdog(
            timeout_sec=self._idle_timeout_sec,
            get_session=lambda: getattr(self, "_session", None),
            get_room=lambda: getattr(self, "_room", None),
            get_timeline=lambda: getattr(self, "_timeline", None),
            session_closed_event=self._session_closed_event,
            on_idle_disconnect=self._on_idle_disconnect,
            disconnect_grace_sec=self._idle_disconnect_grace_sec,
        )

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
        self._soft_interrupt_timeout: float = self._turn_runtime.decision_timeout_sec
        self._soft_interrupt = SoftInterruptController(
            timeout_sec=self._soft_interrupt_timeout,
            on_timeout=lambda: self._interrupt_current_turn(),
        )

        # DuckingMixer + suspend-window timeout fallback task. Both lazy.
        # Mixer is installed in ``run()`` after ``session.start()`` returns,
        # so the AudioOutput chain is fully assembled. The timeout task is
        # started on every VAD-start (user_state listening → speaking) and
        # cancelled the moment EOT decides to cancel/unduck.
        self._ducking = OutputDuckingController()

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
        self._interrupted_context = InterruptedContextManager()

        # Eagerly trigger EOT model loading so the ONNX session is ready before
        # the first user audio frame arrives. This avoids cold-start delay after
        # session.start() is called.
        _get_shared_eot_model(self._turn_policy)
        self._install_provider_observers()

    def _get_eot_model(self) -> Any:
        """Return the shared EOT model instance."""
        return _get_shared_eot_model(self._turn_policy)

    @property
    def _duck_mixer(self) -> OutputController | None:
        self._ensure_ducking_controller()
        return self._ducking.mixer

    @_duck_mixer.setter
    def _duck_mixer(self, value: OutputController | None) -> None:
        self._ensure_ducking_controller()
        self._ducking.mixer = value

    @property
    def _duck_timeout_task(self) -> asyncio.Task | None:
        self._ensure_ducking_controller()
        return self._ducking.timeout_task

    @_duck_timeout_task.setter
    def _duck_timeout_task(self, value: asyncio.Task | None) -> None:
        self._ensure_ducking_controller()
        self._ducking.timeout_task = value

    @property
    def _last_unduck_time(self) -> float:
        self._ensure_ducking_controller()
        return self._ducking.last_unduck_time

    @_last_unduck_time.setter
    def _last_unduck_time(self, value: float) -> None:
        self._ensure_ducking_controller()
        self._ducking.last_unduck_time = value

    @property
    def _duck_suspend_start(self) -> float:
        self._ensure_ducking_controller()
        return self._ducking.suspend_start

    @_duck_suspend_start.setter
    def _duck_suspend_start(self, value: float) -> None:
        self._ensure_ducking_controller()
        self._ducking.suspend_start = value

    def _ensure_ducking_controller(self) -> None:
        if not hasattr(self, "_ducking"):
            self._ducking = OutputDuckingController()

    def _build_decision_effect_applier(self) -> DecisionEffectApplier:
        return DecisionEffectApplier(
            factory=getattr(self, "_factory", None),
            turn_runtime=self._turn_runtime,
            get_timeline=lambda: self._timeline,
            on_cancel=lambda: self._duck_cancel_and_interrupt(),
            on_rollback=lambda reason, drop_buffered: self._duck_unduck_if_suspended(
                reason=reason,
                drop_buffered=drop_buffered,
            ),
        )

    def _ensure_decision_effect_applier(self) -> None:
        if not hasattr(self, "_decision_effects"):
            self._decision_effects = self._build_decision_effect_applier()

    def _build_attention_effect_handler(self) -> AttentionEffectHandler:
        return AttentionEffectHandler(
            turn_policy=self._turn_policy,
            turn_runtime=self._turn_runtime,
            get_agent_speaking=lambda: self._state == PipelineState.SPEAKING,
            get_duck_active=lambda: self._ducking.is_suspended,
            latest_client_audio_state=lambda participant_identity: (
                self._latest_client_audio_state(
                    participant_identity=participant_identity,
                )
            ),
            get_timeline=lambda: self._timeline,
            on_duck=lambda: self._duck_and_arm_timeout(),
            on_interrupt=lambda: self._interrupt_current_turn(),
        )

    def _ensure_attention_effect_handler(self) -> None:
        handler = getattr(self, "_attention_effects", None)
        if (
            handler is None
            or getattr(handler, "_turn_policy", None) is not self._turn_policy
            or getattr(handler, "_turn_runtime", None) is not self._turn_runtime
        ):
            self._attention_effects = self._build_attention_effect_handler()

    def _build_session_signal_bridge(self) -> SessionSignalBridge:
        return SessionSignalBridge(
            factory=getattr(self, "_factory", None),
            get_eot_model=lambda: self._get_eot_model(),
        )

    def _ensure_session_signal_bridge(self) -> None:
        if not hasattr(self, "_session_signals"):
            self._session_signals = self._build_session_signal_bridge()

    def _ensure_turn_committer(self) -> None:
        if not hasattr(self, "_turn_committer"):
            self._turn_committer = UserTurnCommitter()

    def _build_agent_state_effect_handler(self) -> AgentStateEffectHandler:
        return AgentStateEffectHandler(
            get_timeline=lambda: self._timeline,
            mark_activity=lambda: self._mark_activity(),
            cancel_soft_interrupt=lambda: self._cancel_soft_interrupt(),
            soft_interrupt_active=lambda: self._soft_interrupt_active,
            ducking=self._ducking,
            get_filler=lambda: self._filler,
            flush_timeline_debug=lambda reason, clear: self._append_timeline_debug(
                reason,
                clear=clear,
            ),
        )

    def _ensure_agent_state_effect_handler(self) -> None:
        handler = getattr(self, "_agent_state_effects", None)
        if handler is None or getattr(handler, "_ducking", None) is not self._ducking:
            self._agent_state_effects = self._build_agent_state_effect_handler()

    def _build_semantic_interrupt_handler(self) -> SemanticInterruptHandler:
        return SemanticInterruptHandler(
            get_eot_model=lambda: self._get_eot_model(),
            turn_runtime=self._turn_runtime,
            get_timeline=lambda: self._timeline,
            get_duck_active=lambda: self._ducking.is_suspended,
            get_duck_stats=lambda: self._ducking.stats(),
            get_vad_active=lambda: (
                self._session is not None
                and self._session.user_state == "speaking"
            ),
            soft_interrupt_active=lambda: self._soft_interrupt_active,
            soft_interrupt_timeout=lambda: self._soft_interrupt_timeout,
            apply_decision=lambda decision, **kwargs: self._apply_decision(
                decision,
                **kwargs,
            ),
            record_decision_attrs=lambda decision, **kwargs: (
                self._record_decision_attrs(decision, **kwargs)
            ),
            publish_turn_control=lambda metadata: self._publish_turn_control(metadata),
            cancel_duck_and_interrupt=lambda: self._duck_cancel_and_interrupt(),
            interrupt_current_turn=lambda: self._interrupt_current_turn(),
            enter_soft_interrupt=lambda: self._enter_soft_interrupt(),
        )

    def _ensure_semantic_interrupt_handler(self) -> None:
        handler = getattr(self, "_semantic_interrupts", None)
        if (
            handler is None
            or getattr(handler, "_turn_runtime", None) is not self._turn_runtime
        ):
            self._semantic_interrupts = self._build_semantic_interrupt_handler()

    def _build_duck_suspend_timeout_handler(self) -> DuckSuspendTimeoutHandler:
        return DuckSuspendTimeoutHandler(
            turn_runtime=self._turn_runtime,
            sleep=lambda timeout_sec: asyncio.sleep(timeout_sec),
            create_task=lambda coro: asyncio.create_task(coro),
            get_duck_suspended=lambda: self._ducking.is_suspended,
            get_duck_stats=lambda: self._ducking.stats(),
            get_suspend_start=lambda: self._ducking.suspend_start,
            set_timeout_task=lambda task: setattr(self._ducking, "timeout_task", task),
            get_latest_asr_text=lambda: self._latest_asr_text,
            get_vad_active=lambda: (
                self._session is not None
                and self._session.user_state == "speaking"
            ),
            get_eot_model=lambda: self._get_eot_model(),
            apply_decision=lambda decision, **kwargs: self._apply_decision(
                decision,
                **kwargs,
            ),
        )

    def _ensure_duck_suspend_timeout_handler(self) -> None:
        handler = getattr(self, "_duck_deadline", None)
        if (
            handler is None
            or getattr(handler, "_turn_runtime", None) is not self._turn_runtime
        ):
            self._duck_deadline = self._build_duck_suspend_timeout_handler()

    def _install_provider_observers(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_all()
        self._sync_provider_event_compat_attrs()

    def _install_llm_metrics_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_llm_metrics_observer()
        self._sync_provider_event_compat_attrs()

    def _install_brain_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_brain_provider_event_observer()
        self._sync_provider_event_compat_attrs()

    def _install_tts_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_tts_provider_event_observer()
        self._sync_provider_event_compat_attrs()

    def _install_stt_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_stt_provider_event_observer()
        self._sync_provider_event_compat_attrs()

    def _remember_pending_stt_provider_event(self, event: dict[str, Any]) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.remember_pending_stt_provider_event(event)
        self._sync_provider_event_compat_attrs()

    def _apply_pending_stt_provider_events(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.apply_pending_stt_provider_events()
        self._sync_provider_event_compat_attrs()

    def _record_stt_provider_event(self, event: dict[str, Any]) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.record_stt_provider_event(event)

    def _observe_stt_turn_audio(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.observe_stt_turn_audio()

    def _ensure_provider_event_observer(self) -> None:
        if not hasattr(self, "_provider_events"):
            self._provider_events = ProviderEventObserver(
                factory=self._factory,
                get_timeline=lambda: self._timeline,
            )
        self._sync_provider_event_compat_attrs()

    def _sync_provider_event_compat_attrs(self) -> None:
        provider_events = getattr(self, "_provider_events", None)
        if provider_events is None:
            return
        self._llm_metrics_observer_installed = (
            provider_events.llm_metrics_observer_installed
        )
        self._brain_provider_observer_installed = (
            provider_events.brain_provider_observer_installed
        )
        self._stt_provider_observer_installed = (
            provider_events.stt_provider_observer_installed
        )
        self._tts_provider_observer_installed = (
            provider_events.tts_provider_observer_installed
        )
        self._pending_stt_provider_events = (
            provider_events.pending_stt_provider_events
        )

    def _ensure_runtime_defaults(self) -> None:
        """Ensure new runtime helpers exist on test-built pipeline objects.

        Some focused unit tests instantiate ``StreamingPipeline`` via
        ``__new__`` to avoid LiveKit setup. Keep that fast path working while
        the production constructor remains the single source of defaults.
        """
        if not hasattr(self, "_turn_policy"):
            self._turn_policy = TurnPolicyConfig()
        if not hasattr(self, "_turn_runtime"):
            self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        if not hasattr(self, "_observability"):
            self._observability = ObservabilityConfig()
        if not hasattr(self, "_timeline"):
            self._timeline = None
        if not hasattr(self, "_timeline_debug_flushed"):
            self._timeline_debug_flushed = False
        if not hasattr(self, "_latest_asr_text"):
            self._latest_asr_text = ""
        if not hasattr(self, "_llm_metrics_observer_installed"):
            self._llm_metrics_observer_installed = False
        if not hasattr(self, "_brain_provider_observer_installed"):
            self._brain_provider_observer_installed = False
        if not hasattr(self, "_stt_provider_observer_installed"):
            self._stt_provider_observer_installed = False
        if not hasattr(self, "_pending_stt_provider_events"):
            self._pending_stt_provider_events = []
        if not hasattr(self, "_client_audio_states"):
            self._client_audio_states = {}
        self._ensure_ducking_controller()
        self._ensure_decision_effect_applier()
        self._ensure_attention_effect_handler()
        self._ensure_session_signal_bridge()
        self._ensure_turn_committer()
        self._ensure_agent_state_effect_handler()
        self._ensure_semantic_interrupt_handler()
        self._ensure_duck_suspend_timeout_handler()
        self._ensure_room_data_handler()

    async def run(self, room: Room) -> None:
        """Start the streaming pipeline. Blocks until room disconnects."""
        from livekit.agents.voice import AgentSession

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
                # Phase 2 (2026-05-30): preemptive (speculative) brain
                # generation, config-gated via turn_policy.preemptive.
                # Hides the ~990ms STT-final wait by starting the brain on a
                # stable interim/preflight transcript; the framework reuses it
                # if the final transcript matches, else cancels via our gRPC
                # CancelTurn (clean: stops the upstream LLM, no orphan tokens,
                # no history mutation). ``preemptive_tts`` stays False so audio
                # output is still gated by our commit (no partial-audio leak,
                # which was the R8.12.c concern). Was hard-disabled in R8.12.c
                # because _inject_interrupted_context() mutates chat_ctx before
                # commit on *post-interruption* turns, breaking is_equivalent;
                # normal turns do not diverge and now get the speedup.
                "preemptive_generation": {
                    "enabled": self._turn_policy.preemptive.enabled,
                    "preemptive_tts": self._turn_policy.preemptive.preemptive_tts,
                },
            },
            # G9 (2026-05-17): framework public API. Default 3.0s only covers
            # ~2/3 of a typical Chinese welcome; we expose this via env so
            # deployments can pick: 0/None = always interruptible; 0.5-1.0 =
            # brief protect; longer = full welcome protected.
            aec_warmup_duration=self._aec_warmup_duration,
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
        self._install_room_data_observer(room)
        # G3 (2026-05-16): migrated from deprecated RoomInputOptions/
        # RoomOutputOptions to the new RoomOptions schema. Equivalent
        # behaviour:
        #   * audio_input / text_input / text_output left NOT_GIVEN → framework
        #     defaults (all enabled), matching the old RoomInputOptions() and
        #     RoomOutputOptions(transcription_enabled=True) behaviour.
        #   * audio_output overridden only to set the sample rate (we want
        #     to align the entire chain to the TTS native rate, avoiding
        #     unnecessary resampling in the framework).
        from livekit.agents.voice.room_io import RoomOptions, AudioOutputOptions

        await session.start(
            agent=agent,
            room=room,
            room_options=RoomOptions(
                audio_output=AudioOutputOptions(
                    sample_rate=self._audio_sample_rate,
                ),
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

        # Start the idle-disconnect watchdog (after session.start() so the
        # welcome message — which counts as agent activity — has set the
        # initial activity timestamp). See _idle_watchdog for the policy.
        self._start_idle_watchdog()

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
        self._stop_idle_watchdog()
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:
                logger.exception("[StreamingPipeline] error shutting down session")
            self._session = None
        # Tear down persistent-connection stages (STT, TTS, etc.).
        await self._shutdown_stages()
        await super().shutdown()

    def _install_room_data_observer(self, room: Room) -> None:
        """Observe client-side audio hints for later admission policy use."""
        self._ensure_room_data_handler()
        self._room_data.install(room)
        self._sync_room_data_compat_attrs()

    def _on_room_data_received(self, packet: Any) -> None:
        self._ensure_room_data_handler()
        self._room_data.handle_packet(packet)
        self._sync_room_data_compat_attrs()

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
        duck_metrics = self._ducking.get_metrics()
        if duck_metrics is not None:
            logger.info(
                "[StreamingPipeline] session duck metrics: %s", duck_metrics,
            )
        self._append_timeline_debug("session_closed")
        self._session_closed_event.set()

    # ------------------------------------------------------------------
    # Idle-disconnect watchdog
    # ------------------------------------------------------------------

    def _mark_activity(self) -> None:
        """Record that the session is doing real work *right now*.

        Refreshing this timestamp pushes back the idle-disconnect deadline.
        Called on recognized ASR text and on agent thinking/speaking — never
        on bare VAD/noise, so a connected-but-silent client still times out.
        """
        self._ensure_idle_watchdog_controller()
        self._idle_watchdog_controller.mark_activity()
        self._sync_idle_watchdog_compat_attrs()

    def _start_idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        self._idle_watchdog_controller.start()
        self._sync_idle_watchdog_compat_attrs()

    def _stop_idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        self._idle_watchdog_controller.stop()
        self._sync_idle_watchdog_compat_attrs()

    async def _idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.run()
        self._sync_idle_watchdog_compat_attrs()

    async def _disconnect_idle(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.disconnect_idle()
        self._sync_idle_watchdog_compat_attrs()

    async def _notify_client_idle_timeout(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.notify_client_idle_timeout()

    def _ensure_idle_watchdog_controller(self) -> None:
        if not hasattr(self, "_session_closed_event"):
            self._session_closed_event = asyncio.Event()
        if not hasattr(self, "_idle_timeout_sec"):
            self._idle_timeout_sec = 0.0
        if not hasattr(self, "_on_idle_disconnect"):
            self._on_idle_disconnect = None
        if not hasattr(self, "_idle_disconnect_grace_sec"):
            self._idle_disconnect_grace_sec = 0.3
        if not hasattr(self, "_idle_watchdog_controller"):
            self._idle_watchdog_controller = IdleWatchdog(
                timeout_sec=self._idle_timeout_sec,
                get_session=lambda: getattr(self, "_session", None),
                get_room=lambda: getattr(self, "_room", None),
                get_timeline=lambda: getattr(self, "_timeline", None),
                session_closed_event=self._session_closed_event,
                on_idle_disconnect=self._on_idle_disconnect,
                disconnect_grace_sec=self._idle_disconnect_grace_sec,
            )
            self._idle_watchdog_controller.last_activity_monotonic = getattr(
                self,
                "_last_activity_monotonic",
                0.0,
            )
        self._idle_watchdog_controller.timeout_sec = self._idle_timeout_sec
        self._idle_watchdog_controller.disconnect_grace_sec = (
            self._idle_disconnect_grace_sec
        )
        self._sync_idle_watchdog_compat_attrs()

    def _sync_idle_watchdog_compat_attrs(self) -> None:
        controller = getattr(self, "_idle_watchdog_controller", None)
        if controller is None:
            return
        self._idle_watchdog_task = controller.task
        self._last_activity_monotonic = controller.last_activity_monotonic

    def _ensure_room_data_handler(self) -> None:
        if not hasattr(self, "_room_data"):
            self._room_data = RoomDataHandler(
                get_timeline=lambda: getattr(self, "_timeline", None),
            )
            if hasattr(self, "_client_audio_states"):
                self._room_data.client_audio_states = self._client_audio_states
            self._room_data.room_data_packet_count = getattr(
                self,
                "_room_data_packet_count",
                0,
            )
            self._room_data.client_audio_state_packet_count = getattr(
                self,
                "_client_audio_state_packet_count",
                0,
            )
        self._sync_room_data_compat_attrs()

    def _sync_room_data_compat_attrs(self) -> None:
        room_data = getattr(self, "_room_data", None)
        if room_data is None:
            return
        self._client_audio_states = room_data.client_audio_states
        self._room_data_packet_count = room_data.room_data_packet_count
        self._client_audio_state_packet_count = (
            room_data.client_audio_state_packet_count
        )

    def _on_agent_state_changed(self, event: Any) -> None:
        """Forward agent state changes through BasePipeline and session effects."""
        self._ensure_runtime_defaults()
        super()._on_agent_state_changed(event)
        self._ensure_agent_state_effect_handler()
        self._agent_state_effects.handle(event)

    def _register_vad_inference_callback(self) -> None:
        self._ensure_session_signal_bridge()
        self._session_signals.register_vad_inference_callback()

    def _signal_stt_user_away(self) -> None:
        self._ensure_session_signal_bridge()
        self._session_signals.signal_stt_user_away()

    def _signal_stt_user_present(self) -> None:
        self._ensure_session_signal_bridge()
        self._session_signals.signal_stt_user_present()

    def _on_user_state_changed(self, event: Any) -> None:
        self._ensure_runtime_defaults()
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
                self._timeline = TurnTimeline(generate_turn_id())
                self._timeline_debug_flushed = False
                if self._room is not None:
                    self._timeline.set_attr("room_name", self._room.name or "")
                self._timeline.mark("speech_started_at")
                self._apply_pending_stt_provider_events()
                self._observe_stt_turn_audio()
                # Immediately clear stale text so EOT only sees text from THIS speech turn.
                self._latest_asr_text = ""

                # Feed VAD signal into EOT model so VADState reflects user activity.
                self._get_eot_model().update_vad(True)

                # Phase C: immediately fade agent output to silence and arm
                # the suspend-window fallback. EOT decisions in
                # ``_run_eot_check`` will resolve us out of SUSPENDED before
                # the timeout fires in the typical case.
                self._handle_attention_on_speaking_started()

                # If agent is speaking and interruptions are allowed, EOT check is
                # triggered synchronously in _on_user_transcribed as soon as STT
                # delivers the first transcript (INTERIM or FINAL) — no polling needed.

            elif old == "speaking" and new == "listening":
                self._user_speaking_start_time = None
                if self._timeline is not None:
                    self._timeline.mark("speech_stopped_at")

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
                # output smoothly and record the rollback decision.
                if self._ducking.is_suspended:
                    decision = self._turn_runtime.user_silent_decision(
                        self._latest_asr_text
                    )
                    self._apply_decision(
                        decision,
                        resolved_reason="user_silent",
                        transcript=self._latest_asr_text,
                        vad_active=False,
                    )

                self._callbacks.on_user_ended_speaking()
                if self._session is not None:
                    self._ensure_turn_committer()
                    self._turn_committer.commit_or_skip(
                        session=self._session,
                        eot_model=eot_model,
                        transcript=self._latest_asr_text,
                        transcript_timeout=self._stt_commit_transcript_timeout,
                        timeline=self._timeline,
                        inject_interrupted_context=self._inject_interrupted_context,
                        filler=self._filler,
                    )

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
        self._ensure_runtime_defaults()
        if event.transcript:
            # Real recognized speech (interim or final) — keeps the session
            # alive. Empty/noise transcripts deliberately don't, so a silent
            # room still trips the idle watchdog.
            self._mark_activity()
            self._latest_asr_text = event.transcript
            if self._timeline is not None:
                if event.is_final:
                    self._timeline.mark("transcript_final_at")
                else:
                    self._timeline.mark("transcript_interim_first_at")
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
        interruption_timeline_active = (
            self._timeline is not None
            and "interrupt_started_at" in self._timeline.timestamps
        )
        if (
            self._allow_interruptions
            and event.transcript
            and (agent_is_speaking or interruption_timeline_active)
        ):
            if not self._attention_allows_eot_check(
                event.transcript,
                speaker_id=getattr(event, "speaker_id", None),
            ):
                super()._on_user_transcribed(event)
                return
            self._run_eot_check(event.transcript, is_final=event.is_final)

        super()._on_user_transcribed(event)

    def _handle_attention_on_speaking_started(self) -> None:
        self._ensure_attention_effect_handler()
        self._attention_effects.handle_speaking_started()

    def _attention_allows_eot_check(
        self,
        transcript: str,
        *,
        speaker_id: str | None = None,
    ) -> bool:
        self._ensure_attention_effect_handler()
        return self._attention_effects.allows_eot_check(
            transcript,
            speaker_id=speaker_id,
        )

    def _decide_attention(
        self,
        transcript: str,
        *,
        participant_identity: str | None = None,
    ) -> AttentionDecision:
        self._ensure_runtime_defaults()
        self._ensure_attention_effect_handler()
        return self._attention_effects.decide(
            transcript,
            participant_identity=participant_identity,
        )

    def _latest_client_audio_state(
        self,
        *,
        participant_identity: str | None = None,
    ) -> ClientAudioState | None:
        self._ensure_runtime_defaults()
        max_age_sec = self._turn_policy.attention.client_state_max_age_ms / 1000.0
        self._ensure_room_data_handler()
        if self._client_audio_states is not self._room_data.client_audio_states:
            self._room_data.client_audio_states = self._client_audio_states
        return self._room_data.latest_client_audio_state(
            participant_identity=participant_identity,
            max_age_sec=max_age_sec,
        )

    def _record_attention_admission(self, decision: AttentionDecision) -> None:
        self._ensure_attention_effect_handler()
        self._attention_effects.record_admission(decision)

    def _run_eot_check(self, text: str, is_final: bool = False) -> None:
        """
        Synchronous EOT semantic check triggered by each STT transcript event.

        ``turn_policy`` owns tier/intent/action decisions. The session semantic
        handler owns the hot-path side effects: strong-stop fast path,
        duck-active resolution and fallback soft interrupt staging.
        """
        if not text or not text.strip():
            return
        self._ensure_runtime_defaults()
        self._ensure_semantic_interrupt_handler()
        self._semantic_interrupts.run(text, is_final=is_final)

    def _interrupt_current_turn(self) -> None:
        """Interrupt the currently in-progress agent turn via session.interrupt()."""
        self._ensure_ducking_controller()
        if not self._allow_interruptions:
            return
        if self._ducking.is_cancelled:
            logger.debug(
                "[StreamingPipeline] interrupt skipped — output already CANCELLED"
            )
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
        self._ensure_soft_interrupt_controller()
        self._soft_interrupt.enter()
        self._sync_soft_interrupt_compat_attrs()

    async def _soft_interrupt_timeout_task(self) -> None:
        """Timer task: fires after _soft_interrupt_timeout → upgrade to hard interrupt."""
        self._ensure_soft_interrupt_controller()
        await self._soft_interrupt.run_timeout_task()
        self._sync_soft_interrupt_compat_attrs()

    def _cancel_soft_interrupt(self) -> None:
        """Cancel soft interrupt (detected as a false interruption)."""
        self._ensure_soft_interrupt_controller()
        self._soft_interrupt.cancel()
        self._sync_soft_interrupt_compat_attrs()

    def _ensure_soft_interrupt_controller(self) -> None:
        if not hasattr(self, "_soft_interrupt_timeout"):
            self._soft_interrupt_timeout = (
                self._turn_runtime.decision_timeout_sec
                if hasattr(self, "_turn_runtime")
                else 0.5
            )
        if not hasattr(self, "_soft_interrupt"):
            self._soft_interrupt = SoftInterruptController(
                timeout_sec=self._soft_interrupt_timeout,
                on_timeout=lambda: self._interrupt_current_turn(),
            )
            self._soft_interrupt.active = getattr(
                self,
                "_soft_interrupt_active",
                False,
            )
            self._soft_interrupt.task = getattr(
                self,
                "_soft_interrupt_timer",
                None,
            )
        self._soft_interrupt.timeout_sec = self._soft_interrupt_timeout
        self._sync_soft_interrupt_compat_attrs()

    def _sync_soft_interrupt_compat_attrs(self) -> None:
        controller = getattr(self, "_soft_interrupt", None)
        if controller is None:
            return
        self._soft_interrupt_active = controller.active
        self._soft_interrupt_timer = controller.task

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
        mixer = self._ducking.install(session, cfg)
        if mixer is None:
            return
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
        if not self._ducking.installed:
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
        if now - self._ducking.last_unduck_time < cfg.duck_cooldown_sec:
            logger.info(
                "[StreamingPipeline] duck skipped — within cooldown (%.2fs since last unduck)",
                now - self._ducking.last_unduck_time,
            )
            return
        # Cancel any prior timeout before re-arming.
        self._ducking.cancel_timeout()
        self._ducking.duck(now=now)
        if self._timeline is not None:
            self._timeline.mark("interrupt_started_at")
        self._callbacks.on_duck_started()
        vad_to_duck_ms = 0.0
        if self._user_speaking_start_time is not None:
            vad_to_duck_ms = (now - self._user_speaking_start_time) * 1000
        logger.info(
            "[StreamingPipeline] duck armed  vad→duck=%.1fms  "
            "timeout=%.2fs  cooldown=%.2fs",
            vad_to_duck_ms, cfg.duck_suspend_timeout_sec, cfg.duck_cooldown_sec,
        )
        self._ducking.timeout_task = asyncio.create_task(
            self._duck_suspend_timeout_fallback(cfg.duck_suspend_timeout_sec)
        )

    async def _duck_suspend_timeout_fallback(self, timeout_sec: float) -> None:
        """500ms decision-budget deadline. Delegates the branch decision
        to :class:`InterruptDecider` so the policy stays in one place
        (G18b)."""
        self._ensure_runtime_defaults()
        self._ensure_duck_suspend_timeout_handler()
        await self._duck_deadline.run(timeout_sec)

    # ------------------------------------------------------------------
    # G18b (2026-05-18) — decider integration
    # ------------------------------------------------------------------

    def _apply_decision(
        self,
        decision: Decision,
        *,
        resolved_reason: str | None = None,
        eot_score: float | None = None,
        transcript: str = "",
        vad_active: bool | None = None,
    ) -> None:
        """Execute the side effects implied by a :class:`Decision`.

        Args:
            decision: The decider's verdict.
            resolved_reason: If set, override the on_duck_resolved
                callback's reason string (used by the timeout path so
                metrics show "timeout" rather than the decider's
                internal classification).
        """
        self._ensure_runtime_defaults()
        self._ensure_decision_effect_applier()
        self._decision_effects.apply(
            decision,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript=transcript,
            vad_active=vad_active,
        )

    def _record_decision_attrs(
        self,
        decision: Decision,
        *,
        source: str = "turn_policy",
        resolved_reason: str | None = None,
        eot_score: float | None = None,
        transcript: str = "",
        vad_active: bool | None = None,
    ) -> None:
        self._ensure_decision_effect_applier()
        self._decision_effects.record_decision_attrs(
            decision,
            source=source,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript=transcript,
            vad_active=vad_active,
        )

    def _append_timeline_debug(self, reason: str, *, clear: bool = False) -> None:
        if self._timeline is None or self._timeline_debug_flushed:
            return
        self._timeline.set_attr("timeline_flush_reason", reason)
        self._timeline.append_debug_jsonl(self._observability.timeline_debug_path)
        self._timeline_debug_flushed = True
        if clear:
            self._timeline = None

    def _publish_turn_control(self, metadata: dict[str, object]) -> None:
        """Attach control hints to the next remote-brain turn when supported."""
        self._ensure_decision_effect_applier()
        self._decision_effects.publish_turn_control(metadata)

    def _cancel_duck_timeout(self) -> None:
        """Cancel the suspend-window fallback task if active. Safe to call any time."""
        self._ensure_ducking_controller()
        self._ducking.cancel_timeout()

    def _duck_cancel_and_interrupt(self) -> None:
        """Confirm interrupt: discard buffer + cancel TTS generation."""
        self._ensure_runtime_defaults()
        if self._ducking.is_cancelled:
            logger.debug(
                "[StreamingPipeline] duplicate duck cancel ignored — "
                "output already CANCELLED"
            )
            return
        stats = self._ducking.stats()
        logger.info(
            "[StreamingPipeline] duck resolved  reason=eot_cancel  "
            "action=cancel  suspend_ms=%.0f  discarded=%d frames (%.3fs)",
            stats.suspend_ms,
            stats.buffered_frames,
            stats.buffered_sec,
        )
        self._snapshot_interrupted_context()
        self._ducking.cancel_output()
        self._callbacks.on_duck_resolved("cancel")
        if self._timeline is not None:
            self._timeline.mark("interrupt_resolved_at")
            self._timeline.set_attr("cancel_reason", "eot_cancel")
            self._append_timeline_debug("interrupt_cancel")
        self._interrupt_current_turn()

    def _duck_unduck_if_suspended(
        self, reason: str = "user_silent", *, drop_buffered: bool = False
    ) -> None:
        """Resume the agent's TTS if we're SUSPENDED. No-op otherwise.

        Args:
            reason: log + telemetry reason string.
            drop_buffered: passed through to ``OutputController.unduck``.
                True for the timeout-deadline (G17a) path where the
                buffered frames are stale.
        """
        self._ensure_runtime_defaults()
        if not self._ducking.installed:
            return
        if self._ducking.is_suspended:
            stats = self._ducking.stats()
            logger.info(
                "[StreamingPipeline] duck resolved  reason=%s  "
                "action=unduck(drop_buffered=%s)  suspend_ms=%.0f  "
                "buffered=%d frames (%.3fs)",
                reason, drop_buffered,
                stats.suspend_ms,
                stats.buffered_frames,
                stats.buffered_sec,
            )
            self._ducking.unduck_if_suspended(drop_buffered=drop_buffered)
            self._callbacks.on_duck_resolved("unduck")
            if self._timeline is not None:
                self._timeline.mark("interrupt_resolved_at")
                self._timeline.set_attr("rollback_reason", reason)

    # ------------------------------------------------------------------
    # Interrupted content tracking (Phase 3)
    # ------------------------------------------------------------------

    def _snapshot_interrupted_context(self) -> None:
        """Capture the agent's last response text at the point of interruption.

        G6 (2026-05-17): augmented with ``played_seconds`` from
        :class:`DuckingMixer` so the LLM context conveys "you only got the
        first 1.2s out" instead of just "you said X".

        G21 (2026-05-18): primary source is the TTS plugin's in-flight
        ``current_pushed_text`` — captures exactly what the agent was
        synthesizing at the cancel moment. ``session.history`` is the
        fallback for the (rare) case where TTS doesn't expose the
        property: history is only updated AFTER speech_handle winds down,
        which is AFTER our cancel snapshot runs, so we'd otherwise capture
        the PREVIOUS turn's assistant text rather than the in-flight one.
        """
        self._ensure_interrupted_context_manager()
        self._interrupted_context.snapshot(
            session=getattr(self, "_session", None),
            factory=getattr(self, "_factory", None),
            duck_mixer=getattr(self, "_duck_mixer", None),
            config=self._get_eot_model()._config,
        )
        self._sync_interrupted_context_compat_attrs()

    def _inject_interrupted_context(self) -> None:
        """Inject interrupted context into the conversation history.

        Called before ``commit_user_turn()`` so the LLM sees the context
        when generating its next response.
        """
        self._ensure_interrupted_context_manager()
        self._interrupted_context.last_context = self._last_interrupted_context
        self._interrupted_context.inject(
            session=getattr(self, "_session", None),
            config=self._get_eot_model()._config,
        )
        self._sync_interrupted_context_compat_attrs()

    def _ensure_interrupted_context_manager(self) -> None:
        if not hasattr(self, "_interrupted_context"):
            self._interrupted_context = InterruptedContextManager()
            self._interrupted_context.last_context = getattr(
                self,
                "_last_interrupted_context",
                None,
            )

    def _sync_interrupted_context_compat_attrs(self) -> None:
        manager = getattr(self, "_interrupted_context", None)
        if manager is None:
            return
        self._last_interrupted_context = manager.last_context
