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
     own (see ``agent/shared/pipeline.py`` ADR).

  3. **Bridges framework events to our extensions** —
     ``user_state_changed`` → STT user_away signal,
     ``agent_state_changed`` → soft-interrupt timer cancel,
     VAD inference probabilities → EOT state (R7 G6, framework
     doesn't expose this).

  4. **Implements soft/hard interrupt staging** — the framework has
     ``false_interruption_timeout`` for similar intent, but we need
     a state machine that can be cancelled by silence (the user
     paused, didn't actually interrupt) AND that integrates with
     Eidolon's multi-signal interruption owner. Our soft-interrupt
     path is owned by channel in the default hybrid profile.

  5. **Selects one interruption owner through public LiveKit options** —
     native-adaptive profiles let LiveKit detect and apply interruptions;
     channel-owned profiles disable LiveKit's automatic interruption while
     preserving overlapping input audio, then use the public forced-interrupt
     API only after Eidolon's multi-signal policy accepts a candidate.

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

from eidolon_sdk.biz.contracts import (
    CONTROL_OP_PLAYBACK_STOP,
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
)
from eidolon.livekit.common.config import (
    AvatarConfig,
    CoreConfig,
    ObservabilityConfig,
    TurnPolicyConfig,
    VoiceprintConfig,
)

from .agent_builder import build_full_duplex_agent
from ..runtime.interaction_mode import resolve_idle_policy, resolve_welcome_text
from ..turn_policy import TurnPolicyRuntime
from ..observability import ChannelTurnEventSink, TurnTimeline
from ..factory import SharedStageFactory
from ..output import FillerManager, OutputDuckingController
from ..shared.pipeline import BasePipeline
from ..shared.types import PipelineCallbacks, PipelineState, generate_turn_id
from ..session.agent_state import AgentStateEffectHandler
from ..session.agent_output_coordinator import AgentOutputCoordinator
from ..session.assistant_speech import AssistantSpeechLedger
from ..session.attention_effects import AttentionEffectHandler
from .client_preempt import (
    ExplicitClientPreemptHandler,
    ExplicitClientPreemptLedger,
)
from .client_control_publisher import FullDuplexClientControlPublisher
from .client_audio import FullDuplexClientAudioStateView, FullDuplexRoomDataBridge
from .context_ledger import FullDuplexContextLedger
from .idle_watchdog import (
    build_full_duplex_idle_watchdog,
    ensure_full_duplex_idle_watchdog,
)
from .interruption_effects import FullDuplexInterruptionEffects
from .lifecycle import FullDuplexSessionLifecycle
from .output_flow import FullDuplexOutputFlow
from .runtime_defaults import ensure_full_duplex_runtime_defaults
from .state_machine import (
    FullDuplexPhase,
    FullDuplexSideEffect,
    FullDuplexStateMachine,
)
from .transcript_admission import TranscriptAdmissionGate
from .transcript_handler import FullDuplexTranscriptHandler
from .transcript_evidence_buffer import TranscriptEvidenceBuffer
from .transcript_ingress_ledger import FullDuplexTranscriptIngressLedger
from .transcript_recorder import FullDuplexTranscriptRecorder
from .turn_handling import (
    build_full_duplex_turn_handling,
    uses_livekit_native_adaptive_interruption,
)
from .turn_completion import FullDuplexTurnCompletion
from .speech_lifecycle import FullDuplexSpeechLifecycle
from .user_state_handler import FullDuplexUserStateHandler
from ..session.decision_effects import DecisionEffectApplier
from ..session.duck_timeout import DuckSuspendTimeoutHandler
from ..session.eot_model import get_shared_eot_model
from ..session.interruption_orchestrator import (
    InterruptionOrchestrator,
    InterruptionVerdict,
    InterruptionVerdictAction,
)
from ..session.provider_events import ProviderEventObserver
from ..session.room_data import RoomDataHandler
from ..session.semantic_interrupt import SemanticInterruptHandler
from ..session.signals import SessionSignalBridge
from ..session.transcript_echo import TranscriptEchoGate
from ..session.user_turn_coordinator import UserTurnCoordinator
from ..session.voiceprint import VoiceprintTurnObserver

logger = logging.getLogger("agent")

_UNSET = object()


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
        false_interruption_timeout: float | None | object = _UNSET,
        audio_sample_rate: int = 16000,
        stt_commit_transcript_timeout: float | object = _UNSET,
        aec_warmup_duration: float | None | object = _UNSET,
        turn_policy: TurnPolicyConfig | None = None,
        observability: ObservabilityConfig | None = None,
        voiceprint_config: VoiceprintConfig | None = None,
        on_idle_disconnect: Callable[[], Awaitable[None]] | None = None,
        on_session_started: Callable[[], Awaitable[None]] | None = None,
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        on_session_closed: Callable[[], Awaitable[None]] | None = None,
        interaction_mode: str = INTERACTION_MODE_FULL_DUPLEX,
        session_intent: str = SESSION_INTENT_USER_INITIATED,
        avatar_enabled: bool = False,
        avatar_config: AvatarConfig | None = None,
        core_config: CoreConfig | None = None,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        # StreamingPipeline serves BOTH full_duplex and half_duplex: both do
        # streaming STT + EOT-based turn commit. The difference is barge-in, which
        # is carried by allow_interruptions (False for half_duplex) — not by a
        # separate pipeline. Only ptt uses HalfDuplexPttPipeline (button turns).
        if interaction_mode not in (
            INTERACTION_MODE_FULL_DUPLEX,
            INTERACTION_MODE_HALF_DUPLEX,
        ):
            raise ValueError(
                "StreamingPipeline serves full_duplex/half_duplex; use "
                f"HalfDuplexPttPipeline for interaction_mode={interaction_mode!r}"
            )
        self._interaction_mode = interaction_mode
        # Session intent (plan §3.2/§3.3) is orthogonal to interaction_mode and
        # drives only idle window + teardown reason in the full-duplex pipeline.
        self._session_intent = session_intent
        self._is_proactive = session_intent == SESSION_INTENT_PROACTIVE
        # Digital-human video avatar (per-session, resolved at connect time). When
        # enabled, TTS audio is routed to an avatar worker via DataStreamAudioOutput
        # instead of the room; the barge-in machine is reused unchanged. Default off
        # → audio path is byte-for-byte the current behaviour.
        self._avatar_enabled = avatar_enabled
        self._avatar_config = avatar_config or AvatarConfig()
        self._core_config = core_config or CoreConfig()
        self._avatar_worker: Any | None = None
        self._turn_policy = turn_policy or TurnPolicyConfig()
        self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        self._observability = observability or ObservabilityConfig()
        self._turn_events = ChannelTurnEventSink()
        self._voiceprint_config = voiceprint_config or VoiceprintConfig()
        self._timeline: TurnTimeline | None = None
        self._timeline_debug_flushed = False
        self._flushed_timeline_ids: set[str] = set()
        self._agent_output = AgentOutputCoordinator()
        self._explicit_preempt_control_timeline: TurnTimeline | None = None
        self._assistant_speech = AssistantSpeechLedger()
        self._pending_client_control_events: list[dict[str, Any]] = []
        self._full_duplex_irreversible_side_effects: set[tuple[str, str, str]] = set()
        self._skip_commit_after_interrupt_cancel = False
        self._suppress_commit_after_interrupt_until = 0.0
        # Ducking state is shared by several effect handlers. It must exist
        # before those handlers are built, because AgentStateEffectHandler keeps
        # a direct reference to the controller.
        self._ducking = OutputDuckingController()
        self._output_flow = FullDuplexOutputFlow(self)
        self._soft_interrupt_timeout: float = self._turn_runtime.decision_timeout_sec
        self._interruption_effects = self._build_interruption_effects()
        self._decision_effects = self._build_decision_effect_applier()
        self._explicit_preempts = self._build_explicit_client_preempt_ledger()
        self._interruption_orchestrator = self._build_interruption_orchestrator()
        self._attention_effects = self._build_attention_effect_handler()
        self._session_signals = self._build_session_signal_bridge()
        self._client_preempts = self._build_client_preempt_handler()
        self._transcript_echo_gate = self._build_transcript_echo_gate()
        self._user_turns = self._build_user_turn_coordinator()
        self._turn_completion = FullDuplexTurnCompletion(self)
        self._agent_state_effects = self._build_agent_state_effect_handler()
        self._semantic_interrupts = self._build_semantic_interrupt_handler()
        self._duck_deadline = self._build_duck_suspend_timeout_handler()
        self._candidate_voiceprint_tasks: list[asyncio.Task] = []
        self._suppress_transcripts_until_next_speech = False
        self._transcript_admission = self._build_transcript_admission_gate()
        self._transcript_evidence_buffer = TranscriptEvidenceBuffer()
        self._transcript_ingress_ledger = FullDuplexTranscriptIngressLedger()
        self._completed_turn_voiceprint_task: asyncio.Task | None = None
        self._completed_turn_voiceprint_result: Any | None = None
        self._completed_turn_voiceprint_timeline: TurnTimeline | None = None
        self._provider_events = self._build_provider_event_observer()
        self._voiceprint_turns = VoiceprintTurnObserver(
            service=getattr(self._factory, "voiceprint_service", None),
            context_resolver=getattr(self._factory, "runtime_context_resolver", None),
            sample_rate=audio_sample_rate,
            max_audio_ms=self._voiceprint_config.turn_max_audio_ms,
            accept_cache_ttl_sec=(self._voiceprint_config.accept_cache_ttl_ms / 1000.0),
            accept_cache_short_audio_max_ms=(
                self._voiceprint_config.accept_cache_short_audio_max_ms
            ),
            commit_threshold=self._voiceprint_config.owner_commit_threshold,
            owner_short_audio_bypass_ms=(self._voiceprint_config.owner_short_audio_bypass_ms),
            trust_paired_devices=getattr(
                self._factory,
                "voiceprint_trust_paired_devices",
                True,
            ),
        )
        self._instructions = instructions
        self._allow_interruptions = allow_interruptions
        # Round 8 R8.9: fixed welcome (instead of LLM-generated). LLM with
        # only a system prompt context tends to echo back instruction
        # templates, which the user heard as garbled "welcome".
        self._welcome_message = welcome_message
        interrupt_policy = self._turn_policy.interrupt
        # Round 8 R8.9: framework default 2.0s is too short for Chinese
        # STT, which often takes 3-5s to deliver a final transcript. Source of
        # truth is config: turn_policy.interrupt.framework_false_interruption_timeout_ms.
        self._false_interruption_timeout = (
            interrupt_policy.framework_false_interruption_timeout_ms / 1000.0
            if false_interruption_timeout is _UNSET
            else false_interruption_timeout
        )
        self._audio_sample_rate = audio_sample_rate
        # F1 fix (2026-05-16): pass to session.commit_user_turn() so STT FINAL
        # has enough time to arrive before framework promotes the latest INTERIM
        # to a FINAL. Source of truth is config:
        # turn_policy.interrupt.stt_commit_transcript_timeout_ms.
        self._stt_commit_transcript_timeout = (
            interrupt_policy.stt_commit_transcript_timeout_ms / 1000.0
            if stt_commit_transcript_timeout is _UNSET
            else float(stt_commit_transcript_timeout)
        )
        # G9 (2026-05-17): seconds the framework will ignore user audio after
        # the first agent-speaking transition. None / 0 disables. Source of
        # truth is config: turn_policy.interrupt.aec_warmup_ms.
        self._aec_warmup_duration = (
            None
            if aec_warmup_duration is _UNSET and interrupt_policy.aec_warmup_ms is None
            else (
                interrupt_policy.aec_warmup_ms / 1000.0
                if aec_warmup_duration is _UNSET
                else aec_warmup_duration
            )
        )

        self._session: AgentSession | None = None
        # Proactive report consumer: a background stream that lets the brain
        # speak unprompted (e.g. "your meeting notes are ready"). Started after
        # session.start(); torn down in shutdown(). Lazily wired so direct_llm
        # mode (no eidolon_agent gRPC backend) simply skips it.
        self._proactive_task: asyncio.Task | None = None
        self._proactive_subscriber: Any | None = None
        self._room_data = RoomDataHandler(
            get_timeline=lambda: getattr(self, "_timeline", None),
        )
        # Set when AgentSession emits "close" event (e.g. participant disconnect).
        # run() awaits this instead of polling room.isconnected, so shutdown
        # fires within milliseconds of the framework deciding to close.
        self._session_closed_event: asyncio.Event = asyncio.Event()
        # Set when the channel tells us we are no longer in it. run() races this
        # against the event above, because the framework decides to close long
        # after the room does — measured 15s versus 3s after a session was
        # withdrawn, all of it spent holding a metered speech stream open for a
        # conversation that had already ended.
        self._room_disconnected_event: asyncio.Event = asyncio.Event()
        self._lifecycle = FullDuplexSessionLifecycle(self)

        # Idle-disconnect watchdog. A client that connects and is never closed
        # keeps STT streaming (and billing) for the whole connection even while
        # silent. The watchdog closes the session after
        # ``idle.disconnect_after_idle_ms`` of no recognized speech and no agent
        # activity. ``IdleWatchdog.last_activity_monotonic`` is refreshed by
        # ``_mark_activity()`` on real ASR text and on agent thinking/speaking;
        # raw VAD/noise (which yields empty ASR) deliberately does NOT count, so
        # a silent-but-noisy room still disconnects. <=0 disables the watchdog.
        # Idle window + teardown reason are chosen by
        # session_intent (plan §3.2/§3.3): a proactive wake-up nobody answers is
        # reclaimed on a short window with reason=proactive_done; a user session
        # keeps the normal window and idle_normal_end.
        idle_policy = resolve_idle_policy(
            session_intent=session_intent, idle_config=self._turn_policy.idle
        )
        self._idle_timeout_sec: float = idle_policy.timeout_sec
        self._idle_end_reason = idle_policy.end_reason
        # Called when the idle timeout fires — deletes the room so the
        # still-connected client is actively disconnected (see server.py).
        self._on_idle_disconnect = on_idle_disconnect
        self._on_session_started = on_session_started
        # Shared session_end{reason} publisher (server.py owns idempotency + reason
        # taxonomy). The watchdog routes idle_normal_end through it; other teardown
        # paths (user_left/error/superseded) call it directly.
        self._on_session_end = on_session_end
        # Called once the AgentSession closes (device left / error) to delete the
        # room PROMPTLY, before the slow STT/TTS shutdown drain — so a rapid
        # re-JOIN of this fixed-name room gets a fresh room (no stale agent/track).
        self._on_session_closed = on_session_closed
        # Grace between notifying the client and deleting the room, so the
        # reliable data packet reaches the client before it is kicked.
        self._idle_disconnect_grace_sec: float = self._turn_policy.idle.disconnect_grace_ms / 1000.0
        self._idle_watchdog_controller = build_full_duplex_idle_watchdog(self)

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
        # primary interrupt mechanism is the DuckingMixer + early-resume watcher
        # (see ``FullDuplexOutputFlow`` and SemanticInterruptHandler).
        # The soft-interrupt path here is kept as a fallback for the rare
        # cases where EOT signals a cut but the mixer isn't installed
        # (e.g. duck_enabled=False, or audio output sink not yet attached).
        # Read timeout from turn policy (can be overridden in tests via
        # ``_soft_interrupt_timeout``). The controller itself lives in
        # ``FullDuplexInterruptionEffects``.

        # DuckingMixer + suspend-window timeout fallback task. Both lazy.
        # Mixer is installed in ``run()`` after ``session.start()`` returns,
        # so the AudioOutput chain is fully assembled. The timeout task is
        # started on every VAD-start (user_state listening → speaking) and
        # cancelled the moment EOT decides to cancel/unduck.

        # Filler word injection for latency masking.
        eot_cfg = self._get_eot_model()._config
        self._filler: FillerManager | None = None
        if eot_cfg.filler_enabled:
            self._filler = FillerManager(
                self._factory.tts,
                phrases=list(eot_cfg.filler_phrases),
                fade_in_ms=eot_cfg.filler_fade_in_ms,
                fade_out_ms=eot_cfg.filler_fade_out_ms,
                silence_lead_in_ms=eot_cfg.filler_silence_lead_in_ms,
            )

        self._context_ledger = self._build_context_ledger()

        # Eagerly trigger EOT model loading so the ONNX session is ready before
        # the first user audio frame arrives. This avoids cold-start delay after
        # session.start() is called.
        get_shared_eot_model(self._turn_policy)
        self._install_provider_observers()

    def _get_eot_model(self) -> Any:
        """Return the shared EOT model instance."""
        return get_shared_eot_model(self._turn_policy)

    def _ensure_ducking_controller(self) -> None:
        if not hasattr(self, "_ducking"):
            self._ducking = OutputDuckingController()

    def _ensure_output_flow(self) -> FullDuplexOutputFlow:
        if not hasattr(self, "_output_flow"):
            self._output_flow = FullDuplexOutputFlow(self)
        return self._output_flow

    def _pipeline_state_label(self) -> str:
        state = getattr(self, "_state", "unknown")
        return state.name if hasattr(state, "name") else str(state)

    def _set_interrupt_cancel_suppression(
        self,
        active: bool,
        until: float | None = None,
        *,
        reason: str = "",
    ) -> None:
        # Single writer (F1 §1c) for the interrupt-cancel suppression flags.
        # ``until=None`` leaves the residual-commit-suppress window unchanged
        # (callers that only clear the skip flag must not shorten the window).
        self._skip_commit_after_interrupt_cancel = active
        if until is not None:
            self._suppress_commit_after_interrupt_until = until
        self._record_suppression_change(
            "interrupt_cancel",
            active=active,
            reason=reason,
            until=self._suppress_commit_after_interrupt_until,
        )

    def _set_suppress_transcripts_until_next_speech(
        self,
        value: bool,
        *,
        reason: str = "",
    ) -> None:
        # Single writer (F1 §1c) for the transcript-suppression flag, previously
        # mutated directly by several collaborators (owner._x = ...).
        self._suppress_transcripts_until_next_speech = value
        self._record_suppression_change("suppress_transcripts", active=value, reason=reason)

    def _record_suppression_change(
        self,
        flag: str,
        *,
        active: bool,
        reason: str,
        until: float | None = None,
    ) -> None:
        # Observability for turn-suppression state changes: who flipped what and
        # why. Never affects behavior. Serves F1 (single, observable authority).
        logger.debug("[StreamingPipeline] suppression %s=%s reason=%s", flag, active, reason)
        timeline = getattr(self, "_timeline", None)
        if timeline is None:
            return
        entry: dict[str, object] = {"flag": flag, "active": active, "reason": reason}
        if until is not None:
            entry["until"] = until
        events = list(timeline.attrs.get("turn_suppression_changes") or ())
        events.append(entry)
        timeline.set_attr("turn_suppression_changes", events[-24:])

    def _claim_full_duplex_irreversible_side_effect(
        self,
        *,
        event: str,
        reason: str,
        timeline: TurnTimeline | None = None,
    ) -> bool:
        timeline = timeline or getattr(self, "_timeline", None)
        turn_id = getattr(timeline, "turn_id", "") if timeline is not None else ""
        if not turn_id:
            return True
        if not hasattr(self, "_full_duplex_irreversible_side_effects"):
            self._full_duplex_irreversible_side_effects = set()
        key = (turn_id, event, reason)
        if key in self._full_duplex_irreversible_side_effects:
            return False
        self._full_duplex_irreversible_side_effects.add(key)
        return True

    def _build_context_ledger(self) -> FullDuplexContextLedger:
        return FullDuplexContextLedger(
            get_session=lambda: getattr(self, "_session", None),
            get_factory=lambda: getattr(self, "_factory", None),
            get_duck_mixer=lambda: getattr(
                getattr(self, "_ducking", None),
                "mixer",
                None,
            ),
            get_config=lambda: self._get_eot_model()._config,
            get_timeline=lambda: (
                self._active_agent_output_timeline() or getattr(self, "_timeline", None)
            ),
            get_assistant_text=self._current_assistant_speech_text,
        )

    def _ensure_context_ledger(self) -> FullDuplexContextLedger:
        if not hasattr(self, "_context_ledger"):
            self._context_ledger = self._build_context_ledger()
        return self._context_ledger

    def _build_interruption_effects(self) -> FullDuplexInterruptionEffects:
        return FullDuplexInterruptionEffects(
            ducking=self._ducking,
            callbacks=self._callbacks,
            get_session=lambda: getattr(self, "_session", None),
            allow_interruptions=lambda: self._allow_interruptions,
            get_eot_model=lambda: self._get_eot_model(),
            get_timeline=lambda: getattr(self, "_timeline", None),
            get_latest_asr_text=lambda: self._latest_asr_text,
            get_state_label=self._pipeline_state_label,
            get_interruption_orchestrator=lambda: self._interruption_orchestrator,
            publish_playback_stop=lambda reason: self._publish_client_control(
                CONTROL_OP_PLAYBACK_STOP,
                reason=reason,
            ),
            finish_agent_output=self._finish_agent_output,
            snapshot_interrupted_context=lambda: self._ensure_context_ledger().snapshot(),
            cancel_residual_commit_suppress_sec=self._cancel_residual_commit_suppress_sec,
            semantic_interrupt_run=lambda text: self._semantic_interrupts.run(
                text,
                is_final=False,
            ),
            correction_topic_stability_window_ms=(
                lambda: self._turn_policy.interrupt.correction_topic_stability_window_ms
            ),
            set_interrupt_cancel_suppression=self._set_interrupt_cancel_suppression,
            soft_interrupt_timeout_sec=lambda: self._soft_interrupt_timeout,
            playback_evidence_active=lambda: (
                self._ensure_client_audio_state_view().agent_output_active_for_interrupts()
            ),
            record_full_duplex_transition=self._record_full_duplex_transition,
            claim_irreversible_side_effect=(self._claim_full_duplex_irreversible_side_effect),
        )

    def _ensure_interruption_effects(self) -> FullDuplexInterruptionEffects:
        if not hasattr(self, "_turn_policy"):
            self._turn_policy = TurnPolicyConfig()
        if not hasattr(self, "_turn_runtime"):
            self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        if not hasattr(self, "_callbacks"):
            self._callbacks = PipelineCallbacks()
        if not hasattr(self, "_allow_interruptions"):
            self._allow_interruptions = True
        if not hasattr(self, "_state"):
            self._state = PipelineState.IDLE
        self._ensure_full_duplex_state_machine()
        self._ensure_ducking_controller()
        if not hasattr(self, "_soft_interrupt_timeout"):
            self._soft_interrupt_timeout = self._turn_runtime.decision_timeout_sec
        handler = getattr(self, "_interruption_effects", None)
        if handler is None or getattr(handler, "_ducking", None) is not self._ducking:
            self._interruption_effects = self._build_interruption_effects()
        return self._interruption_effects

    def _build_decision_effect_applier(self) -> DecisionEffectApplier:
        interruption_effects = self._ensure_interruption_effects()
        return DecisionEffectApplier(
            factory=getattr(self, "_factory", None),
            turn_runtime=self._turn_runtime,
            get_timeline=lambda: self._timeline,
            on_cancel=lambda: interruption_effects.cancel_and_interrupt(),
            on_rollback=lambda reason, drop_buffered: interruption_effects.rollback_if_suspended(
                reason=reason,
                drop_buffered=drop_buffered,
            ),
            on_hold=interruption_effects.handle_hold_decision,
            on_decision=lambda decision, **kwargs: (
                self._interruption_orchestrator.note_turn_policy_decision(
                    decision,
                    **kwargs,
                )
            ),
            record_full_duplex_transition=self._record_policy_decision_transition,
        )

    def _ensure_decision_effect_applier(self) -> None:
        handler = getattr(self, "_decision_effects", None)
        if handler is None or getattr(handler, "_turn_runtime", None) is not self._turn_runtime:
            self._decision_effects = self._build_decision_effect_applier()

    def _build_interruption_orchestrator(self) -> InterruptionOrchestrator:
        interrupt_policy = self._turn_policy.interrupt
        timeout_sec = interrupt_policy.post_speech_evidence_timeout_ms / 1000.0
        min_speech_sec = interrupt_policy.post_speech_evidence_min_speech_ms / 1000.0
        return InterruptionOrchestrator(
            evidence_timeout_sec=timeout_sec,
            min_speech_sec=min_speech_sec,
            no_evidence_timeout_sec=(interrupt_policy.post_speech_no_evidence_timeout_ms / 1000.0),
            on_terminal_verdict=self._on_terminal_interruption_verdict,
        )

    def _on_terminal_interruption_verdict(self, verdict: InterruptionVerdict) -> None:
        """Close a transcript-less false interruption at its acoustic boundary.

        A terminal resume verdict owns exactly one acoustic generation.  With no
        transcript there is nothing a later framework callback can legitimately
        commit, so retaining the product candidate would only allow unrelated
        speech to inherit this verdict.
        """

        if verdict.transcript or verdict.continue_to_llm:
            return
        if verdict.action not in {
            InterruptionVerdictAction.EXPIRED_RESUME,
            InterruptionVerdictAction.REJECTED_RESUME,
        }:
            return
        self._ensure_user_turn_coordinator()
        candidate = self._user_turns.active
        if (
            candidate is None
            or candidate.state != "open"
            or candidate.candidate_id != verdict.candidate_id
            or self._user_turns.current_generation_id != verdict.generation_id
        ):
            return

        reason = f"interruption_no_transcript_terminal:{verdict.action.value}:{verdict.reason}"
        decision = self._user_turns.reject_active(reason)
        timeline = candidate.timeline
        if decision.action == "reject":
            self._record_full_duplex_transition(
                FullDuplexPhase.USER_TURN_REJECTED,
                event="interruption_no_transcript_terminal",
                reason=reason,
                side_effect="irreversible",
                timeline=timeline,
                details={"acoustic_generation": verdict.generation_id},
            )
        turn_completion = self._ensure_turn_completion()
        turn_completion.cancel_completed_voiceprint_turn()
        turn_completion.reset_candidate_voiceprint_tasks()
        self._flush_turn_timeline(
            timeline,
            f"interruption_{verdict.action.value}_no_transcript",
        )

    def _ensure_interruption_orchestrator(self) -> None:
        if not hasattr(self, "_interruption_orchestrator"):
            self._interruption_orchestrator = self._build_interruption_orchestrator()

    def _build_attention_effect_handler(self) -> AttentionEffectHandler:
        interruption_effects = self._ensure_interruption_effects()
        return AttentionEffectHandler(
            turn_policy=self._turn_policy,
            turn_runtime=self._turn_runtime,
            get_agent_speaking=lambda: (
                self._ensure_client_audio_state_view().agent_output_active_for_interrupts()
            ),
            get_duck_active=lambda: self._ducking.is_suspended,
            latest_client_audio_state=lambda participant_identity: (
                self._ensure_client_audio_state_view().latest_state(
                    participant_identity=participant_identity,
                )
            ),
            get_timeline=lambda: self._timeline,
            on_duck=lambda: self._ensure_output_flow().duck_and_arm_timeout(),
            on_interrupt=lambda: interruption_effects.interrupt_current_turn(),
            get_eot_score=lambda: self._get_eot_model().current_eot_score,
        )

    def _ensure_attention_effect_handler(self) -> None:
        handler = getattr(self, "_attention_effects", None)
        if (
            handler is None
            or getattr(handler, "_turn_policy", None) is not self._turn_policy
            or getattr(handler, "_turn_runtime", None) is not self._turn_runtime
        ):
            self._attention_effects = self._build_attention_effect_handler()

    def _ensure_full_duplex_state_machine(self) -> None:
        if not hasattr(self, "_full_duplex_state"):
            self._full_duplex_state = FullDuplexStateMachine()

    def _record_full_duplex_transition(
        self,
        phase: FullDuplexPhase,
        *,
        event: str,
        reason: str,
        side_effect: FullDuplexSideEffect = "none",
        transcript: str = "",
        details: dict[str, object] | None = None,
        timeline: TurnTimeline | None = None,
    ) -> None:
        self._ensure_full_duplex_state_machine()
        target_timeline = self._timeline if timeline is None else timeline
        previous_phase = self._full_duplex_state.phase_for(target_timeline).value
        transition = self._full_duplex_state.transition(
            phase,
            event=event,
            reason=reason,
            side_effect=side_effect,
            transcript=transcript,
            details=details,
            timeline=target_timeline,
        )
        self._ensure_turn_event_sink().phase_changed(
            timeline=target_timeline,
            previous_phase=previous_phase,
            phase=phase.value,
            event=event,
            reason=reason,
            side_effect=side_effect,
            occurred_at=transition.at,
            details=details,
        )

    def _record_policy_decision_transition(
        self,
        decision: Any,
        *,
        source: str = "turn_policy",
        transcript: str = "",
        vad_active: bool | None = None,
        eot_score: float | None = None,
    ) -> None:
        action = str(getattr(getattr(decision, "action", None), "value", "") or "")
        reason = str(getattr(decision, "reason", "") or source)
        details: dict[str, object] = {"source": source, "action": action}
        if vad_active is not None:
            details["vad_active"] = vad_active
        if eot_score is not None:
            details["eot_score"] = eot_score
        if action == "cancel":
            self._record_full_duplex_transition(
                FullDuplexPhase.ACCEPTED_INTERRUPTION,
                event="turn_policy_cancel_accepted",
                reason=reason,
                side_effect="irreversible",
                transcript=transcript,
                details=details,
            )
            return
        if action == "rollback":
            self._record_full_duplex_transition(
                FullDuplexPhase.REJECTED_INTERRUPTION,
                event="turn_policy_rollback",
                reason=reason,
                side_effect="reversible",
                transcript=transcript,
                details=details,
            )
            return
        if action == "hold":
            self._record_full_duplex_transition(
                FullDuplexPhase.EVIDENCE_ARBITRATION,
                event="turn_policy_hold",
                reason=reason,
                transcript=transcript,
                details=details,
            )

    def _build_session_signal_bridge(self) -> SessionSignalBridge:
        return SessionSignalBridge(
            factory=getattr(self, "_factory", None),
            get_eot_model=lambda: self._get_eot_model(),
        )

    def _ensure_session_signal_bridge(self) -> None:
        if not hasattr(self, "_session_signals"):
            self._session_signals = self._build_session_signal_bridge()

    def _build_explicit_client_preempt_ledger(self) -> ExplicitClientPreemptLedger:
        return ExplicitClientPreemptLedger(
            get_timeline=lambda: getattr(self, "_timeline", None),
            get_turn_runtime=lambda: self._turn_runtime,
            get_decision_effects=lambda: self._decision_effects,
            ensure_decision_effects=self._ensure_decision_effect_applier,
        )

    def _ensure_explicit_client_preempt_ledger(self) -> None:
        if not hasattr(self, "_explicit_preempts"):
            self._explicit_preempts = self._build_explicit_client_preempt_ledger()

    def _build_client_audio_state_view(self) -> FullDuplexClientAudioStateView:
        return FullDuplexClientAudioStateView(
            ensure_runtime_defaults=self._ensure_runtime_defaults,
            ensure_room_data=self._ensure_room_data_handler,
            ensure_ducking=self._ensure_ducking_controller,
            get_room_data=lambda: self._room_data,
            get_turn_policy=lambda: self._turn_policy,
            get_pipeline_state=lambda: getattr(self, "_state", PipelineState.IDLE),
            get_ducking=lambda: self._ducking,
        )

    def _ensure_client_audio_state_view(self) -> FullDuplexClientAudioStateView:
        if not hasattr(self, "_client_audio_state"):
            self._client_audio_state = self._build_client_audio_state_view()
        return self._client_audio_state

    def _build_room_data_bridge(self) -> FullDuplexRoomDataBridge:
        return FullDuplexRoomDataBridge(
            ensure_room_data=self._ensure_room_data_handler,
            ensure_client_preempts=self._ensure_client_preempt_handler,
            get_room_data=lambda: self._room_data,
            get_client_preempts=lambda: self._client_preempts,
        )

    def _ensure_room_data_bridge(self) -> FullDuplexRoomDataBridge:
        if not hasattr(self, "_room_data_bridge"):
            self._room_data_bridge = self._build_room_data_bridge()
        return self._room_data_bridge

    def _build_client_preempt_handler(self) -> ExplicitClientPreemptHandler:
        interruption_effects = self._ensure_interruption_effects()
        return ExplicitClientPreemptHandler(
            latest_client_audio_state=lambda participant_identity=None: (
                self._ensure_client_audio_state_view().latest_state(
                    participant_identity=participant_identity,
                )
            ),
            agent_output_active_for_interrupts=lambda participant_identity=None: (
                self._ensure_client_audio_state_view().agent_output_active_for_interrupts(
                    participant_identity=participant_identity,
                )
            ),
            ensure_ducking_controller=self._ensure_ducking_controller,
            is_output_cancelled=lambda: self._ducking.is_cancelled,
            record_explicit_client_preempt=(
                lambda state_attr, received_at: self._record_explicit_client_preempt_decision(
                    state_attr=state_attr,
                    received_at=received_at,
                )
            ),
            mark_explicit_client_preempt_resolved=(self._mark_explicit_client_preempt_resolved),
            cancel_agent_output=lambda force: interruption_effects.cancel_and_interrupt(
                force=force,
            ),
            agent_turn_active_for_explicit_preempt=lambda participant_identity=None: (
                self._agent_turn_active_for_explicit_preempt(
                    participant_identity=participant_identity,
                )
            ),
            preempt_agent_turn_for_explicit_control=(self._preempt_agent_turn_for_explicit_control),
        )

    def _ensure_client_preempt_handler(self) -> None:
        if not hasattr(self, "_client_preempts"):
            self._client_preempts = self._build_client_preempt_handler()

    def _build_transcript_echo_gate(self) -> TranscriptEchoGate:
        return TranscriptEchoGate(
            get_agent_text=self._current_assistant_speech_text,
            min_normalized_chars=self._turn_policy.attention.echo_min_normalized_chars,
        )

    def _ensure_transcript_echo_gate(self) -> TranscriptEchoGate:
        if not hasattr(self, "_transcript_echo_gate"):
            self._transcript_echo_gate = self._build_transcript_echo_gate()
        return self._transcript_echo_gate

    def _build_transcript_admission_gate(self) -> TranscriptAdmissionGate:
        return TranscriptAdmissionGate(
            suppress_until_next_speech=(lambda: self._suppress_transcripts_until_next_speech),
            agent_output_active=lambda speaker_id: (
                self._ensure_client_audio_state_view().agent_output_active_for_interrupts(
                    participant_identity=speaker_id,
                )
            ),
            echo_gate=lambda: self._ensure_transcript_echo_gate(),
            absorb_committed_turn_revision=(self._absorb_committed_turn_transcript_revision),
        )

    def _ensure_transcript_admission_gate(self) -> TranscriptAdmissionGate:
        if not hasattr(self, "_transcript_admission"):
            self._transcript_admission = self._build_transcript_admission_gate()
        return self._transcript_admission

    def _reject_agent_echo_transcript(self, transcript: str) -> None:
        self._set_suppress_transcripts_until_next_speech(True, reason="agent_echo")
        if self._timeline is not None:
            self._timeline.set_attr(
                "agent_echo_suppressed",
                {
                    "text_preview": transcript[:120],
                    "text_length": len(transcript),
                },
            )
        # The suppress flag above is shared (echo protection applies to any
        # open-mic-during-playback session). The duck rollback below is barge-in
        # only; half_duplex never ducks, so it is a no-op there anyway — gate it
        # on the single barge-in authority so no future path revives it.
        if self._barge_in_enabled:
            self._ensure_interruption_effects().rollback_if_suspended(
                reason="agent_echo",
                drop_buffered=False,
            )

    def _absorb_committed_turn_transcript_revision(
        self,
        transcript: str,
        is_final: bool,
    ) -> bool:
        self._ensure_user_turn_coordinator()
        return self._user_turns.absorb_committed_transcript_revision(
            transcript,
            is_final=is_final,
            require_framework_completed=True,
        )

    def _build_transcript_handler(self) -> FullDuplexTranscriptHandler:
        return FullDuplexTranscriptHandler(
            admission_gate=self._ensure_transcript_admission_gate,
            record_accepted_event=self._ensure_transcript_recorder().record,
            allow_interruptions=lambda: self._allow_interruptions,
            native_adaptive_owner=self._uses_livekit_native_adaptive_interruption,
            agent_output_active=lambda speaker_id: (
                self._ensure_client_audio_state_view().agent_output_active_for_interrupts(
                    participant_identity=speaker_id,
                )
            ),
            interrupt_window_active=self._interrupt_window_active,
            decision_suppressed=self._interrupt_decision_suppressed,
            attention_allows_eot_check=lambda transcript, speaker_id: (
                self._attention_effects.allows_eot_check(
                    transcript,
                    speaker_id=speaker_id,
                )
            ),
            run_semantic_interrupt=lambda transcript, is_final: self._semantic_interrupts.run(
                transcript, is_final=is_final
            ),
            reject_agent_echo=self._reject_agent_echo_transcript,
            forward_to_base=lambda event: BasePipeline._on_user_transcribed(self, event),
            record_transcript_ingress_event=self._record_transcript_ingress_event,
            record_transcript_admission_event=self._record_transcript_admission_event,
            record_semantic_gate_event=self._record_semantic_gate_event,
            warm_preemptive=self._warm_preemptive_from_partial,
        )

    def _record_transcript_ingress_event(self, payload: dict[str, object]) -> None:
        timeline = getattr(self, "_timeline", None)
        event = self._ensure_transcript_ingress_ledger().record(
            payload,
            timeline=timeline,
            user_speaking_active=getattr(self, "_user_speaking_start_time", None) is not None,
            pipeline_state=getattr(self, "_state", None),
        )
        if self._timeline is None:
            return
        events = list(self._timeline.attrs.get("transcript_ingress_events") or ())
        events.append(dict(event))
        self._timeline.set_attr("transcript_ingress_events", events[-16:])
        self._timeline.set_attr("transcript_ingress_last_event", dict(event))

    def _attach_transcript_ingress_recent_events(self, reason: str) -> None:
        self._ensure_transcript_ingress_ledger().attach_to_timeline(
            getattr(self, "_timeline", None),
            reason=reason,
        )

    def _ensure_transcript_ingress_ledger(self) -> FullDuplexTranscriptIngressLedger:
        if not hasattr(self, "_transcript_ingress_ledger"):
            self._transcript_ingress_ledger = FullDuplexTranscriptIngressLedger()
        return self._transcript_ingress_ledger

    def _record_transcript_admission_event(self, payload: dict[str, object]) -> None:
        if self._timeline is None:
            return
        events = list(self._timeline.attrs.get("transcript_admission_events") or ())
        events.append(dict(payload))
        self._timeline.set_attr("transcript_admission_events", events[-16:])
        self._timeline.set_attr("transcript_admission_last_event", dict(payload))

    def _record_semantic_gate_event(self, payload: dict[str, object]) -> None:
        if self._timeline is None:
            return
        events = list(self._timeline.attrs.get("semantic_interrupt_gate_events") or ())
        events.append(dict(payload))
        self._timeline.set_attr("semantic_interrupt_gate_events", events[-16:])
        self._timeline.set_attr("semantic_interrupt_gate_last_event", dict(payload))

    def _warm_preemptive_from_partial(self, transcript: str) -> None:
        """Fire-and-forget: warm the brain on a stabilizing partial transcript.

        Reaches the eidolon brain LLM adapter (when that's the configured LLM)
        and schedules a *speculative* warm-up turn so the real turn's first
        response lands sooner; a no-op for any other LLM. The real turn
        supersedes it (grpc_llm discards the warm-up at real-turn start). The
        trigger fires per accepted non-final transcript; the warmer bounds cost
        (min length, dedup, single in-flight). Can later be tightened to EOT
        confidence — the warm() API is already the right seam for that.
        """
        warm = getattr(getattr(getattr(self._factory, "llm", None), "llm", None), "warm", None)
        if warm is None:
            return
        try:
            asyncio.ensure_future(warm(transcript))
        except Exception:  # noqa: BLE001 — warming must never disturb the turn
            logger.debug("[StreamingPipeline] preemptive warm scheduling failed", exc_info=True)

    def _ensure_transcript_handler(self) -> FullDuplexTranscriptHandler:
        if not hasattr(self, "_transcript_handler"):
            self._transcript_handler = self._build_transcript_handler()
        return self._transcript_handler

    def _ensure_transcript_recorder(self) -> FullDuplexTranscriptRecorder:
        if not hasattr(self, "_transcript_recorder"):
            self._transcript_recorder = FullDuplexTranscriptRecorder(
                self,
                transcript_revision_min_normalized_chars=(
                    self._turn_policy.eot.transcript_revision_min_normalized_chars
                ),
            )
        return self._transcript_recorder

    def _ensure_transcript_evidence_buffer(self) -> TranscriptEvidenceBuffer:
        if not hasattr(self, "_transcript_evidence_buffer"):
            self._transcript_evidence_buffer = TranscriptEvidenceBuffer()
        return self._transcript_evidence_buffer

    def _observe_stt_speech_event(self, event: Any) -> None:
        """Observe a public LiveKit SpeechEvent before framework normalization."""

        self._ensure_transcript_evidence_buffer().observe_speech_event(event)

    def _build_speech_lifecycle(self) -> FullDuplexSpeechLifecycle:
        return FullDuplexSpeechLifecycle(self)

    def _ensure_speech_lifecycle(self) -> FullDuplexSpeechLifecycle:
        if not hasattr(self, "_speech_lifecycle"):
            self._speech_lifecycle = self._build_speech_lifecycle()
        return self._speech_lifecycle

    def _build_user_state_handler(self) -> FullDuplexUserStateHandler:
        speech_lifecycle = self._ensure_speech_lifecycle()
        return FullDuplexUserStateHandler(
            publish_companion_ui_state=self._publish_companion_ui_state,
            signal_stt_user_away=self._session_signals.signal_stt_user_away,
            signal_stt_user_present=self._session_signals.signal_stt_user_present,
            handle_speaking_started=speech_lifecycle.handle_started,
            handle_speaking_stopped=speech_lifecycle.handle_stopped,
        )

    def _ensure_user_state_handler(self) -> FullDuplexUserStateHandler:
        if not hasattr(self, "_user_state_handler"):
            self._user_state_handler = self._build_user_state_handler()
        return self._user_state_handler

    def _cancel_residual_commit_suppress_sec(self) -> float:
        return max(self._turn_policy.interrupt.cancel_residual_commit_suppress_ms, 0) / 1000.0

    def _build_user_turn_coordinator(self) -> UserTurnCoordinator:
        return UserTurnCoordinator(
            transcript_revision_min_normalized_chars=(
                self._turn_policy.eot.transcript_revision_min_normalized_chars
            ),
            speech_merge_grace_sec=(self._turn_policy.eot.speech_merge_grace_ms / 1000.0),
        )

    def _ensure_user_turn_coordinator(self) -> None:
        if not hasattr(self, "_user_turns"):
            self._user_turns = self._build_user_turn_coordinator()

    def _ensure_turn_completion(self) -> FullDuplexTurnCompletion:
        if not hasattr(self, "_turn_completion"):
            self._turn_completion = FullDuplexTurnCompletion(self)
        return self._turn_completion

    def _flush_turn_timeline(
        self,
        timeline: TurnTimeline | None,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        flushed_ids = getattr(self, "_flushed_timeline_ids", None)
        if flushed_ids is None:
            flushed_ids = self._flushed_timeline_ids = set()
        if timeline.turn_id in flushed_ids:
            return
        flushed_ids.add(timeline.turn_id)
        timeline.set_attr("timeline_flush_reason", reason)
        self._ensure_turn_event_sink().terminal(timeline, reason)
        timeline.append_debug_jsonl(self._observability.timeline_debug_path)
        self._ensure_agent_output_coordinator().release(timeline)
        if timeline is self._timeline:
            self._timeline_debug_flushed = True
            self._timeline = None

    def _append_turn_timeline_snapshot(
        self,
        timeline: TurnTimeline | None,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        timeline.set_attr("timeline_snapshot_reason", reason)
        timeline.set_attr("timeline_flush_reason", reason)
        phase = str(((timeline.attrs.get("full_duplex_state") or {}).get("phase")) or "")
        if phase == FullDuplexPhase.USER_TURN_REJECTED.value:
            self._ensure_turn_event_sink().terminal(timeline, reason)
        timeline.append_debug_jsonl(self._observability.timeline_debug_path)

    def _build_agent_state_effect_handler(self) -> AgentStateEffectHandler:
        interruption_effects = self._ensure_interruption_effects()
        return AgentStateEffectHandler(
            get_timeline=self._active_agent_output_timeline,
            mark_activity=lambda: self._mark_activity(),
            cancel_soft_interrupt=lambda: interruption_effects.cancel_soft_interrupt(),
            soft_interrupt_active=lambda: interruption_effects.soft_interrupt_active(),
            ducking=self._ducking,
            get_filler=lambda: self._filler,
            flush_timeline_debug=lambda reason, clear: self._finish_agent_output(reason),
            should_flush_on_playback_done=self._timeline_turn_terminal_for_playback_flush,
            agent_output=self._ensure_agent_output_coordinator(),
            on_playback_started=self._assistant_speech_playback_started,
            on_playback_finished=self._assistant_speech_playback_finished,
        )

    def _ensure_agent_state_effect_handler(self) -> None:
        handler = getattr(self, "_agent_state_effects", None)
        if handler is None or getattr(handler, "_ducking", None) is not self._ducking:
            self._agent_state_effects = self._build_agent_state_effect_handler()

    def _timeline_turn_terminal_for_playback_flush(self) -> bool:
        return self._active_agent_output_timeline() is not None

    def _ensure_agent_output_coordinator(self) -> AgentOutputCoordinator:
        coordinator = getattr(self, "_agent_output", None)
        if coordinator is None:
            coordinator = self._agent_output = AgentOutputCoordinator()
        return coordinator

    def _active_agent_output_timeline(self) -> TurnTimeline | None:
        return self._ensure_agent_output_coordinator().active_timeline

    def _claim_agent_output_timeline(self, timeline: TurnTimeline | None) -> None:
        if timeline is None:
            return
        displaced = self._ensure_agent_output_coordinator().claim(timeline)
        if displaced is not None:
            self._flush_turn_timeline(displaced, "agent_output_superseded")

    def _finish_agent_output(self, reason: str) -> None:
        timeline = self._active_agent_output_timeline()
        if timeline is not None:
            self._flush_turn_timeline(timeline, reason)

    def _build_semantic_interrupt_handler(self) -> SemanticInterruptHandler:
        interruption_effects = self._ensure_interruption_effects()
        return SemanticInterruptHandler(
            get_eot_model=lambda: self._get_eot_model(),
            turn_runtime=self._turn_runtime,
            get_timeline=lambda: self._timeline,
            get_duck_active=lambda: self._ducking.is_suspended,
            get_duck_stats=lambda: self._ducking.stats(),
            get_vad_active=lambda: (
                self._session is not None and self._session.user_state == "speaking"
            ),
            soft_interrupt_active=lambda: interruption_effects.soft_interrupt_active(),
            soft_interrupt_timeout=lambda: self._soft_interrupt_timeout,
            apply_decision=self._decision_effects.apply,
            record_decision_attrs=self._decision_effects.record_decision_attrs,
            publish_turn_control=self._decision_effects.publish_turn_control,
            cancel_duck_and_interrupt=lambda: interruption_effects.cancel_and_interrupt(),
            interrupt_current_turn=lambda: interruption_effects.interrupt_current_turn(),
            enter_soft_interrupt=lambda: interruption_effects.enter_soft_interrupt(),
            decide_from_transcript=(
                lambda text, score, **kwargs: (
                    self._interruption_orchestrator.decide_from_transcript(
                        self._turn_runtime,
                        text,
                        score,
                        **kwargs,
                    )
                )
            ),
        )

    def _ensure_semantic_interrupt_handler(self) -> None:
        handler = getattr(self, "_semantic_interrupts", None)
        if handler is None or getattr(handler, "_turn_runtime", None) is not self._turn_runtime:
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
                self._session is not None and self._session.user_state == "speaking"
            ),
            get_eot_model=lambda: self._get_eot_model(),
            apply_decision=self._decision_effects.apply,
            should_hold_for_evidence=(
                lambda: self._interruption_orchestrator.should_hold_deadline()
            ),
            get_hold_remaining_sec=(
                lambda: self._interruption_orchestrator.hold_remaining_sec()
            ),
        )

    def _ensure_duck_suspend_timeout_handler(self) -> None:
        handler = getattr(self, "_duck_deadline", None)
        if handler is None or getattr(handler, "_turn_runtime", None) is not self._turn_runtime:
            self._duck_deadline = self._build_duck_suspend_timeout_handler()

    def _install_provider_observers(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_all()

    def _build_provider_event_observer(self) -> ProviderEventObserver:
        return ProviderEventObserver(
            factory=self._factory,
            get_timeline=lambda: self._timeline,
            get_output_timeline=self._active_agent_output_timeline,
            agent_output=self._ensure_agent_output_coordinator(),
            flush_timeline=lambda timeline, reason: self._flush_turn_timeline(
                timeline,
                reason,
            ),
            append_timeline_snapshot=lambda timeline, reason: self._append_turn_timeline_snapshot(
                timeline, reason
            ),
            publish_milestone=lambda timeline, milestone, reason: (
                self._ensure_turn_event_sink().milestone(timeline, milestone, reason=reason)
            ),
            first_delta_timeout_sec=(self._observability.llm_first_delta_timeout_ms / 1000.0),
            stt_pending_event_window_sec=(
                self._observability.stt_pending_provider_event_window_ms / 1000.0
            ),
            stt_pending_event_preroll_sec=(
                self._observability.stt_pending_provider_event_preroll_ms / 1000.0
            ),
            stt_pending_event_max_count=(self._observability.stt_pending_provider_event_max_count),
        )

    def _ensure_provider_event_observer(self) -> None:
        if not hasattr(self, "_provider_events"):
            if not hasattr(self, "_observability"):
                self._observability = ObservabilityConfig()
            if not hasattr(self, "_timeline"):
                self._timeline = None
            self._provider_events = self._build_provider_event_observer()

    def _ensure_turn_event_sink(self) -> ChannelTurnEventSink:
        """Return the observer, including for focused tests that skip __init__."""

        if not hasattr(self, "_turn_events"):
            self._turn_events = ChannelTurnEventSink()
        return self._turn_events

    def _ensure_runtime_defaults(self) -> None:
        ensure_full_duplex_runtime_defaults(self)

    def _record_client_control_event(
        self,
        *,
        timeline: TurnTimeline | None,
        op: str,
        reason: str,
        turn_id: str,
    ) -> None:
        self._ensure_client_control_publisher().record_event(
            timeline=timeline,
            op=op,
            reason=reason,
            turn_id=turn_id,
        )

    def _build_turn_handling(self) -> dict:
        """AgentSession ``turn_handling`` options (Round 8 R8.9: must live on the
        AgentSession, not the Agent — ``AgentActivity`` reads
        ``session._opts.turn_handling.interruption``).

        ``StreamingPipeline`` is the full-duplex realtime path. Half-duplex
        sessions are routed to ``HalfDuplexPttPipeline`` before this class is
        constructed.

        ``preemptive_generation`` (Phase 2, 2026-05-30): speculative brain
        generation gated via ``turn_policy.preemptive`` — hides the STT-final
        wait by starting the brain on a stable interim; framework reuses it if
        the final matches, else cancels via gRPC CancelTurn. ``preemptive_tts``
        stays gated by our commit so no partial audio leaks.
        """
        return build_full_duplex_turn_handling(
            turn_policy=self._turn_policy,
            allow_interruptions=self._allow_interruptions,
            false_interruption_timeout=self._false_interruption_timeout,
            avatar_mode=self._avatar_enabled,
        )

    @property
    def _barge_in_enabled(self) -> bool:
        """Single authority for whether this session runs barge-in.

        ``full_duplex`` → True; ``half_duplex`` → False (carried by
        ``allow_interruptions``, the ONLY runtime signal that distinguishes the
        two modes). Every barge-in touch point must consult THIS — never
        ``attention.enabled`` / ``attention.enforce`` /
        ``_uses_livekit_native_adaptive_interruption`` (none of which separate
        the modes: enforce is identical in both, and the native-adaptive gate is
        False in both). ``ptt`` never reaches StreamingPipeline (server routes it
        to HalfDuplexPttPipeline), so only full/half are in scope here.
        """
        return bool(getattr(self, "_allow_interruptions", True))

    def _uses_livekit_native_adaptive_interruption(self) -> bool:
        return uses_livekit_native_adaptive_interruption(
            turn_policy=getattr(self, "_turn_policy", None),
            allow_interruptions=bool(getattr(self, "_allow_interruptions", False)),
        )

    async def run(self, room: Room) -> None:
        """Start the full-duplex pipeline. Blocks until AgentSession closes."""
        await self._ensure_lifecycle().run(room)

    async def shutdown(self) -> None:
        """Gracefully shut down the full-duplex pipeline."""
        await self._ensure_lifecycle().shutdown()

    def _ensure_lifecycle(self) -> FullDuplexSessionLifecycle:
        if not hasattr(self, "_lifecycle"):
            self._lifecycle = FullDuplexSessionLifecycle(self)
        return self._lifecycle

    def _ensure_client_control_publisher(self) -> FullDuplexClientControlPublisher:
        if not hasattr(self, "_client_control_publisher"):
            self._client_control_publisher = FullDuplexClientControlPublisher(self)
        return self._client_control_publisher

    def _record_explicit_client_preempt_decision(
        self,
        *,
        state_attr: dict[str, Any],
        received_at: float,
        resolved_at: float | None = None,
        timeline: TurnTimeline | None = None,
    ) -> None:
        self._ensure_explicit_client_preempt_ledger()
        timeline = timeline or getattr(self, "_timeline", None)
        if timeline is None:
            timeline = self._create_explicit_preempt_control_timeline(received_at)
        self._explicit_preempts.record(
            state_attr=state_attr,
            received_at=received_at,
            resolved_at=resolved_at,
            timeline=timeline,
        )

    def _apply_pending_explicit_client_preempt(
        self,
        timeline: TurnTimeline | None = None,
    ) -> None:
        self._ensure_explicit_client_preempt_ledger()
        self._explicit_preempts.apply_pending(timeline)

    def _apply_pending_client_control_events(
        self,
        timeline: TurnTimeline | None = None,
    ) -> None:
        self._ensure_client_control_publisher().apply_pending(timeline)

    def _mark_explicit_client_preempt_resolved(
        self,
        received_at: float,
        resolved_at: float,
    ) -> None:
        self._ensure_explicit_client_preempt_ledger()
        self._explicit_preempts.mark_resolved(received_at, resolved_at)
        timeline = getattr(self, "_explicit_preempt_control_timeline", None)
        if timeline is None:
            return
        timeline.mark_interrupt_resolved("cancel", timestamp=resolved_at)
        timeline.set_attr("cancel_reason", "explicit_client_ptt")
        self._flush_turn_timeline(
            timeline,
            "explicit_client_preempt_control_only",
        )
        self._explicit_preempt_control_timeline = None

    def _create_explicit_preempt_control_timeline(
        self,
        received_at: float,
    ) -> TurnTimeline:
        timeline = TurnTimeline(generate_turn_id())
        timeline.mark_at("interrupt_started_at", received_at)
        timeline.set_attr("control_only", True)
        timeline.set_attr("control_only_reason", "explicit_client_ptt")
        room = getattr(self, "_room", None)
        if room is not None:
            timeline.set_attr("room_name", getattr(room, "name", "") or "")
        self._timeline = timeline
        self._timeline_debug_flushed = False
        self._explicit_preempt_control_timeline = timeline
        return timeline

    def _turn_detection(self) -> Any:
        """The full-duplex Agent ``turn_detection`` model."""
        return self._get_eot_model()

    def _welcome_on_enter_text(self) -> str | None:
        """Welcome line to speak on session start, or None to stay silent.

        A proactive_initiated session was woken to deliver a report — that
        report IS the opening, so the canned welcome is suppressed. User and
        verified-presence sessions keep their welcome (None when unconfigured
        means wait for the user to speak first).
        """
        return resolve_welcome_text(
            session_intent=self._session_intent,
            welcome_message=self._welcome_message,
        )

    def _current_assistant_speech_text(self) -> str:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        return self._assistant_speech.current_or_recent_text(
            factory=getattr(self, "_factory", None),
            max_age_ms=self._turn_policy.attention.assistant_speech_recent_max_age_ms,
        )

    def _record_assistant_speech_text(self, text: str, *, source: str) -> None:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        self._assistant_speech.record(text, source=source)

    def _begin_streamed_assistant_speech(self) -> int:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        return self._assistant_speech.begin_streamed_speech()

    def _append_streamed_assistant_speech(self, stream_id: int, text: str) -> None:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        self._assistant_speech.append_streamed_speech(stream_id, text)

    def _abort_streamed_assistant_speech(self, stream_id: int) -> None:
        if not hasattr(self, "_assistant_speech"):
            return
        self._assistant_speech.abort_streamed_speech(stream_id)

    def _queue_fixed_assistant_speech(self, text: str, *, source: str) -> None:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        self._assistant_speech.queue_fixed_speech(text, source=source)

    def _assistant_speech_playback_started(self) -> None:
        if not hasattr(self, "_assistant_speech"):
            self._assistant_speech = AssistantSpeechLedger()
        self._assistant_speech.on_playback_started()

    def _assistant_speech_playback_finished(self) -> None:
        if not hasattr(self, "_assistant_speech"):
            return
        self._assistant_speech.on_playback_finished()

    def _build_agent(self) -> lk_Agent:
        """Build the LiveKit Agent."""
        return build_full_duplex_agent(self)

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

    def _start_idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        self._idle_watchdog_controller.start()

    def _stop_idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        self._idle_watchdog_controller.stop()

    async def _idle_watchdog(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.run()

    async def _disconnect_idle(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.disconnect_idle()

    async def _notify_client_idle_timeout(self) -> None:
        self._ensure_idle_watchdog_controller()
        await self._idle_watchdog_controller.notify_client_idle_timeout()

    def _ensure_idle_watchdog_controller(self) -> None:
        ensure_full_duplex_idle_watchdog(self)

    def _ensure_room_data_handler(self) -> None:
        if not hasattr(self, "_room_data"):
            self._room_data = RoomDataHandler(
                get_timeline=lambda: getattr(self, "_timeline", None),
            )

    def _publish_companion_ui_state(self, state: str, reason: str) -> None:
        self._ensure_client_control_publisher().publish_companion_ui_state(
            state,
            reason,
        )

    def _publish_client_control(
        self,
        op: str,
        *,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._ensure_client_control_publisher().publish_client_control(
            op=op,
            reason=reason,
            payload=payload,
        )

    def _on_agent_state_changed(self, event: Any) -> None:
        """Forward agent state changes through BasePipeline and session effects."""
        self._ensure_runtime_defaults()
        timeline = self._active_agent_output_timeline()
        super()._on_agent_state_changed(event)
        self._ensure_agent_state_effect_handler()
        self._agent_state_effects.handle(event)
        new = getattr(event, "new_state", "")
        old = getattr(event, "old_state", "")
        if new == "thinking":
            self._ensure_turn_event_sink().milestone(
                timeline, "generating", reason="agent_thinking"
            )
        elif new == "speaking":
            self._ensure_turn_event_sink().milestone(
                timeline, "first_audio", reason="agent_speaking"
            )
        elif old == "speaking" and new in {"idle", "listening"}:
            self._ensure_turn_event_sink().milestone(
                timeline, "playback_done", reason="agent_playback_done"
            )
        if new:
            self._ensure_client_control_publisher().publish_companion_ui_state_for_agent_state(new)

    def _on_session_error(self, event: Any) -> None:
        """Close the response turn on a non-recoverable framework/provider error."""

        self._ensure_runtime_defaults()
        super()._on_session_error(event)
        timeline = self._active_agent_output_timeline()
        if timeline is None:
            return
        error_type = str(getattr(event, "type", "") or "session_error")
        recoverable = bool(getattr(event, "recoverable", False))
        error = getattr(event, "error", event)
        mark = "tts_error_at" if error_type == "tts_error" else "session_error_at"
        timeline.mark(mark)
        timeline.set_attr(
            "output_error",
            {
                "type": error_type,
                "label": str(getattr(event, "label", "") or ""),
                "recoverable": recoverable,
                "error": str(error or "")[:240],
            },
        )
        self._ensure_turn_event_sink().milestone(
            timeline,
            error_type,
            reason=f"livekit_{error_type}",
        )
        if not recoverable:
            self._ensure_turn_completion().notify_silent_output_failure_once(
                timeline=timeline,
                error_type=error_type,
            )
            self._finish_agent_output(f"nonrecoverable_{error_type}")

    def _on_user_state_changed(self, event: Any) -> None:
        self._ensure_runtime_defaults()
        try:
            self._ensure_user_state_handler().handle(event)
        except Exception:
            logger.exception("[StreamingPipeline] error in _on_user_state_changed")

    def _on_user_transcribed(self, event: Any) -> None:
        """Handle user transcription events.

        Full-duplex transcript routing lives in ``FullDuplexTranscriptHandler``.
        The pipeline keeps this method as the LiveKit event boundary.
        """
        self._ensure_runtime_defaults()
        self._ensure_transcript_handler().handle(event)

    def _on_user_transcription_timeout(self, event: Any) -> None:
        """Apply product recovery after LiveKit observes speech with no final text."""

        self._ensure_runtime_defaults()
        self._ensure_turn_completion().handle_transcription_timeout(event)

    def _interrupt_decision_suppressed(self) -> bool:
        """Ignore residual ASR after a confirmed interrupt cancel."""
        return time.monotonic() < self._suppress_commit_after_interrupt_until

    def _interrupt_window_active(self) -> bool:
        """Return true while an actual interrupt decision window is open."""
        return (
            self._ducking.is_suspended
            or self._ensure_interruption_effects().soft_interrupt_active()
        )

    def _agent_turn_active_for_explicit_preempt(
        self,
        *,
        participant_identity: str | None = None,
    ) -> bool:
        """Return true when an explicit client control should preempt output.

        Audible playback uses the normal interrupt path. A still-silent
        generation also needs cancellation so a stale LiveKit speech handle does
        not block the next user turn.
        """

        if self._ensure_client_audio_state_view().agent_output_active_for_interrupts(
            participant_identity=participant_identity,
        ):
            return True
        return getattr(self, "_state", PipelineState.IDLE) in {
            PipelineState.GENERATING,
            PipelineState.SPEAKING,
        }

    def _preempt_agent_turn_for_explicit_control(self) -> None:
        """Apply the explicit client preempt side effect for the active turn.

        Audible output takes the full playback-interrupt path: stop playback,
        capture what the user heard, and cancel the framework speech handle.
        Silent generation takes a narrower path: cancel the generation/speech
        handle and drop any late TTS frames, but do not snapshot interrupted
        context because the user has not heard that assistant content yet.
        """

        if self._ensure_client_audio_state_view().agent_output_active_for_interrupts():
            self._ensure_interruption_effects().cancel_and_interrupt(force=True)
            return
        self._ensure_interruption_effects().cancel_silent_generation_for_explicit_preempt()

    def _append_timeline_debug(self, reason: str, *, clear: bool = False) -> None:
        if self._timeline is None:
            return
        timeline = self._timeline
        self._flush_turn_timeline(timeline, reason)
        if not clear and self._timeline is None:
            self._timeline = timeline
