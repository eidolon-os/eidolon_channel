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
     Eidolon's multi-signal interruption owner. Our soft-interrupt
     path is owned by channel in the default hybrid profile.

  5. **Forces framework's auto-interrupt OFF** via
     ``integration.framework_patches.disable_audio_activity_interruption`` —
     so Eidolon's InterruptionOrchestrator / turn policy is the sole
     authority. See
     ``integration/framework_patches.py`` for the rationale (no public API
     does this without breaking endpointing).

If you find yourself adding logic here that AgentSession already
handles, push back — likely the right shape is to USE the
AgentSession config rather than override.
============================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from livekit.agents.voice import Agent as lk_Agent
    from livekit.agents.voice import AgentSession
    from livekit.rtc import Room

from eidolon_sdk.biz.contracts import (
    COMPANION_UI_STATE_TOPIC,
    CONTROL_OP_PLAYBACK_STOP,
    CONTROL_TOPIC,
    INTERACTION_MODE_FULL_DUPLEX,
    PLAYBACK_STATE_AGENT_SPEAKING,
    SESSION_END_ERROR,
    SESSION_END_IDLE_NORMAL,
    SESSION_END_USER_LEFT,
    SESSION_INTENT_PROACTIVE,
    SESSION_INTENT_USER_INITIATED,
    WIRE_SCHEMA_VERSION,
)
from eidolon.livekit.common.config import (
    ObservabilityConfig,
    TurnPolicyConfig,
    VoiceprintConfig,
)

from ..context import InterruptedContextManager
from ..integration import framework_patches
from ..integration.client_audio_state import ClientAudioState
from ..runtime.interaction_mode import resolve_idle_policy
from ..turn_policy import (
    Decision,
    TranscriptEvidenceGate,
    TurnPolicyRuntime,
)
from ..turn_policy.constants import STABLE_SIGNAL_WAIT_REASON_PREFIX
from ..observability import TurnTimeline
from ..factory import SharedStageFactory
from ..output import FillerManager, OutputDuckingController
from ..pipeline.base import BasePipeline
from ..pipeline.types import PipelineCallbacks, PipelineState, generate_turn_id
from ..session.agent_state import AgentStateEffectHandler
from ..session.attention_effects import AttentionEffectHandler
from ..session.client_control import (
    append_client_control_event,
    build_client_control_event,
    build_session_client_control_envelope,
)
from .client_preempt import (
    ExplicitClientPreemptHandler,
    ExplicitClientPreemptLedger,
)
from .semantic_interrupt_gate import evaluate_semantic_interrupt_gate
from .transcript_admission import TranscriptAdmissionGate
from .transcript_event import FullDuplexTranscriptEvent
from .user_state_event import FullDuplexUserStateEvent
from ..session.decision_effects import DecisionEffectApplier
from ..session.duck_timeout import DuckSuspendTimeoutHandler
from ..session.eot_model import get_shared_eot_model
from ..session.idle import IdleWatchdog
from ..session.interruption import SoftInterruptController
from ..session.interruption_orchestrator import InterruptionOrchestrator
from ..session.messages import message_text
from ..session.provider_events import ProviderEventObserver
from ..session.room_data import RoomDataHandler
from ..session.semantic_interrupt import SemanticInterruptHandler
from ..session.signals import SessionSignalBridge
from ..session.transcript_echo import TranscriptEchoGate
from ..session.turn_commit import UserTurnCommitter
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
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        on_session_closed: Callable[[], Awaitable[None]] | None = None,
        interaction_mode: str = INTERACTION_MODE_FULL_DUPLEX,
        session_intent: str = SESSION_INTENT_USER_INITIATED,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        if interaction_mode != INTERACTION_MODE_FULL_DUPLEX:
            raise ValueError(
                "StreamingPipeline is full-duplex only; use HalfDuplexPttPipeline "
                f"for interaction_mode={interaction_mode!r}"
            )
        self._interaction_mode = INTERACTION_MODE_FULL_DUPLEX
        # Session intent (plan §3.2/§3.3) is orthogonal to interaction_mode and
        # drives only idle window + teardown reason in the full-duplex pipeline.
        self._session_intent = session_intent
        self._is_proactive = session_intent == SESSION_INTENT_PROACTIVE
        self._turn_policy = turn_policy or TurnPolicyConfig()
        self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        self._observability = observability or ObservabilityConfig()
        self._voiceprint_config = voiceprint_config or VoiceprintConfig()
        self._timeline: TurnTimeline | None = None
        self._timeline_debug_flushed = False
        self._pending_client_control_events: list[dict[str, Any]] = []
        self._skip_commit_after_interrupt_cancel = False
        self._suppress_commit_after_interrupt_until = 0.0
        # Ducking state is shared by several effect handlers. It must exist
        # before those handlers are built, because AgentStateEffectHandler keeps
        # a direct reference to the controller.
        self._ducking = OutputDuckingController()
        self._decision_effects = self._build_decision_effect_applier()
        self._explicit_preempts = self._build_explicit_client_preempt_ledger()
        self._interruption_orchestrator = self._build_interruption_orchestrator()
        self._attention_effects = self._build_attention_effect_handler()
        self._session_signals = self._build_session_signal_bridge()
        self._client_preempts = self._build_client_preempt_handler()
        self._turn_committer = UserTurnCommitter()
        self._transcript_echo_gate = self._build_transcript_echo_gate()
        self._user_turns = self._build_user_turn_coordinator()
        self._agent_state_effects = self._build_agent_state_effect_handler()
        self._semantic_interrupts = self._build_semantic_interrupt_handler()
        self._duck_deadline = self._build_duck_suspend_timeout_handler()
        self._stable_signal_timer: asyncio.Task | None = None
        self._pending_voiceprint_commit_tasks: set[asyncio.Task] = set()
        self._candidate_voiceprint_tasks: list[asyncio.Task] = []
        self._deferred_low_eot_commit_task: asyncio.Task | None = None
        self._suppress_transcripts_until_next_speech = False
        self._transcript_admission = self._build_transcript_admission_gate()
        self._completed_turn_voiceprint_task: asyncio.Task | None = None
        self._completed_turn_voiceprint_result: Any | None = None
        self._completed_turn_voiceprint_timeline: TurnTimeline | None = None
        self._provider_events = ProviderEventObserver(
            factory=self._factory,
            get_timeline=lambda: self._timeline,
            flush_timeline=lambda timeline, reason: self._flush_turn_timeline(
                timeline,
                reason,
            ),
            append_timeline_snapshot=lambda timeline, reason: self._append_turn_timeline_snapshot(
                timeline, reason
            ),
            first_delta_timeout_sec=(self._observability.llm_first_delta_timeout_ms / 1000.0),
        )
        self._voiceprint_turns = VoiceprintTurnObserver(
            service=getattr(self._factory, "voiceprint_service", None),
            runtime_admin=getattr(self._factory, "runtime_admin", None),
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
        self._idle_disconnect_grace_sec: float = (
            self._turn_policy.idle.disconnect_grace_ms / 1000.0
        )
        self._idle_watchdog_controller = IdleWatchdog(
            timeout_sec=self._idle_timeout_sec,
            get_session=lambda: getattr(self, "_session", None),
            get_room=lambda: getattr(self, "_room", None),
            get_timeline=lambda: getattr(self, "_timeline", None),
            session_closed_event=self._session_closed_event,
            on_idle_disconnect=self._on_idle_disconnect,
            on_session_end=self._on_session_end,
            disconnect_grace_sec=self._idle_disconnect_grace_sec,
            idle_end_reason=self._idle_end_reason,
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
        # watcher (see ``_install_duck_mixer`` and SemanticInterruptHandler).
        # The soft-interrupt path here is kept as a fallback for the rare
        # cases where EOT signals a cut but the mixer isn't installed
        # (e.g. duck_enabled=False, or audio output sink not yet attached).
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
        self._interrupted_context = InterruptedContextManager()

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
            on_hold=self._handle_hold_decision,
            on_decision=lambda decision, **kwargs: (
                self._interruption_orchestrator.note_turn_policy_decision(
                    decision,
                    **kwargs,
                )
            ),
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
        )

    def _ensure_interruption_orchestrator(self) -> None:
        if not hasattr(self, "_interruption_orchestrator"):
            self._interruption_orchestrator = self._build_interruption_orchestrator()

    def _build_attention_effect_handler(self) -> AttentionEffectHandler:
        return AttentionEffectHandler(
            turn_policy=self._turn_policy,
            turn_runtime=self._turn_runtime,
            get_agent_speaking=lambda: self._agent_output_active_for_interrupts(),
            get_duck_active=lambda: self._ducking.is_suspended,
            latest_client_audio_state=lambda participant_identity: self._latest_client_audio_state(
                participant_identity=participant_identity,
            ),
            get_timeline=lambda: self._timeline,
            on_duck=lambda: self._duck_and_arm_timeout(),
            on_interrupt=lambda: self._interrupt_current_turn(),
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

    def _build_client_preempt_handler(self) -> ExplicitClientPreemptHandler:
        return ExplicitClientPreemptHandler(
            latest_client_audio_state=lambda participant_identity=None: (
                self._latest_client_audio_state(participant_identity=participant_identity)
            ),
            agent_output_active_for_interrupts=lambda participant_identity=None: (
                self._agent_output_active_for_interrupts(
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
            cancel_agent_output=lambda force: self._duck_cancel_and_interrupt(force=force),
            agent_turn_active_for_explicit_preempt=lambda participant_identity=None: (
                self._agent_turn_active_for_explicit_preempt(
                    participant_identity=participant_identity,
                )
            ),
            preempt_agent_turn_for_explicit_control=(
                self._preempt_agent_turn_for_explicit_control
            ),
        )

    def _ensure_client_preempt_handler(self) -> None:
        if not hasattr(self, "_client_preempts"):
            self._client_preempts = self._build_client_preempt_handler()

    def _ensure_turn_committer(self) -> None:
        if not hasattr(self, "_turn_committer"):
            self._turn_committer = UserTurnCommitter()

    def _build_transcript_echo_gate(self) -> TranscriptEchoGate:
        return TranscriptEchoGate(factory=getattr(self, "_factory", None))

    def _ensure_transcript_echo_gate(self) -> TranscriptEchoGate:
        if not hasattr(self, "_transcript_echo_gate"):
            self._transcript_echo_gate = self._build_transcript_echo_gate()
        return self._transcript_echo_gate

    def _build_transcript_admission_gate(self) -> TranscriptAdmissionGate:
        return TranscriptAdmissionGate(
            suppress_until_next_speech=(
                lambda: self._suppress_transcripts_until_next_speech
            ),
            agent_output_active=lambda speaker_id: (
                self._agent_output_active_for_interrupts(
                    participant_identity=speaker_id,
                )
            ),
            echo_gate=lambda: self._ensure_transcript_echo_gate(),
        )

    def _ensure_transcript_admission_gate(self) -> TranscriptAdmissionGate:
        if not hasattr(self, "_transcript_admission"):
            self._transcript_admission = self._build_transcript_admission_gate()
        return self._transcript_admission

    def _low_eot_commit_grace_max_sec(self) -> float:
        return max(self._turn_policy.eot.low_eot_commit_grace_max_ms, 0) / 1000.0

    def _statement_deferred_merge_grace_sec(self) -> float:
        return max(self._turn_policy.eot.statement_deferred_merge_grace_ms, 0) / 1000.0

    def _voiceprint_deferred_merge_grace_sec(self) -> float:
        return max(self._turn_policy.eot.voiceprint_deferred_merge_grace_ms, 0) / 1000.0

    def _cancel_residual_commit_suppress_sec(self) -> float:
        return max(self._turn_policy.interrupt.cancel_residual_commit_suppress_ms, 0) / 1000.0

    def _build_user_turn_coordinator(self) -> UserTurnCoordinator:
        delay = min(
            max(self._turn_policy.eot.tail_hang_silence_ms / 1000.0, 0.0),
            self._low_eot_commit_grace_max_sec(),
        )
        return UserTurnCoordinator(
            merge_grace_sec=delay,
            statement_deferred_merge_grace_sec=max(
                delay,
                self._statement_deferred_merge_grace_sec(),
            ),
            voiceprint_deferred_merge_grace_sec=max(
                delay,
                self._voiceprint_deferred_merge_grace_sec(),
            ),
            low_eot_delay_sec=delay,
            statement_sequence_merge_max_cjk_chars=(
                self._turn_policy.eot.statement_sequence_merge_max_cjk_chars
            ),
            statement_sequence_fragment_max_cjk_chars=(
                self._turn_policy.eot.statement_sequence_fragment_max_cjk_chars
            ),
            transcript_revision_min_normalized_chars=(
                self._turn_policy.eot.transcript_revision_min_normalized_chars
            ),
        )

    def _ensure_user_turn_coordinator(self) -> None:
        if not hasattr(self, "_user_turns"):
            self._user_turns = self._build_user_turn_coordinator()

    def _cancel_pending_voiceprint_commits(self, reason: str) -> None:
        tasks = getattr(self, "_pending_voiceprint_commit_tasks", set())
        for task in list(tasks):
            if not task.done():
                logger.info(
                    "[StreamingPipeline] cancelling pending voiceprint-gated commit reason=%s",
                    reason,
                )
                task.cancel()

    def _reset_candidate_voiceprint_tasks(self) -> None:
        self._candidate_voiceprint_tasks = []

    def _remember_candidate_voiceprint_task(self, task: asyncio.Task | None) -> None:
        if task is None:
            return
        if not hasattr(self, "_candidate_voiceprint_tasks"):
            self._candidate_voiceprint_tasks = []
        self._candidate_voiceprint_tasks.append(task)

    def _candidate_voiceprint_gate_task(self) -> asyncio.Task | None:
        tasks = list(getattr(self, "_candidate_voiceprint_tasks", []))
        self._candidate_voiceprint_tasks = []
        if not tasks:
            return None
        if len(tasks) == 1:
            return tasks[0]
        return asyncio.create_task(self._combine_candidate_voiceprint_results(tasks))

    async def _combine_candidate_voiceprint_results(
        self,
        tasks: list[asyncio.Task],
    ) -> Any:
        results = await asyncio.gather(*tasks)
        inconclusive = None
        for result in results:
            if not bool(getattr(result, "commit_allowed", False)):
                if _voiceprint_result_is_inconclusive(result):
                    inconclusive = inconclusive or result
                    continue
                return result
        if results and bool(getattr(results[-1], "commit_allowed", False)):
            return results[-1]
        if inconclusive is not None:
            return inconclusive
        return results[-1]

    def _cancel_deferred_low_eot_commit(self, reason: str) -> None:
        task = getattr(self, "_deferred_low_eot_commit_task", None)
        if task is None or task.done():
            self._deferred_low_eot_commit_task = None
            return
        logger.info(
            "[StreamingPipeline] cancelling deferred low-EOT commit reason=%s",
            reason,
        )
        task.cancel()
        self._deferred_low_eot_commit_task = None

    def _should_defer_low_eot_commit(self, *, transcript: str, eot_model: Any) -> bool:
        self._ensure_runtime_defaults()
        if not transcript.strip():
            return False
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 1.0),
            )
            or 0.0
        )
        if score < float(self._turn_policy.eot.eot_unlikely_threshold):
            return True
        return self._looks_like_short_statement_continuation(transcript)

    def _playback_low_evidence_reject_reason(
        self,
        *,
        transcript: str,
        eot_model: Any,
    ) -> str:
        if not transcript.strip():
            return ""
        timeline = getattr(self, "_timeline", None)
        if timeline is None:
            return ""
        events = timeline.attrs.get("attention_admission_events") or ()
        playback_observed = any(
            isinstance(event, dict)
            and event.get("action") == "observe"
            and str(event.get("reason") or "").startswith(
                (
                    "client_playback_active_without_direct_signal",
                    "playback_low_evidence_transcript",
                )
            )
            for event in events
        )
        if not playback_observed:
            return ""
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 0.0),
            )
            or 0.0
        )
        evidence = TranscriptEvidenceGate(self._turn_policy.interrupt).evaluate_attention(
            transcript, eot_score=score
        )
        if evidence.allow_decision:
            return ""
        return f"playback_low_evidence_artifact:{evidence.reason}"

    def _looks_like_short_statement_continuation(self, transcript: str) -> bool:
        text = transcript.strip()
        if not text:
            return False
        if any(mark in text for mark in ("？", "?", "！", "!")):
            return False
        cjk_chars = _count_cjk_chars(text)
        if cjk_chars <= 0:
            return False
        if cjk_chars > self._turn_policy.eot.short_statement_defer_max_cjk_chars:
            return False
        if text.startswith(("帮我", "请", "麻烦", "换个话题", "换一个话题")):
            return False
        # Short declarative fragments like "私立医院的。" or "给医生做的系统。"
        # often arrive before the user has finished a multi-clause thought.
        return text.endswith(("。", "，", ",", "、", "的", "了", "呢", "吧"))

    def _schedule_deferred_low_eot_commit(
        self,
        *,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
        delay_sec: float | None = None,
    ) -> None:
        self._cancel_deferred_low_eot_commit("replace_deferred_commit")
        if delay_sec is None:
            delay = min(
                max(self._turn_policy.eot.tail_hang_silence_ms / 1000.0, 0.0),
                self._low_eot_commit_grace_max_sec(),
            )
        else:
            delay = max(float(delay_sec), 0.0)
        task = asyncio.create_task(
            self._run_deferred_low_eot_commit(
                delay=delay,
                verify_task=verify_task,
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
        )
        self._deferred_low_eot_commit_task = task
        logger.info(
            "[StreamingPipeline] deferred low-EOT commit delay=%.3fs score=%s transcript=%r",
            delay,
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            transcript[:80],
        )

    async def _run_deferred_low_eot_commit(
        self,
        *,
        delay: float,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            self._ensure_user_turn_coordinator()
            decision = self._user_turns.deferred_ready()
            if decision.action != "commit":
                logger.info(
                    "[StreamingPipeline] deferred low-EOT commit skipped reason=%s",
                    decision.reason,
                )
                return
            final_transcript = (
                decision.transcript.strip() or self._latest_asr_text.strip() or transcript
            )
            self._schedule_voiceprint_gated_commit(
                verify_task=self._candidate_voiceprint_gate_task() or verify_task,
                eot_model=eot_model,
                transcript=final_transcript,
                timeline=timeline,
            )
            self._latest_asr_text = ""
        except asyncio.CancelledError:
            raise
        finally:
            if self._deferred_low_eot_commit_task is asyncio.current_task():
                self._deferred_low_eot_commit_task = None

    def _schedule_voiceprint_gated_commit(
        self,
        *,
        verify_task: asyncio.Task | None,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        if verify_task is None:
            self._commit_user_turn_now(
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
            return
        self._completed_turn_voiceprint_task = verify_task
        self._completed_turn_voiceprint_result = None
        self._completed_turn_voiceprint_timeline = timeline
        task = asyncio.create_task(
            self._finalize_voiceprint_gated_commit(
                verify_task=verify_task,
                eot_model=eot_model,
                transcript=transcript,
                timeline=timeline,
            )
        )
        self._pending_voiceprint_commit_tasks.add(task)
        task.add_done_callback(self._pending_voiceprint_commit_tasks.discard)

    async def _finalize_voiceprint_gated_commit(
        self,
        *,
        verify_task: asyncio.Task,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> None:
        try:
            result = await verify_task
        except asyncio.CancelledError:
            eot_model.reset()
            raise
        except Exception as exc:  # noqa: BLE001 - conservative gate
            eot_model.reset()
            self._clear_session_user_turn("voiceprint_error")
            self._record_voiceprint_commit_gate(
                timeline,
                allowed=False,
                reason=f"voiceprint_error:{type(exc).__name__}",
            )
            self._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
            logger.exception("[StreamingPipeline] voiceprint gate failed")
            return

        self._completed_turn_voiceprint_result = result
        self._ensure_user_turn_coordinator()
        allowed = bool(getattr(result, "commit_allowed", False))
        raw_reason = str(getattr(result, "commit_reason", "") or "unknown")
        if not allowed and self._should_keep_waiting_merge_after_inconclusive_voiceprint(
            result,
            transcript=transcript,
        ):
            self._defer_inconclusive_voiceprint_result(
                transcript=transcript,
                timeline=timeline,
                reason=raw_reason,
            )
            self._record_voiceprint_commit_gate(
                timeline,
                allowed=False,
                reason=raw_reason,
            )
            return

        decision = self._user_turns.apply_voiceprint_result(result)
        allowed = decision.action == "commit"
        reason = decision.reason or str(getattr(result, "commit_reason", "") or "unknown")
        self._record_voiceprint_commit_gate(timeline, allowed=allowed, reason=reason)
        if not allowed:
            eot_model.reset()
            self._suppress_transcripts_until_next_speech = True
            self._clear_session_user_turn(f"voiceprint_blocked:{reason}")
            self._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
            logger.info(
                "[StreamingPipeline] voiceprint gate blocked commit reason=%s transcript=%r",
                reason,
                transcript[:80],
            )
            return

        self._commit_user_turn_now(
            eot_model=eot_model,
            transcript=decision.transcript or transcript,
            timeline=timeline,
        )

    def _commit_user_turn_now(
        self,
        *,
        eot_model: Any,
        transcript: str,
        timeline: TurnTimeline | None,
    ) -> bool:
        if self._session is None:
            eot_model.reset()
            return False
        if timeline is not None:
            timeline.set_attr(
                "framework_commit_request",
                {
                    "transcript_preview": transcript[:120],
                    "transcript_length": len(transcript),
                },
            )
        self._publish_canonical_user_text(
            transcript,
            source="user_turn_coordinator",
            timeline=timeline,
        )
        self._ensure_turn_committer()
        committed = self._turn_committer.commit_or_skip(
            session=self._session,
            eot_model=eot_model,
            transcript=transcript,
            transcript_timeout=self._stt_commit_transcript_timeout,
            timeline=timeline,
            inject_interrupted_context=self._inject_interrupted_context,
            filler=self._filler,
        )
        if not committed:
            self._ensure_user_turn_coordinator()
            self._user_turns.reject_active("empty_transcript")
            self._clear_session_user_turn("empty_transcript")
        else:
            self._ensure_user_turn_coordinator()
            self._user_turns.mark_committed(
                transcript=transcript,
                reason="framework_commit_user_turn",
            )
        return committed

    def _publish_canonical_user_text(
        self,
        transcript: str,
        *,
        source: str,
        timeline: TurnTimeline | None,
    ) -> None:
        stripped = transcript.strip()
        if not stripped:
            return
        try:
            llm_plugin = getattr(getattr(self._factory, "llm", None), "llm", None)
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is None:
                return
            setter(stripped, source=source)
            if timeline is not None:
                timeline.set_attr(
                    "canonical_user_text",
                    {
                        "source": source,
                        "text_preview": stripped[:120],
                        "text_length": len(stripped),
                    },
                )
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to publish canonical user text",
                exc_info=True,
            )

    def _clear_pending_canonical_user_text(self, reason: str) -> None:
        try:
            factory = getattr(self, "_factory", None)
            llm_plugin = getattr(getattr(factory, "llm", None), "llm", None)
            clearer = getattr(llm_plugin, "clear_next_user_text", None)
            if clearer is not None:
                clearer(reason=reason)
                return
            setter = getattr(llm_plugin, "set_next_user_text", None)
            if setter is not None:
                setter("", source=f"clear:{reason}")
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to clear canonical user text",
                exc_info=True,
            )

    def _clear_session_user_turn(self, reason: str) -> None:
        self._clear_pending_canonical_user_text(reason)
        session = getattr(self, "_session", None)
        if session is None:
            return
        clear_user_turn = getattr(session, "clear_user_turn", None)
        if clear_user_turn is None:
            return
        try:
            clear_user_turn()
            logger.info("[StreamingPipeline] cleared user turn reason=%s", reason)
        except Exception:
            logger.exception(
                "[StreamingPipeline] failed to clear user turn reason=%s",
                reason,
            )
        # task #9: a turn cleared because the session CONTEXT could not be
        # resolved (e.g. the user/device is bound to a deleted agent →
        # AdminResolveNotFound) is an operator-actionable misconfiguration, not a
        # routine voiceprint reject. Every turn will be dropped, so don't leave
        # the user in an indefinite silent dead-end ("connects, plays welcome,
        # never answers"): surface it loudly + tell them ONCE.
        if "context_error" in (reason or ""):
            self._notify_context_error_once(reason)

    def _notify_context_error_once(self, reason: str) -> None:
        """Loudly report an unresolved-context turn drop and tell the user once."""
        if getattr(self, "_context_error_notified", False):
            return
        self._context_error_notified = True
        logger.error(
            "[StreamingPipeline] conversation blocked: session context unresolved "
            "(reason=%s). The user/device likely references a missing agent "
            "binding; turns are dropped until it is rebound in admin.",
            reason,
        )
        session = getattr(self, "_session", None)
        say = getattr(session, "say", None) if session is not None else None
        if not callable(say):
            return
        try:
            say(
                "抱歉，我暂时无法连接到你的助手，请检查账号绑定或联系管理员。",
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("[StreamingPipeline] context-error fallback announcement failed")
            return
        try:
            self._mark_activity()
        except Exception:
            logger.debug(
                "[StreamingPipeline] mark_activity after context-error say failed",
                exc_info=True,
            )

    async def _voiceprint_allows_completed_turn(self, *, new_message: Any) -> bool:
        """Gate LiveKit's final turn-completed hook with voiceprint ownership.

        This is the last public lifecycle boundary before LiveKit starts the
        LLM reply, so it catches both our explicit commit path and framework
        auto-EOU paths such as late STT FINAL delivery.
        """
        self._ensure_runtime_defaults()
        task = getattr(self, "_completed_turn_voiceprint_task", None)
        result = getattr(self, "_completed_turn_voiceprint_result", None)
        timeline = getattr(self, "_completed_turn_voiceprint_timeline", None) or getattr(
            self, "_timeline", None
        )
        completed_transcript = message_text(new_message)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_turn",
                {
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
        if self._stop_active_interruption_framework_completed_turn(
            completed_transcript,
            timeline=timeline,
        ):
            return False
        if task is None and result is None:
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason="",
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason="",
            )
            return True
        if result is None and task is not None:
            try:
                result = await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - conservative gate
                self._record_voiceprint_commit_gate(
                    timeline,
                    allowed=False,
                    reason=f"voiceprint_error:{type(exc).__name__}",
                )
                self._clear_session_user_turn(f"voiceprint_error:{type(exc).__name__}")
                self._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
                logger.exception("[StreamingPipeline] voiceprint gate failed in turn hook")
                return False
            self._completed_turn_voiceprint_result = result

        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        self._record_voiceprint_commit_gate(timeline, allowed=allowed, reason=reason)
        if allowed:
            if self._stop_active_interruption_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._stop_non_semantic_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
            ):
                return False
            if self._should_defer_framework_completed_turn(completed_transcript):
                self._defer_framework_completed_turn(
                    completed_transcript=completed_transcript,
                    timeline=timeline,
                    voiceprint_reason=reason,
                )
                return False
            self._align_framework_completed_turn(
                completed_transcript,
                timeline=timeline,
                voiceprint_reason=reason,
            )
            return True

        if self._should_keep_waiting_merge_after_inconclusive_voiceprint(
            result,
            transcript=completed_transcript,
        ):
            self._defer_inconclusive_voiceprint_result(
                transcript=completed_transcript,
                timeline=timeline,
                reason=reason,
            )
            return False

        self._suppress_transcripts_until_next_speech = True
        self._clear_session_user_turn(f"voiceprint_blocked:{reason}")
        self._flush_turn_timeline(timeline, "voiceprint_commit_blocked")
        logger.info(
            "[StreamingPipeline] voiceprint gate stopped completed turn reason=%s transcript=%r",
            reason,
            completed_transcript[:80],
        )
        return False

    def _eot_thinks_turn_complete(self) -> bool:
        """True when the learned EOT model is confident the user's turn is done.

        Single source of truth for "trust the end-of-turn model". The
        framework-completed defer path uses it so a confident EOT (the model's
        own completeness call) is not second-guessed by the short-statement text
        heuristic — that heuristic keys off trailing punctuation/particles which
        the ASR routinely drops (e.g. a question's trailing 「吗？」), so on its own
        it mis-holds genuinely complete turns. Mirrors the score idiom in
        ``_should_defer_low_eot_commit``.
        """
        eot_model = self._get_eot_model()
        if eot_model is None:
            return True
        score = float(
            getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", 1.0),
            )
            or 0.0
        )
        return score >= float(self._turn_policy.eot.eot_unlikely_threshold)

    def _should_defer_framework_completed_turn(self, transcript: str) -> bool:
        self._ensure_user_turn_coordinator()
        candidate = self._user_turns.active
        if candidate is None:
            return False
        if candidate.state in {"committed", "rejected"}:
            return False
        if self._user_turns.should_wait_for_deferred_voiceprint_merge():
            return True
        if self._user_turns.should_wait_for_statement_sequence_merge():
            return True
        # The LiveKit framework already decided this turn is complete. If our EOT
        # model agrees, don't re-hold it on the short-statement text heuristic
        # (which a dropped 「吗？」 defeats) — let it take the normal reply path.
        # The heuristic still hedges when EOT itself is unsure.
        if self._eot_thinks_turn_complete():
            return False
        selected = candidate.selected_text or transcript
        return self._looks_like_short_statement_continuation(selected)

    def _defer_framework_completed_turn(
        self,
        *,
        completed_transcript: str,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        self._clear_session_user_turn("framework_completed_wait_for_continuation")
        self._ensure_user_turn_coordinator()
        decision = self._user_turns.defer_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_wait_for_continuation",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_deferred",
                {
                    "reason": decision.reason,
                    "state": "waiting_merge",
                    "text_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            self._append_turn_timeline_snapshot(
                timeline,
                "framework_completed_waiting_merge",
            )
        self._schedule_deferred_low_eot_commit(
            verify_task=None,
            eot_model=self._get_eot_model(),
            transcript=decision.transcript or completed_transcript,
            timeline=timeline,
            delay_sec=decision.delay_sec,
        )
        self._completed_turn_voiceprint_task = None
        self._completed_turn_voiceprint_result = None
        self._completed_turn_voiceprint_timeline = None
        logger.info(
            "[StreamingPipeline] deferred framework completed turn "
            "for continuation transcript=%r voiceprint_reason=%s",
            completed_transcript[:80],
            voiceprint_reason,
        )

    def _should_keep_waiting_merge_after_inconclusive_voiceprint(
        self,
        result: Any,
        *,
        transcript: str,
    ) -> bool:
        if not _voiceprint_result_is_inconclusive(result):
            return False
        self._ensure_user_turn_coordinator()
        candidate = self._user_turns.active
        if candidate is None:
            return False
        if candidate.state == "waiting_merge":
            return True
        if candidate.state in {"committed", "rejected"}:
            return False
        selected = candidate.selected_text or transcript
        return self._looks_like_short_statement_continuation(selected)

    def _defer_inconclusive_voiceprint_result(
        self,
        *,
        transcript: str,
        timeline: TurnTimeline | None,
        reason: str,
    ) -> None:
        defer_reason = f"voiceprint_inconclusive:{reason}"
        self._clear_session_user_turn(defer_reason)
        self._ensure_user_turn_coordinator()
        decision = self._user_turns.defer_voiceprint_inconclusive(
            transcript=transcript,
            reason=defer_reason,
            timeline=timeline,
        )
        if timeline is not None:
            timeline.set_attr(
                "voiceprint_deferred",
                {
                    "reason": defer_reason,
                    "state": "waiting_merge",
                    "text_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            self._append_turn_timeline_snapshot(timeline, "voiceprint_waiting_merge")
        logger.info(
            "[StreamingPipeline] deferred inconclusive voiceprint result reason=%s transcript=%r",
            reason,
            transcript[:80],
        )

    def _align_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
        voiceprint_reason: str,
    ) -> None:
        self._cancel_deferred_low_eot_commit("framework_completed_turn")
        self._ensure_user_turn_coordinator()
        decision = self._user_turns.mark_framework_completed(
            transcript=completed_transcript,
            reason="framework_completed_turn",
            timeline=timeline,
            voiceprint_reason=voiceprint_reason,
        )
        canonical = decision.transcript or completed_transcript
        self._publish_canonical_user_text(
            canonical,
            source="framework_completed_turn",
            timeline=timeline,
        )

    @staticmethod
    def _non_semantic_completed_turn_reason(
        timeline: TurnTimeline | None,
    ) -> str:
        if timeline is None:
            return ""
        decision = timeline.attrs.get("decision")
        if not isinstance(decision, dict):
            return ""
        action = str(decision.get("action") or "")
        intent = str(decision.get("intent") or "")
        if action == "rollback":
            return f"non_semantic_completed_turn:{intent or action}"
        if intent in {"backchannel", "noise", "hard_stop"}:
            return f"non_semantic_completed_turn:{intent}"
        return ""

    def _stop_non_semantic_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        stop_reason = self._non_semantic_completed_turn_reason(timeline)
        if not stop_reason:
            return False
        self._cancel_deferred_low_eot_commit(stop_reason)
        self._ensure_user_turn_coordinator()
        self._user_turns.reject_active(stop_reason)
        self._clear_session_user_turn(stop_reason)
        self._flush_turn_timeline(timeline, stop_reason)
        logger.info(
            "[StreamingPipeline] stopped framework completed turn reason=%s transcript=%r",
            stop_reason,
            completed_transcript[:80],
        )
        return True

    def _stop_active_interruption_framework_completed_turn(
        self,
        completed_transcript: str,
        *,
        timeline: TurnTimeline | None,
    ) -> bool:
        """Block framework context commit while Channel still owns evidence."""

        owner = getattr(self, "_interruption_orchestrator", None)
        if owner is None or not owner.blocks_framework_completed_turn():
            return False
        reason = "interruption_owner_waiting_for_evidence"
        self._cancel_deferred_low_eot_commit(reason)
        self._clear_session_user_turn(reason)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_blocked_by_interruption_owner",
                {
                    "reason": reason,
                    "state": owner.state.value,
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
            self._append_turn_timeline_snapshot(timeline, reason)
        logger.info(
            "[StreamingPipeline] blocked framework completed turn while "
            "interruption owner waits reason=%s state=%s transcript=%r",
            reason,
            owner.state.value,
            completed_transcript[:80],
        )
        return True

    def _commit_post_speech_interruption_candidate(
        self,
        reason: str,
        *,
        transcript_override: str = "",
    ) -> bool:
        """Commit a confirmed semantic interrupt after output cancellation."""

        owner = getattr(self, "_interruption_orchestrator", None)
        self._ensure_user_turn_coordinator()
        owner_transcript = owner.current_transcript if owner is not None else ""
        transcript = (
            transcript_override
            or owner_transcript
            or self._user_turns.selected_text
            or self._latest_asr_text
        ).strip()
        if not transcript:
            return False
        timeline = getattr(self, "_timeline", None)
        self._cancel_deferred_low_eot_commit(reason)
        if self._user_turns.active is None:
            if self._timeline is None:
                self._timeline = TurnTimeline(generate_turn_id())
                self._timeline_debug_flushed = False
                timeline = self._timeline
            self._user_turns.start_speech(timeline=self._timeline)
            self._apply_pending_explicit_client_preempt(self._timeline)
            self._apply_pending_client_control_events(self._timeline)
        self._user_turns.add_transcript(transcript, is_final=True)
        eot_model = self._get_eot_model()
        decision = self._user_turns.finish_speech(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            should_defer=False,
        )
        if decision.action == "reject":
            self._clear_session_user_turn(decision.reason)
            return False
        committed_text = decision.transcript or transcript
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_committed",
                {
                    "reason": reason,
                    "transcript_preview": committed_text[:120],
                    "text_length": len(committed_text),
                },
            )
        self._schedule_voiceprint_gated_commit(
            verify_task=self._candidate_voiceprint_gate_task(),
            eot_model=eot_model,
            transcript=committed_text,
            timeline=timeline,
        )
        self._latest_asr_text = ""
        logger.info(
            "[StreamingPipeline] committed post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            committed_text[:80],
        )
        return True

    def _reject_post_speech_interruption_candidate(self, reason: str) -> None:
        """Close a false interruption that waited for delayed STT evidence."""

        timeline = getattr(self, "_timeline", None)
        self._cancel_deferred_low_eot_commit(reason)
        self._ensure_user_turn_coordinator()
        decision = self._user_turns.reject_active(reason)
        eot_model = self._get_eot_model()
        try:
            eot_model.reset()
        except Exception:
            logger.debug(
                "[StreamingPipeline] EOT reset failed while rejecting "
                "post-speech interruption candidate",
                exc_info=True,
            )
        task = getattr(self, "_completed_turn_voiceprint_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._completed_turn_voiceprint_task = None
        self._completed_turn_voiceprint_result = None
        self._completed_turn_voiceprint_timeline = None
        self._reset_candidate_voiceprint_tasks()
        self._clear_session_user_turn(reason)
        self._latest_asr_text = ""
        if timeline is not None:
            timeline.set_attr(
                "post_speech_interruption_candidate_rejected",
                {
                    "reason": reason,
                    "transcript_preview": decision.transcript[:120],
                    "text_length": len(decision.transcript),
                },
            )
            self._flush_turn_timeline(timeline, reason)
        logger.info(
            "[StreamingPipeline] rejected post-speech interruption candidate "
            "reason=%s transcript=%r",
            reason,
            decision.transcript[:80],
        )

    def _record_voiceprint_commit_gate(
        self,
        timeline: TurnTimeline | None,
        *,
        allowed: bool,
        reason: str,
    ) -> None:
        if timeline is None:
            return
        timeline.set_attr(
            "voiceprint_commit_gate",
            {
                "allowed": allowed,
                "reason": reason,
            },
        )

    def _flush_turn_timeline(
        self,
        timeline: TurnTimeline | None,
        reason: str,
    ) -> None:
        if timeline is None or getattr(self, "_timeline_debug_flushed", False):
            return
        timeline.set_attr("timeline_flush_reason", reason)
        timeline.append_debug_jsonl(self._observability.timeline_debug_path)
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
        timeline.append_debug_jsonl(self._observability.timeline_debug_path)

    def _build_agent_state_effect_handler(self) -> AgentStateEffectHandler:
        return AgentStateEffectHandler(
            get_timeline=lambda: self._timeline,
            mark_activity=lambda: self._mark_activity(),
            cancel_soft_interrupt=lambda: self._cancel_soft_interrupt(),
            soft_interrupt_active=lambda: self._soft_interrupt_is_active(),
            ducking=self._ducking,
            get_filler=lambda: self._filler,
            flush_timeline_debug=lambda reason, clear: self._append_timeline_debug(
                reason,
                clear=clear,
            ),
            should_flush_on_playback_done=self._timeline_turn_terminal_for_playback_flush,
        )

    def _ensure_agent_state_effect_handler(self) -> None:
        handler = getattr(self, "_agent_state_effects", None)
        if handler is None or getattr(handler, "_ducking", None) is not self._ducking:
            self._agent_state_effects = self._build_agent_state_effect_handler()

    def _timeline_turn_terminal_for_playback_flush(self) -> bool:
        turns = getattr(self, "_user_turns", None)
        if turns is None:
            return True
        active = getattr(turns, "active", None)
        if active is None:
            return True
        return getattr(active, "state", "") in {"committed", "rejected"}

    def _build_semantic_interrupt_handler(self) -> SemanticInterruptHandler:
        return SemanticInterruptHandler(
            get_eot_model=lambda: self._get_eot_model(),
            turn_runtime=self._turn_runtime,
            get_timeline=lambda: self._timeline,
            get_duck_active=lambda: self._ducking.is_suspended,
            get_duck_stats=lambda: self._ducking.stats(),
            get_vad_active=lambda: (
                self._session is not None and self._session.user_state == "speaking"
            ),
            soft_interrupt_active=lambda: self._soft_interrupt_is_active(),
            soft_interrupt_timeout=lambda: self._soft_interrupt_timeout,
            apply_decision=self._decision_effects.apply,
            record_decision_attrs=self._decision_effects.record_decision_attrs,
            publish_turn_control=self._decision_effects.publish_turn_control,
            cancel_duck_and_interrupt=lambda: self._duck_cancel_and_interrupt(),
            interrupt_current_turn=lambda: self._interrupt_current_turn(),
            enter_soft_interrupt=lambda: self._enter_soft_interrupt(),
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
            get_max_suspend_sec=lambda: self._interruption_orchestrator.max_suspend_sec(),
            deadline_decision=(
                lambda vad_still_active, **kwargs: (
                    self._interruption_orchestrator.deadline_decision(
                        self._turn_runtime,
                        vad_still_active,
                        **kwargs,
                    )
                )
            ),
        )

    def _ensure_duck_suspend_timeout_handler(self) -> None:
        handler = getattr(self, "_duck_deadline", None)
        if handler is None or getattr(handler, "_turn_runtime", None) is not self._turn_runtime:
            self._duck_deadline = self._build_duck_suspend_timeout_handler()

    def _install_provider_observers(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_all()

    def _install_llm_metrics_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_llm_metrics_observer()

    def _install_brain_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_brain_provider_event_observer()

    def _install_tts_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_tts_provider_event_observer()

    def _install_stt_provider_event_observer(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.install_stt_provider_event_observer()

    def _remember_pending_stt_provider_event(self, event: dict[str, Any]) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.remember_pending_stt_provider_event(event)

    def _apply_pending_stt_provider_events(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.apply_pending_stt_provider_events()

    def _record_stt_provider_event(self, event: dict[str, Any]) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.record_stt_provider_event(event)

    def _observe_stt_turn_audio(self) -> None:
        self._ensure_provider_event_observer()
        self._provider_events.observe_stt_turn_audio()

    def _ensure_provider_event_observer(self) -> None:
        if not hasattr(self, "_provider_events"):
            if not hasattr(self, "_observability"):
                self._observability = ObservabilityConfig()
            if not hasattr(self, "_timeline"):
                self._timeline = None
            self._provider_events = ProviderEventObserver(
                factory=self._factory,
                get_timeline=lambda: self._timeline,
                flush_timeline=lambda timeline, reason: self._flush_turn_timeline(
                    timeline,
                    reason,
                ),
                append_timeline_snapshot=lambda timeline, reason: (
                    self._append_turn_timeline_snapshot(timeline, reason)
                ),
                first_delta_timeout_sec=(self._observability.llm_first_delta_timeout_ms / 1000.0),
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
        if not hasattr(self, "_voiceprint_config"):
            self._voiceprint_config = VoiceprintConfig()
        if not hasattr(self, "_timeline"):
            self._timeline = None
        if not hasattr(self, "_timeline_debug_flushed"):
            self._timeline_debug_flushed = False
        if not hasattr(self, "_pending_client_control_events"):
            self._pending_client_control_events = []
        if not hasattr(self, "_skip_commit_after_interrupt_cancel"):
            self._skip_commit_after_interrupt_cancel = False
        if not hasattr(self, "_suppress_commit_after_interrupt_until"):
            self._suppress_commit_after_interrupt_until = 0.0
        if not hasattr(self, "_latest_asr_text"):
            self._latest_asr_text = ""
        if not hasattr(self, "_stable_signal_timer"):
            self._stable_signal_timer = None
        if not hasattr(self, "_pending_voiceprint_commit_tasks"):
            self._pending_voiceprint_commit_tasks = set()
        if not hasattr(self, "_candidate_voiceprint_tasks"):
            self._candidate_voiceprint_tasks = []
        if not hasattr(self, "_deferred_low_eot_commit_task"):
            self._deferred_low_eot_commit_task = None
        self._ensure_user_turn_coordinator()
        if not hasattr(self, "_interaction_mode"):
            self._interaction_mode = INTERACTION_MODE_FULL_DUPLEX
        if not hasattr(self, "_suppress_transcripts_until_next_speech"):
            self._suppress_transcripts_until_next_speech = False
        self._ensure_transcript_admission_gate()
        if not hasattr(self, "_completed_turn_voiceprint_task"):
            self._completed_turn_voiceprint_task = None
        if not hasattr(self, "_completed_turn_voiceprint_result"):
            self._completed_turn_voiceprint_result = None
        if not hasattr(self, "_completed_turn_voiceprint_timeline"):
            self._completed_turn_voiceprint_timeline = None
        if not hasattr(self, "_voiceprint_turns"):
            factory = getattr(self, "_factory", None)
            self._voiceprint_turns = VoiceprintTurnObserver(
                service=getattr(factory, "voiceprint_service", None),
                runtime_admin=getattr(factory, "runtime_admin", None),
                sample_rate=getattr(self, "_audio_sample_rate", 16000),
                max_audio_ms=self._voiceprint_config.turn_max_audio_ms,
                accept_cache_ttl_sec=(self._voiceprint_config.accept_cache_ttl_ms / 1000.0),
                accept_cache_short_audio_max_ms=(
                    self._voiceprint_config.accept_cache_short_audio_max_ms
                ),
                commit_threshold=self._voiceprint_config.owner_commit_threshold,
                owner_short_audio_bypass_ms=(self._voiceprint_config.owner_short_audio_bypass_ms),
                trust_paired_devices=getattr(
                    factory,
                    "voiceprint_trust_paired_devices",
                    True,
                ),
            )
        self._ensure_ducking_controller()
        self._ensure_decision_effect_applier()
        self._ensure_explicit_client_preempt_ledger()
        self._ensure_interruption_orchestrator()
        self._ensure_attention_effect_handler()
        self._ensure_session_signal_bridge()
        self._ensure_client_preempt_handler()
        self._ensure_turn_committer()
        self._ensure_agent_state_effect_handler()
        self._ensure_semantic_interrupt_handler()
        self._ensure_duck_suspend_timeout_handler()
        self._ensure_room_data_handler()

    def _record_client_control_event(
        self,
        *,
        timeline: TurnTimeline | None,
        op: str,
        reason: str,
        turn_id: str,
    ) -> None:
        event = build_client_control_event(op=op, reason=reason, turn_id=turn_id)
        if timeline is None:
            pending = list(getattr(self, "_pending_client_control_events", []) or [])
            self._pending_client_control_events = append_client_control_event(
                pending,
                event,
            )
            return
        events = list(timeline.attrs.get("client_control_events") or ())
        timeline.set_attr(
            "client_control_events",
            append_client_control_event(events, event),
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
        interruption: dict[str, Any] = {
            "enabled": self._allow_interruptions,
            "discard_audio_if_uninterruptible": True,
            "false_interruption_timeout": self._false_interruption_timeout,
        }
        if self._uses_livekit_native_adaptive_interruption():
            interruption["mode"] = "adaptive"
            interruption["resume_false_interruption"] = True

        return {
            "interruption": interruption,
            "preemptive_generation": {
                "enabled": self._turn_policy.preemptive.enabled,
                "preemptive_tts": self._turn_policy.preemptive.preemptive_tts,
            },
        }

    def _uses_livekit_native_adaptive_interruption(self) -> bool:
        return (
            getattr(getattr(self, "_turn_policy", None), "interruption_owner", "channel")
            == "livekit_native_adaptive"
            and bool(getattr(self, "_allow_interruptions", False))
        )

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
            turn_handling=self._build_turn_handling(),
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
        self._session_signals.register_vad_inference_callback()

        # Warm up persistent-connection stages (STT, TTS) before starting the
        # session. Stages without a warmup() are silently skipped.
        await self._warmup_stages()
        if self._filler is not None:
            await self._filler.warmup()

        logger.info("[StreamingPipeline] calling session.start()...")
        self._install_room_data_observer(room)
        self._voiceprint_turns.install(room)
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
        self._publish_companion_ui_state("listening", "session_started")
        if self._uses_livekit_native_adaptive_interruption():
            logger.info(
                "[StreamingPipeline] LiveKit native adaptive interruption owner "
                "enabled; channel audio-activity patch skipped"
            )
        else:
            # Disable framework's built-in audio-activity auto-interrupt so
            # Eidolon's InterruptionOrchestrator / turn policy (and the
            # DuckingMixer below) is the sole authority on interrupt decisions.
            # See integration.framework_patches.disable_audio_activity_interruption for the
            # full rationale (no public API alternative — internal flags must be
            # patched). The patch sets BOTH the runtime flag AND the default-
            # value flag, so framework's restore logic on agent state transitions
            # doesn't undo us. No re-patch needed in _on_agent_state_changed.
            framework_patches.disable_audio_activity_interruption(session)

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

        # Subscribe to proactive brain reports so finished background tasks can
        # be spoken unprompted. After session.start() so session.say() has a
        # fully-assembled audio output chain to render into.
        self._start_proactive_consumer()

        try:
            # Wait for AgentSession to close (e.g. participant disconnect →
            # framework auto-closes session via close_on_disconnect=True).
            # This replaces the old `while room.isconnected:` polling, which
            # didn't react to session-level close in time and required the
            # 30s entrypoint watchdog to force shutdown — leaving TTS
            # connections open and heartbeats firing for tens of seconds.
            await self._session_closed_event.wait()
            logger.info("[StreamingPipeline] session closed event received, exiting run()")
            # Delete the room NOW — before shutdown()'s STT/TTS drain — so this
            # fixed-name room (device-<id>) and its agent/track are gone before a
            # rapid re-JOIN. Without this the old agent lingers for the whole
            # drain; an auto_subscribe=false client re-joining the still-alive
            # room subscribes to the STALE track → in-room + agent_speaking state
            # but NO audio (real-device confirmed: JOIN→X→quick JOIN → silent).
            await self._delete_room_on_close()
        except asyncio.CancelledError:
            logger.info("[StreamingPipeline] cancelled")
            raise
        finally:
            await self.shutdown()

    async def _delete_room_on_close(self) -> None:
        """Prompt room teardown on session close (device left / error).

        Runs before shutdown()'s STT/TTS drain so the fixed-name room and its
        agent/track are gone before a rapid re-JOIN (plan §10 follow-up). No-op
        when no callback is wired (direct-construction / tests).

        B2 (plan §3.2): every room-deletion path must carry a ``session_end``
        reason. Publish it (idempotent — no-op if the idle watchdog already sent
        ``idle_normal_end``) BEFORE the prompt delete, so an error-close while the
        client is still connected is not a silent ROOM_DELETED. The close event's
        ``error`` distinguishes ``error`` from a clean ``user_left``.
        """
        on_end = getattr(self, "_on_session_end", None)
        if on_end is not None:
            reason = (
                SESSION_END_ERROR if getattr(self, "_close_error", None) else SESSION_END_USER_LEFT
            )
            try:
                await on_end(reason)
            except Exception:
                logger.exception(
                    "[StreamingPipeline] session_end on close failed (reason=%s)",
                    reason,
                )
        cb = getattr(self, "_on_session_closed", None)
        if cb is None:
            return
        try:
            await cb()
        except Exception:
            logger.exception("[StreamingPipeline] on_session_closed (prompt room delete) failed")

    def _start_proactive_consumer(self) -> None:
        """Spawn the background proactive-report stream (best-effort)."""
        if self._proactive_task is not None and not self._proactive_task.done():
            return
        self._proactive_task = asyncio.create_task(
            self._run_proactive_consumer(),
            name="eidolon-proactive-consumer",
        )

    async def _run_proactive_consumer(self) -> None:
        """Open the proactive stream against the brain and keep it running.

        Only the eidolon_agent gRPC LLM backend can push proactive reports; any
        other LLM plugin (e.g. direct_llm) lacks ``open_proactive_subscriber``
        and is skipped silently. The subscriber's own ``run()`` handles
        reconnect/backoff, so this returns only on cancellation or close.
        """
        llm_plugin = getattr(getattr(self._factory, "llm", None), "llm", None)
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
        self._proactive_subscriber = subscriber
        try:
            await subscriber.run()
        except asyncio.CancelledError:
            raise
        finally:
            await subscriber.aclose()
            self._proactive_subscriber = None

    async def _on_proactive_report(self, report: Any) -> None:
        """Speak a proactive brain report into the room via TTS."""
        text = (getattr(report, "text", "") or "").strip()
        if not text:
            return
        session = self._session
        if session is None:
            logger.info(
                "[StreamingPipeline] dropping proactive report (session closed) intent=%s",
                getattr(report, "intent", ""),
            )
            return
        logger.info(
            "[StreamingPipeline] proactive report intent=%s chars=%d — speaking",
            getattr(report, "intent", ""),
            len(text),
        )
        self._mark_activity()
        # allow_interruptions so the user can cut in if they start talking, the
        # same contract as the welcome message.
        session.say(text, allow_interruptions=True)

    def _stop_proactive_consumer(self) -> None:
        task = self._proactive_task
        if task is not None and not task.done():
            task.cancel()
        self._proactive_task = None

    async def shutdown(self) -> None:
        """Gracefully shut down the session."""
        logger.info("[StreamingPipeline] shutting down")
        self._stop_proactive_consumer()
        # Cancel any pending soft interrupt / duck timeout before closing.
        self._cancel_soft_interrupt()
        self._cancel_stable_signal_timer()
        self._cancel_pending_voiceprint_commits("shutdown")
        if hasattr(self, "_provider_events"):
            self._provider_events.cancel_output_watchdog()
        self._ducking.cancel_timeout()
        self._stop_idle_watchdog()
        if hasattr(self, "_voiceprint_turns"):
            await self._voiceprint_turns.aclose()
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
        """Observe client-side audio hints and drive explicit client preempt.

        The explicit-preempt fast path is wired into the SAME registered
        ``data_received`` callback (via ``on_packet``) so a ``client.audio_state``
        explicit-control edge during playback actually preempts the agent.
        """
        self._ensure_room_data_handler()
        self._ensure_client_preempt_handler()
        self._room_data.install(room, on_packet=self._on_client_room_packet)

    def _on_client_room_packet(self, packet: Any) -> None:
        # Runs after RoomDataHandler.handle_packet has stored the latest client
        # audio state (so do NOT handle_packet again here — that would double-count).
        self._handle_explicit_client_preempt(packet)

    def _record_explicit_client_preempt_decision(
        self,
        *,
        state_attr: dict[str, Any],
        received_at: float,
        resolved_at: float | None = None,
        timeline: TurnTimeline | None = None,
    ) -> None:
        self._ensure_explicit_client_preempt_ledger()
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
        pending = list(getattr(self, "_pending_client_control_events", []) or [])
        if not pending:
            return
        timeline = timeline or getattr(self, "_timeline", None)
        if timeline is None:
            return
        turn_id = getattr(timeline, "turn_id", "")
        events = list(timeline.attrs.get("client_control_events") or ())
        for event in pending:
            attached = dict(event)
            if not attached.get("turn_id"):
                attached["turn_id"] = turn_id
            events = append_client_control_event(events, attached)
        timeline.set_attr("client_control_events", events)
        self._pending_client_control_events = []

    def _mark_explicit_client_preempt_resolved(
        self,
        received_at: float,
        resolved_at: float,
    ) -> None:
        self._ensure_explicit_client_preempt_ledger()
        self._explicit_preempts.mark_resolved(received_at, resolved_at)

    def _handle_explicit_client_preempt(self, packet: Any) -> None:
        self._ensure_client_preempt_handler()
        self._client_preempts.handle_explicit_client_preempt(packet)

    def _turn_detection(self) -> Any:
        """The full-duplex Agent ``turn_detection`` model."""
        return self._get_eot_model()

    def _welcome_on_enter_text(self) -> str | None:
        """Welcome line to speak on session start, or None to stay silent.

        Plan §4.3.1 / §5.1: a proactive_initiated session was woken to deliver a
        report — the report (spoken by the proactive consumer) IS the opening, so
        the canned welcome is suppressed (otherwise the device would say "你好…"
        and then the report). A user_initiated session keeps its welcome (None
        when unconfigured → wait for the user to speak first).
        """
        if self._is_proactive:
            return None
        return self._welcome_message or None

    def _build_agent(self) -> lk_Agent:
        """Build the LiveKit Agent."""
        from livekit.agents.voice import Agent
        from livekit.agents import StopResponse

        pipeline = self

        class VoiceAgent(Agent):
            async def on_enter(self) -> None:
                # [lifecycle] welcome timestamp — anchors "welcome played" so Phase
                # 0 can measure the gap to a later idle room-delete and confirm
                # whether "回 JOIN after welcome" is the idle watchdog firing.
                room_name = getattr(getattr(pipeline, "_room", None), "name", None)
                welcome = pipeline._welcome_on_enter_text()
                if welcome is None:
                    # Suppressed: proactive session (report is the opening, §4.3.1)
                    # or no configured welcome (wait for the user to speak first).
                    logger.info(
                        "[lifecycle] welcome on_enter room=%s suppressed (proactive=%s)",
                        room_name,
                        pipeline._is_proactive,
                    )
                    return
                logger.info(
                    "[lifecycle] welcome on_enter room=%s welcome=%r",
                    room_name,
                    welcome[:30],
                )
                # Round 8 R8.9: use ``session.say(welcome)`` instead of
                # ``session.generate_reply()`` for the initial greeting.
                # generate_reply with no user message hands an empty
                # context to the LLM, which then frequently echoes the
                # system prompt template back as the "welcome". Fixed text
                # is faster (no LLM call), more deterministic, and avoids
                # leaking instruction text to users.
                self.session.say(welcome, allow_interruptions=True)

            async def on_user_turn_completed(
                self,
                turn_ctx: Any,
                new_message: Any,
            ) -> None:
                del turn_ctx
                allowed = await pipeline._voiceprint_allows_completed_turn(new_message=new_message)
                if not allowed:
                    raise StopResponse()

        # Round 8 R8.9 (re-fix): turn_handling config (including
        # false_interruption_timeout, preemptive_generation) lives on
        # AGENTSESSION, not Agent. Putting it here was silently ignored.
        # Agent only carries per-agent override of ``turn_detection`` (the
        # EOT model instance, which is per-agent semantic).
        turn_detection = self._turn_detection()
        return VoiceAgent(
            instructions=self._instructions,
            stt=self._factory.stt.stt,
            llm=self._factory.llm.llm,
            tts=self._factory.tts.tts,
            vad=self._factory.vad.vad if self._factory.vad else None,
            turn_detection=turn_detection,
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
        # Captured for _delete_room_on_close → session_end reason (B2): error
        # close → "error", clean close → "user_left".
        self._close_reason = reason
        self._close_error = error
        logger.info(
            "[StreamingPipeline] session close event received reason=%s error=%s",
            reason,
            error,
        )
        try:
            self._get_eot_model().end_session()
        except Exception:
            logger.exception("[StreamingPipeline] eot_model.end_session failed (non-fatal)")
        duck_metrics = self._ducking.get_metrics()
        if duck_metrics is not None:
            logger.info(
                "[StreamingPipeline] session duck metrics: %s",
                duck_metrics,
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
        if not hasattr(self, "_session_closed_event"):
            self._session_closed_event = asyncio.Event()
        if not hasattr(self, "_idle_timeout_sec"):
            self._idle_timeout_sec = 0.0
        if not hasattr(self, "_on_idle_disconnect"):
            self._on_idle_disconnect = None
        if not hasattr(self, "_on_session_end"):
            self._on_session_end = None
        if not hasattr(self, "_idle_disconnect_grace_sec"):
            turn_policy = getattr(self, "_turn_policy", TurnPolicyConfig())
            self._idle_disconnect_grace_sec = (
                turn_policy.idle.disconnect_grace_ms / 1000.0
            )
        if not hasattr(self, "_idle_end_reason"):
            self._idle_end_reason = SESSION_END_IDLE_NORMAL
        if not hasattr(self, "_idle_watchdog_controller"):
            self._idle_watchdog_controller = IdleWatchdog(
                timeout_sec=self._idle_timeout_sec,
                get_session=lambda: getattr(self, "_session", None),
                get_room=lambda: getattr(self, "_room", None),
                get_timeline=lambda: getattr(self, "_timeline", None),
                session_closed_event=self._session_closed_event,
                on_idle_disconnect=self._on_idle_disconnect,
                on_session_end=self._on_session_end,
                disconnect_grace_sec=self._idle_disconnect_grace_sec,
                idle_end_reason=self._idle_end_reason,
            )
        self._idle_watchdog_controller.timeout_sec = self._idle_timeout_sec
        self._idle_watchdog_controller.disconnect_grace_sec = self._idle_disconnect_grace_sec

    def _ensure_room_data_handler(self) -> None:
        if not hasattr(self, "_room_data"):
            self._room_data = RoomDataHandler(
                get_timeline=lambda: getattr(self, "_timeline", None),
            )

    def _publish_companion_ui_state(self, state: str, reason: str) -> None:
        """Best-effort state bridge for thin clients such as ESP32 displays."""
        room = getattr(self, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        payload = {
            "schema_v": WIRE_SCHEMA_VERSION,
            "type": COMPANION_UI_STATE_TOPIC,
            "state": state,
            "reason": reason,
            "ts_ms": int(time.time() * 1000),
        }

        async def _send() -> None:
            await local.publish_data(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=COMPANION_UI_STATE_TOPIC,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.debug(
                    "[StreamingPipeline] failed to publish companion UI state",
                    exc_info=True,
                )

        task.add_done_callback(_log_failure)

    def _publish_client_control(
        self,
        op: str,
        *,
        reason: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Best-effort session-local command for thin clients.

        Uses the shared ``eidolon.control`` topic with ``src.type=channel`` so
        clients can distinguish it from Hub's audited cross-session commands.
        """
        room = getattr(self, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            logger.warning(
                "[StreamingPipeline] skipped client control op=%s reason=%s "
                "turn_id=%s because local participant is unavailable",
                op,
                reason,
                getattr(getattr(self, "_timeline", None), "turn_id", ""),
            )
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "[StreamingPipeline] skipped client control op=%s reason=%s "
                "because no running event loop is available",
                op,
                reason,
            )
            return

        timeline = getattr(self, "_timeline", None)
        turn_id = getattr(timeline, "turn_id", "") if timeline is not None else ""
        envelope = build_session_client_control_envelope(
            op=op,
            reason=reason,
            payload=payload,
            turn_id=turn_id,
        )
        control_payload = envelope["payload"]
        self._record_client_control_event(
            timeline=timeline,
            op=op,
            reason=reason,
            turn_id=turn_id,
        )

        outcome = control_payload.get("outcome")
        logger.info(
            "[StreamingPipeline] queued client control op=%s reason=%s outcome=%s "
            "turn_id=%s topic=%s",
            op,
            reason,
            outcome,
            turn_id,
            CONTROL_TOPIC,
        )

        async def _send() -> None:
            await local.publish_data(
                json.dumps(envelope, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=CONTROL_TOPIC,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.warning(
                    "[StreamingPipeline] failed to publish client control op=%s "
                    "reason=%s outcome=%s turn_id=%s topic=%s",
                    op,
                    reason,
                    outcome,
                    turn_id,
                    CONTROL_TOPIC,
                    exc_info=True,
                )
                return
            logger.info(
                "[StreamingPipeline] published client control op=%s reason=%s "
                "outcome=%s turn_id=%s topic=%s",
                op,
                reason,
                outcome,
                turn_id,
                CONTROL_TOPIC,
            )

        task.add_done_callback(_log_failure)

    def _agent_state_to_companion_ui_state(self, state: str) -> str:
        if state == "thinking":
            return "thinking"
        if state == "speaking":
            return "speaking"
        return "listening"

    def _on_agent_state_changed(self, event: Any) -> None:
        """Forward agent state changes through BasePipeline and session effects."""
        self._ensure_runtime_defaults()
        super()._on_agent_state_changed(event)
        self._ensure_agent_state_effect_handler()
        self._agent_state_effects.handle(event)
        new = getattr(event, "new_state", "")
        if new:
            self._publish_companion_ui_state(
                self._agent_state_to_companion_ui_state(new),
                f"agent_state:{new}",
            )

    def _publish_user_state_companion_ui(self, *, old: str, new: str) -> None:
        if new == "speaking":
            self._publish_companion_ui_state("listening", "user_state:speaking")
        elif old == "speaking" and new == "listening":
            self._publish_companion_ui_state("listening", "user_state:listening")
        elif new == "away":
            self._publish_companion_ui_state("idle", "user_state:away")

    def _sync_stt_presence_from_user_state(self, *, old: str, new: str) -> None:
        """Translate framework user_state into plugin-level STT presence."""

        if new == "away":
            self._session_signals.signal_stt_user_away()
        elif old == "away" and new in ("listening", "speaking"):
            self._session_signals.signal_stt_user_present()

    def _handle_user_speaking_started(self) -> None:
        self._ensure_user_turn_coordinator()
        merge_continuation = self._user_turns.can_merge_new_speech()
        self._skip_commit_after_interrupt_cancel = False
        self._suppress_transcripts_until_next_speech = False
        self._cancel_deferred_low_eot_commit("new_speech_started")
        if not merge_continuation:
            self._cancel_pending_voiceprint_commits("new_speech_started")
            self._completed_turn_voiceprint_task = None
            self._completed_turn_voiceprint_result = None
            self._completed_turn_voiceprint_timeline = None
            self._reset_candidate_voiceprint_tasks()
        self._callbacks.on_user_started_speaking()
        self._user_speaking_start_time = time.monotonic()
        if not merge_continuation or self._timeline is None:
            self._timeline = TurnTimeline(generate_turn_id())
            self._timeline_debug_flushed = False
            if self._room is not None:
                self._timeline.set_attr("room_name", self._room.name or "")
        self._user_turns.start_speech(timeline=self._timeline)
        self._timeline.mark("speech_started_at")
        self._apply_pending_explicit_client_preempt(self._timeline)
        self._apply_pending_client_control_events(self._timeline)
        self._voiceprint_turns.start_turn(timeline=self._timeline)
        self._apply_pending_stt_provider_events()
        self._observe_stt_turn_audio()
        # Immediately clear stale text so EOT only sees text from THIS speech turn.
        self._latest_asr_text = ""

        # Feed VAD signal into EOT model so VADState reflects user activity.
        self._get_eot_model().update_vad(True)

        if self._uses_livekit_native_adaptive_interruption():
            if self._timeline is not None:
                self._timeline.set_attr("interruption_owner", "livekit_native_adaptive")
            return

        # Immediately fade agent output to silence and arm the suspend-window
        # fallback. EOT decisions in the semantic interrupt handler will resolve
        # SUSPENDED output before the timeout fires in the typical case.
        self._attention_effects.handle_speaking_started()

        # If agent is speaking and interruptions are allowed, EOT check is
        # triggered synchronously in _on_user_transcribed as soon as STT delivers
        # the first transcript (INTERIM or FINAL) -- no polling needed.
        if self._ducking.is_suspended:
            self._interruption_orchestrator.start_candidate(
                timeline=self._timeline,
            )

    def _handle_user_speaking_stopped(self) -> None:
        self._user_speaking_start_time = None
        if self._timeline is not None:
            self._timeline.mark("speech_stopped_at")
        voiceprint_task = self._voiceprint_turns.finish_turn()
        self._completed_turn_voiceprint_task = voiceprint_task
        self._completed_turn_voiceprint_result = None
        self._completed_turn_voiceprint_timeline = self._timeline

        eot_model = self._get_eot_model()
        eot_model.update_vad(False)

        if self._soft_interrupt_is_active():
            logger.info(
                "[StreamingPipeline] user fell silent during soft interrupt; "
                "false interruption, cancelling"
            )
            self._cancel_soft_interrupt()

        defer_post_speech_evidence = False
        # VAD silence is not itself a false-interruption decision. If the agent
        # is suspended and no transcript has arrived yet, let the interruption
        # owner keep the candidate alive for delayed STT evidence before resume.
        if self._ducking.is_suspended and not self._uses_livekit_native_adaptive_interruption():
            defer_post_speech_evidence = (
                self._interruption_orchestrator.defer_false_resume_after_speech_end(
                    transcript=self._latest_asr_text,
                    duck_suspended=True,
                )
            )
            if not defer_post_speech_evidence:
                decision = self._turn_runtime.user_silent_decision(self._latest_asr_text)
                self._decision_effects.apply(
                    decision,
                    resolved_reason="user_silent",
                    transcript=self._latest_asr_text,
                    vad_active=False,
                )

        self._callbacks.on_user_ended_speaking()
        self._skip_commit_after_interrupt_cancel = False
        if defer_post_speech_evidence:
            self._remember_candidate_voiceprint_task(voiceprint_task)
            return
        if self._session is None:
            self._latest_asr_text = ""
            return

        transcript = self._user_turns.selected_text or self._latest_asr_text
        if transcript:
            self._remember_candidate_voiceprint_task(voiceprint_task)
        if self._user_turns.active is None and transcript:
            if self._timeline is None:
                self._timeline = TurnTimeline(generate_turn_id())
                self._timeline_debug_flushed = False
            self._user_turns.start_speech(timeline=self._timeline)
            self._apply_pending_explicit_client_preempt(self._timeline)
            self._apply_pending_client_control_events(self._timeline)
            self._user_turns.add_transcript(transcript, is_final=True)
        low_evidence_reason = self._playback_low_evidence_reject_reason(
            transcript=transcript,
            eot_model=eot_model,
        )
        if low_evidence_reason:
            logger.info(
                "[StreamingPipeline] rejecting playback low-evidence turn reason=%s "
                "transcript=%r",
                low_evidence_reason,
                transcript[:80],
            )
            self._user_turns.reject_active(low_evidence_reason)
            eot_model.reset()
            self._clear_session_user_turn(low_evidence_reason)
            self._reset_candidate_voiceprint_tasks()
            self._latest_asr_text = ""
            return
        should_defer = self._should_defer_low_eot_commit(
            transcript=transcript,
            eot_model=eot_model,
        )
        decision = self._user_turns.finish_speech(
            eot_score=getattr(
                eot_model,
                "current_eot_score",
                getattr(eot_model, "_current_eot_score", None),
            ),
            should_defer=should_defer,
        )
        if decision.action == "reject":
            eot_model.reset()
            self._clear_session_user_turn(decision.reason)
            self._reset_candidate_voiceprint_tasks()
            self._latest_asr_text = ""
        elif decision.action == "defer":
            self._schedule_deferred_low_eot_commit(
                verify_task=None,
                eot_model=eot_model,
                transcript=decision.transcript,
                timeline=self._timeline,
                delay_sec=decision.delay_sec,
            )
        else:
            self._schedule_voiceprint_gated_commit(
                verify_task=self._candidate_voiceprint_gate_task(),
                eot_model=eot_model,
                transcript=decision.transcript or transcript,
                timeline=self._timeline,
            )
            self._latest_asr_text = ""

    def _on_user_state_changed(self, event: Any) -> None:
        self._ensure_runtime_defaults()
        try:
            state_event = FullDuplexUserStateEvent.from_event(event)
            old = state_event.old_state
            new = state_event.new_state
            logger.info("[StreamingPipeline] user_state: %s -> %s", old, new)
            self._publish_user_state_companion_ui(old=old, new=new)
            self._sync_stt_presence_from_user_state(old=old, new=new)
            if state_event.started_speaking:
                self._handle_user_speaking_started()
            elif state_event.stopped_speaking:
                self._handle_user_speaking_stopped()
        except Exception:
            logger.exception("[StreamingPipeline] error in _on_user_state_changed")

    def _record_accepted_transcript_event(
        self,
        transcript_event: FullDuplexTranscriptEvent,
    ) -> None:
        if not transcript_event.has_transcript:
            return

        # Real recognized speech (interim or final) — keeps the session
        # alive. Empty/noise transcripts deliberately don't, so a silent
        # room still trips the idle watchdog.
        self._mark_activity()
        self._latest_asr_text = transcript_event.transcript
        orchestrator = getattr(self, "_interruption_orchestrator", None)
        if orchestrator is not None and not self._uses_livekit_native_adaptive_interruption():
            orchestrator.note_transcript(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
        self._ensure_user_turn_coordinator()
        self._user_turns.add_transcript(
            transcript_event.transcript,
            is_final=transcript_event.is_final,
        )
        if self._timeline is not None:
            self._timeline.mark(transcript_event.timeline_mark)
        # Round 8 R8.5.c: drive phase tracking + ONNX-debounced
        # scoring on every ASR event (interim + final). The 200ms
        # debounce inside update_asr coexists with EotManager's 50ms
        # cache — both contribute to keeping CPU bounded under the
        # ~100ms FunASR interim cadence.
        try:
            self._get_eot_model().update_asr(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )
        except Exception:
            logger.exception("[StreamingPipeline] eot_model.update_asr failed")

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
        transcript_event = FullDuplexTranscriptEvent.from_event(event)
        admission = self._ensure_transcript_admission_gate().evaluate(transcript_event)
        if not admission.accepted:
            if admission.reason == "suppressed_until_next_speech":
                logger.info(
                    "[StreamingPipeline] dropping post-turn transcript after "
                    "voiceprint ownership gate transcript=%r final=%s",
                    admission.transcript[:80],
                    admission.is_final,
                )
            elif admission.reason == "agent_echo":
                logger.info(
                    "[StreamingPipeline] dropping agent-echo transcript during playback "
                    "transcript=%r",
                    admission.transcript[:80],
                )
            return
        self._record_accepted_transcript_event(transcript_event)

        # Event-driven EOT check: react immediately when STT delivers text,
        # instead of polling for it. This ensures we analyze the CURRENT
        # speech turn's text, not a stale one from a previous turn.
        agent_is_speaking = self._agent_output_active_for_interrupts(
            participant_identity=transcript_event.speaker_id,
        )
        interrupt_window_active = self._interrupt_window_active()
        semantic_gate = evaluate_semantic_interrupt_gate(
            allow_interruptions=self._allow_interruptions,
            native_adaptive=self._uses_livekit_native_adaptive_interruption(),
            transcript=transcript_event.transcript,
            agent_output_active=agent_is_speaking,
            interrupt_window_active=interrupt_window_active,
            decision_suppressed=self._interrupt_decision_suppressed(),
        )
        if semantic_gate.should_forward_and_stop:
            if semantic_gate.reason == "decision_suppressed":
                logger.debug("[StreamingPipeline] interrupt decision suppressed after cancel")
            super()._on_user_transcribed(event)
            return
        if semantic_gate.needs_attention:
            semantic_gate = semantic_gate.with_attention_result(
                self._attention_effects.allows_eot_check(
                    transcript_event.transcript,
                    speaker_id=transcript_event.speaker_id,
                )
            )
            if semantic_gate.should_forward_and_stop:
                super()._on_user_transcribed(event)
                return
        if semantic_gate.should_run:
            self._semantic_interrupts.run(
                transcript_event.transcript,
                is_final=transcript_event.is_final,
            )

        super()._on_user_transcribed(event)

    def _interrupt_decision_suppressed(self) -> bool:
        """Ignore residual ASR after a confirmed interrupt cancel."""
        return time.monotonic() < self._suppress_commit_after_interrupt_until

    def _interrupt_window_active(self) -> bool:
        """Return true while an actual interrupt decision window is open."""
        return self._ducking.is_suspended or self._soft_interrupt_is_active()

    def _agent_output_active_for_interrupts(
        self,
        *,
        participant_identity: str | None = None,
    ) -> bool:
        """Return true when user speech should be evaluated as an interrupt.

        LiveKit's internal agent state can briefly disagree with the browser or
        device playback state. For hot-path interruption, user experience cares
        about audible agent output, so a fresh client ``agent_speaking`` signal
        is also authoritative.
        """
        self._ensure_ducking_controller()
        if self._ducking.is_cancelled:
            return False
        if (
            getattr(self, "_state", PipelineState.IDLE) == PipelineState.SPEAKING
            or self._ducking.is_suspended
        ):
            return True
        self._ensure_room_data_handler()
        states = self._room_data.client_audio_states
        if not states:
            return False
        max_age_sec = (
            self._turn_policy.attention.client_state_max_age_ms / 1000.0
            if hasattr(self, "_turn_policy")
            else 2.0
        )
        now = time.monotonic()
        if participant_identity:
            client = states.get(participant_identity)
            if (
                client is not None
                and client.is_fresh(now=now, max_age_sec=max_age_sec)
                and client.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
            ):
                return True
        return any(
            state.is_fresh(now=now, max_age_sec=max_age_sec)
            and state.playback_state == PLAYBACK_STATE_AGENT_SPEAKING
            for state in states.values()
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

        if self._agent_output_active_for_interrupts(
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

        if self._agent_output_active_for_interrupts():
            self._duck_cancel_and_interrupt(force=True)
            return
        self._cancel_silent_agent_generation_for_explicit_preempt()

    def _cancel_silent_agent_generation_for_explicit_preempt(self) -> None:
        self._ensure_runtime_defaults()
        self._cancel_stable_signal_timer()
        self._ducking.cancel_output()
        if self._timeline is not None:
            self._timeline.set_attr(
                "explicit_client_generation_preempt",
                {
                    "reason": "explicit_client_ptt",
                    "agent_state": (
                        self._state.name
                        if hasattr(getattr(self, "_state", None), "name")
                        else str(getattr(self, "_state", "unknown"))
                    ),
                    "playback_state": "not_audible",
                },
            )
        logger.info(
            "[StreamingPipeline] explicit client preempted silent agent generation state=%s",
            (
                self._state.name
                if hasattr(getattr(self, "_state", None), "name")
                else getattr(self, "_state", "unknown")
            ),
        )
        self._interrupt_current_turn(force=True)

    def _latest_client_audio_state(
        self,
        *,
        participant_identity: str | None = None,
    ) -> ClientAudioState | None:
        self._ensure_runtime_defaults()
        max_age_sec = self._turn_policy.attention.client_state_max_age_ms / 1000.0
        self._ensure_room_data_handler()
        return self._room_data.latest_client_audio_state(
            participant_identity=participant_identity,
            max_age_sec=max_age_sec,
        )

    def _interrupt_current_turn(self, *, force: bool = False) -> None:
        """Interrupt the currently in-progress agent turn via session.interrupt().

        ``force`` skips the ``allow_interruptions`` gate and passes through to
        ``session.interrupt(force=True)``, which cancels even a speech handle
        that was started with interruptions disabled. Used by explicit client
        controls; policy-driven interrupts leave it False.
        """
        self._ensure_ducking_controller()
        if not force and not self._allow_interruptions:
            return
        # The cancelled-output short-circuit is only for the policy path (avoid a
        # redundant session.interrupt after cancel_output). The FORCED explicit
        # client path MUST still call session.interrupt(force=True):
        # _duck_cancel_and_interrupt already set is_cancelled=True, but without
        # this an uninterruptible speech handle is never
        # actually ended → agent_state stays "speaking" → the captured barge-in
        # turn can't commit → no reply. force must reach the framework.
        if not force and self._ducking.is_cancelled:
            logger.debug("[StreamingPipeline] interrupt skipped — output already CANCELLED")
            return

        if self._session is not None:
            self._session.interrupt(force=force)

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

    async def _soft_interrupt_timeout_task(self) -> None:
        """Timer task: fires after _soft_interrupt_timeout → upgrade to hard interrupt."""
        self._ensure_soft_interrupt_controller()
        await self._soft_interrupt.run_timeout_task()

    def _cancel_soft_interrupt(self) -> None:
        """Cancel soft interrupt (detected as a false interruption)."""
        self._ensure_soft_interrupt_controller()
        self._soft_interrupt.cancel()

    def _handle_hold_decision(
        self,
        decision: Decision,
        transcript: str,
        eot_score: float | None,
        vad_active: bool | None,
    ) -> None:
        if decision.hold_recheck_ms is None and not decision.reason.startswith(
            STABLE_SIGNAL_WAIT_REASON_PREFIX
        ):
            return
        if not self._ducking.is_suspended:
            return
        if not transcript.strip():
            return
        recheck_ms = decision.hold_recheck_ms
        if recheck_ms is None:
            recheck_ms = self._turn_policy.interrupt.correction_topic_stability_window_ms
        timeout_sec = max(0.0, recheck_ms) / 1000.0
        self._cancel_stable_signal_timer()
        logger.info(
            "[StreamingPipeline] stable-signal recheck armed "
            "timeout=%.3fs reason=%s text=%r eot_score=%s vad_active=%s",
            timeout_sec,
            decision.reason,
            transcript[:80],
            f"{eot_score:.2f}" if eot_score is not None else "None",
            vad_active,
        )
        self._stable_signal_timer = asyncio.create_task(
            self._stable_signal_recheck(timeout_sec, transcript)
        )

    async def _stable_signal_recheck(self, timeout_sec: float, transcript: str) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(timeout_sec)
            if not self._ducking.is_suspended:
                return
            latest = (self._latest_asr_text or transcript).strip()
            if not latest:
                return
            logger.info(
                "[StreamingPipeline] stable-signal recheck firing text=%r",
                latest[:80],
            )
            self._semantic_interrupts.run(latest, is_final=False)
        except asyncio.CancelledError:
            return
        finally:
            if self._stable_signal_timer is current_task:
                self._stable_signal_timer = None

    def _cancel_stable_signal_timer(self) -> None:
        task = getattr(self, "_stable_signal_timer", None)
        if task is not None and not task.done():
            task.cancel()
        self._stable_signal_timer = None

    def _ensure_soft_interrupt_controller(self) -> None:
        if not hasattr(self, "_soft_interrupt_timeout"):
            self._soft_interrupt_timeout = (
                self._turn_runtime.decision_timeout_sec if hasattr(self, "_turn_runtime") else 0.5
            )
        if not hasattr(self, "_soft_interrupt"):
            self._soft_interrupt = SoftInterruptController(
                timeout_sec=self._soft_interrupt_timeout,
                on_timeout=lambda: self._interrupt_current_turn(),
            )
        self._soft_interrupt.timeout_sec = self._soft_interrupt_timeout

    def _soft_interrupt_is_active(self) -> bool:
        self._ensure_soft_interrupt_controller()
        return self._soft_interrupt.active

    # ------------------------------------------------------------------
    # DuckingMixer integration
    # ------------------------------------------------------------------
    #
    # State machine driven by VAD + EOT events:
    #
    #   user_state listening → speaking
    #     → _duck_and_arm_timeout()
    #         mixer.duck()  (50 ms fade-out to silence)
    #         start _ducking.timeout_task (default 0.5 s fallback)
    #
    #   SemanticInterruptHandler.run (per STT interim/final):
    #     strong_interrupt_intent OR score >= duck_early_cancel_score_threshold
    #       → _duck_cancel_and_interrupt()  (real interrupt, no resume)
    #     score <= duck_early_resume_score_threshold (and > 0)
    #       → _duck_unduck()  (false interrupt, smooth fade-in)
    #     mid-band → leave SUSPENDED, let timeout decide
    #
    #   _ducking.timeout_task fires (no decision in window):
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
            self._record_duck_event(
                "duck_started",
                vad_to_duck_ms=(
                    (now - self._user_speaking_start_time) * 1000
                    if self._user_speaking_start_time is not None
                    else None
                ),
                timeout_sec=cfg.duck_suspend_timeout_sec,
                cooldown_sec=cfg.duck_cooldown_sec,
            )
        self._callbacks.on_duck_started()
        vad_to_duck_ms = 0.0
        if self._user_speaking_start_time is not None:
            vad_to_duck_ms = (now - self._user_speaking_start_time) * 1000
        logger.info(
            "[StreamingPipeline] duck armed  vad→duck=%.1fms  timeout=%.2fs  cooldown=%.2fs",
            vad_to_duck_ms,
            cfg.duck_suspend_timeout_sec,
            cfg.duck_cooldown_sec,
        )
        self._ducking.timeout_task = asyncio.create_task(
            self._duck_deadline.run(cfg.duck_suspend_timeout_sec)
        )

    def _record_duck_event(self, event: str, **fields: object) -> None:
        timeline = self._timeline
        if timeline is None:
            return
        payload = {"event": event, **fields}
        events = list(timeline.attrs.get("duck_events") or ())
        events.append(payload)
        timeline.set_attr("duck_events", events)
        timeline.set_attr("duck_last_event", payload)

    def _append_timeline_debug(self, reason: str, *, clear: bool = False) -> None:
        if self._timeline is None or self._timeline_debug_flushed:
            return
        self._timeline.set_attr("timeline_flush_reason", reason)
        self._timeline.append_debug_jsonl(self._observability.timeline_debug_path)
        self._timeline_debug_flushed = True
        if clear:
            self._timeline = None

    def _duck_cancel_and_interrupt(self, *, force: bool = False) -> None:
        """Confirm interrupt: discard buffer + cancel TTS generation.

        ``force`` propagates to ``session.interrupt(force=True)`` so an explicit
        client request interrupts even when the current speech handle disallows
        interruptions. Policy-driven callers leave it False so the
        ``allow_interruptions`` gate still applies.
        """
        self._ensure_runtime_defaults()
        if self._ducking.is_cancelled:
            logger.debug(
                "[StreamingPipeline] duplicate duck cancel ignored — output already CANCELLED"
            )
            return
        commit_post_speech_candidate = (
            self._interruption_orchestrator.should_commit_after_confirmed_cancel()
        )
        post_speech_transcript = (
            self._interruption_orchestrator.current_transcript
            if commit_post_speech_candidate
            else ""
        )
        stats = self._ducking.stats()
        logger.info(
            "[StreamingPipeline] duck resolved  reason=eot_cancel  "
            "action=cancel  suspend_ms=%.0f  discarded=%d frames (%.3fs)",
            stats.suspend_ms,
            stats.buffered_frames,
            stats.buffered_sec,
        )
        self._cancel_stable_signal_timer()
        self._snapshot_interrupted_context()
        self._publish_client_control(CONTROL_OP_PLAYBACK_STOP, reason="interrupt_cancel")
        self._ducking.cancel_output()
        self._interruption_orchestrator.resolve(
            action="cancel",
            reason="eot_cancel",
        )
        self._callbacks.on_duck_resolved("cancel")
        if self._timeline is not None:
            self._record_duck_event(
                "duck_cancelled",
                reason="eot_cancel",
                suspend_ms=stats.suspend_ms,
                buffered_frames=stats.buffered_frames,
                buffered_sec=stats.buffered_sec,
                drop_buffered=True,
            )
            self._timeline.mark("interrupt_resolved_at")
            self._timeline.set_attr("cancel_reason", "eot_cancel")
            # Output interruption is no longer a terminal user-turn event.
            # Keep the timeline open so the same owner utterance can still be
            # assembled, voiceprint-gated, and committed after the agent yields.
        self._skip_commit_after_interrupt_cancel = True
        self._suppress_commit_after_interrupt_until = (
            time.monotonic() + self._cancel_residual_commit_suppress_sec()
        )
        self._interrupt_current_turn(force=force)
        if commit_post_speech_candidate:
            committed = self._commit_post_speech_interruption_candidate(
                "post_speech_confirmed_cancel",
                transcript_override=post_speech_transcript,
            )
            if committed:
                self._skip_commit_after_interrupt_cancel = False

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
            waiting_post_speech_evidence = (
                self._interruption_orchestrator.awaiting_post_speech_evidence
            )
            stats = self._ducking.stats()
            logger.info(
                "[StreamingPipeline] duck resolved  reason=%s  "
                "action=unduck(drop_buffered=%s)  suspend_ms=%.0f  "
                "buffered=%d frames (%.3fs)",
                reason,
                drop_buffered,
                stats.suspend_ms,
                stats.buffered_frames,
                stats.buffered_sec,
            )
            self._cancel_stable_signal_timer()
            self._ducking.unduck_if_suspended(drop_buffered=drop_buffered)
            self._interruption_orchestrator.resolve(
                action="rollback",
                reason=reason,
            )
            self._callbacks.on_duck_resolved("unduck")
            if self._timeline is not None:
                self._record_duck_event(
                    "duck_unducked",
                    reason=reason,
                    suspend_ms=stats.suspend_ms,
                    buffered_frames=stats.buffered_frames,
                    buffered_sec=stats.buffered_sec,
                    drop_buffered=drop_buffered,
                )
                self._timeline.mark("interrupt_resolved_at")
                self._timeline.set_attr("rollback_reason", reason)
            if waiting_post_speech_evidence:
                reject_reason = (
                    "post_speech_evidence_timeout"
                    if reason == "timeout"
                    else f"post_speech_false_interruption:{reason}"
                )
                self._reject_post_speech_interruption_candidate(reject_reason)

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
        self._ensure_ducking_controller()
        self._interrupted_context.snapshot(
            session=getattr(self, "_session", None),
            factory=getattr(self, "_factory", None),
            duck_mixer=self._ducking.mixer,
            config=self._get_eot_model()._config,
        )
        context = self._interrupted_context.last_context
        timeline = getattr(self, "_timeline", None)
        if timeline is not None and context is not None:
            timeline.set_attr(
                "interrupted_context",
                {
                    "source": context.get("source"),
                    "played_seconds": context.get("played_seconds"),
                    "text_preview": str(context.get("text") or "")[:120],
                },
            )

    def _inject_interrupted_context(self) -> None:
        """Inject interrupted context into the conversation history.

        Called before ``commit_user_turn()`` so the LLM sees the context
        when generating its next response.
        """
        self._ensure_interrupted_context_manager()
        self._interrupted_context.inject(
            session=getattr(self, "_session", None),
            config=self._get_eot_model()._config,
        )

    def _ensure_interrupted_context_manager(self) -> None:
        if not hasattr(self, "_interrupted_context"):
            self._interrupted_context = InterruptedContextManager()


def _count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _voiceprint_result_is_inconclusive(result: Any) -> bool:
    reason = str(getattr(result, "commit_reason", "") or "").lower()
    return reason in {"audio_too_short", "insufficient_audio", "too_short"}
