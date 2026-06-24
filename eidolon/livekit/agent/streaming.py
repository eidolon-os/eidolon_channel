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
import json
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
from .client_audio_state import (
    CLIENT_AUDIO_STATE_TOPIC,
    PLAYBACK_STATE_AGENT_SPEAKING,
    ClientAudioState,
)
from .context import InterruptedContextManager
from .runtime.interaction_mode import (
    INTENT_PROACTIVE,
    INTENT_USER_INITIATED,
    INTERACTION_MODE_FULL_DUPLEX,
    INTERACTION_MODE_HALF_DUPLEX,
)
from .turn_policy import (
    Decision,
    TurnPolicyRuntime,
    eot_kwargs_from_turn_policy,
)
from .turn_policy.constants import STABLE_SIGNAL_WAIT_REASON_PREFIX
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
    UserTurnCoordinator,
    VoiceprintTurnObserver,
)

logger = logging.getLogger("agent")

_INTERRUPT_CANCEL_RESIDUAL_COMMIT_SUPPRESS_SEC = 2.0
_LOW_EOT_COMMIT_GRACE_MAX_SEC = 2.0
_SHORT_STATEMENT_DEFER_MAX_CJK_CHARS = 12
_COMPANION_UI_STATE_TOPIC = "eidolon.ui_state"
_CLIENT_CONTROL_TOPIC = "eidolon.control"


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


def _message_text(message: Any) -> str:
    text_content = getattr(message, "text_content", None)
    if isinstance(text_content, str):
        return text_content
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content or "")


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
        on_session_end: Callable[[str], Awaitable[None]] | None = None,
        on_session_closed: Callable[[], Awaitable[None]] | None = None,
        interaction_mode: str = INTERACTION_MODE_FULL_DUPLEX,
        session_intent: str = INTENT_USER_INITIATED,
    ) -> None:
        super().__init__(factory=factory, callbacks=callbacks)
        # Session-level interaction mode (plan Phase 5), authoritative from the
        # LiveKit token metadata that hub stamps. half_duplex = push-to-talk
        # appliance (mic device-gated; explicit tap-to-stop; no barge-in / no
        # server-side echo suppression); full_duplex = open mic + hardware AEC
        # (server-judged barge-in + content echo gate). Defaults to full_duplex
        # so a directly-constructed pipeline keeps the open-mic behaviour; the
        # worker (server.py) always passes the resolved mode.
        self._interaction_mode = interaction_mode
        self._is_half_duplex = interaction_mode == INTERACTION_MODE_HALF_DUPLEX
        # Session intent (plan §3.2/§3.3) — orthogonal to interaction_mode. Drives
        # the idle window + teardown reason + whether the half_duplex keep-alive
        # exemption applies. Defaults user_initiated for a directly-constructed
        # pipeline; the worker passes the resolved value.
        self._session_intent = session_intent
        self._is_proactive = session_intent == INTENT_PROACTIVE
        # Half-duplex (manual turn_detection) PTT turn boundary (plan §10):
        #   _last_ptt_held       — track the ptt edge so release (True→False)
        #                          commits exactly one turn.
        #   _ptt_turn_had_speech — 守空 guard: only commit if some speech arrived
        #                          this hold (manual mode would otherwise fire an
        #                          empty EOU on a no-speech press). Reset on press
        #                          and after the release commit.
        self._last_ptt_held = False
        self._ptt_turn_had_speech = False
        self._turn_policy = turn_policy or TurnPolicyConfig()
        self._turn_runtime = TurnPolicyRuntime(self._turn_policy)
        self._observability = observability or ObservabilityConfig()
        self._timeline: TurnTimeline | None = None
        self._timeline_debug_flushed = False
        self._skip_commit_after_interrupt_cancel = False
        self._suppress_commit_after_interrupt_until = 0.0
        # Ducking state is shared by several effect handlers. It must exist
        # before those handlers are built, because AgentStateEffectHandler keeps
        # a direct reference to the controller.
        self._ducking = OutputDuckingController()
        self._decision_effects = self._build_decision_effect_applier()
        self._attention_effects = self._build_attention_effect_handler()
        self._session_signals = self._build_session_signal_bridge()
        self._turn_committer = UserTurnCommitter()
        self._user_turns = self._build_user_turn_coordinator()
        self._agent_state_effects = self._build_agent_state_effect_handler()
        self._semantic_interrupts = self._build_semantic_interrupt_handler()
        self._duck_deadline = self._build_duck_suspend_timeout_handler()
        self._stable_signal_timer: asyncio.Task | None = None
        self._pending_voiceprint_commit_tasks: set[asyncio.Task] = set()
        self._candidate_voiceprint_tasks: list[asyncio.Task] = []
        self._deferred_low_eot_commit_task: asyncio.Task | None = None
        self._suppress_transcripts_until_next_speech = False
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
            append_timeline_snapshot=lambda timeline, reason: (
                self._append_turn_timeline_snapshot(timeline, reason)
            ),
        )
        self._voiceprint_turns = VoiceprintTurnObserver(
            service=getattr(self._factory, "voiceprint_service", None),
            runtime_admin=getattr(self._factory, "runtime_admin", None),
            sample_rate=audio_sample_rate,
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
        # Proactive report consumer: a background stream that lets the brain
        # speak unprompted (e.g. "your meeting notes are ready"). Started after
        # session.start(); torn down in shutdown(). Lazily wired so direct_llm
        # mode (no eidolon_agent gRPC backend) simply skips it.
        self._proactive_task: asyncio.Task | None = None
        self._proactive_subscriber: Any | None = None
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
        # Idle window + teardown reason + keep-alive exemption are chosen by
        # session_intent (plan §3.2/§3.3): a proactive wake-up nobody answers is
        # reclaimed on a short window with reason=proactive_done and gets NO
        # half_duplex keep-alive; a user session keeps the 60s window,
        # idle_normal_end, and the half_duplex keep-alive.
        if self._is_proactive:
            self._idle_timeout_sec: float = (
                self._turn_policy.idle.proactive_disconnect_after_idle_ms / 1000.0
            )
            self._idle_end_reason = "proactive_done"
            self._idle_keep_alive_half_duplex = False
        else:
            self._idle_timeout_sec = (
                self._turn_policy.idle.disconnect_after_idle_ms / 1000.0
            )
            self._idle_end_reason = "idle_normal_end"
            self._idle_keep_alive_half_duplex = True
        self._idle_watchdog_task: asyncio.Task | None = None
        self._last_activity_monotonic: float = 0.0
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
        self._idle_disconnect_grace_sec: float = 0.3
        self._idle_watchdog_controller = IdleWatchdog(
            timeout_sec=self._idle_timeout_sec,
            get_session=lambda: getattr(self, "_session", None),
            get_room=lambda: getattr(self, "_room", None),
            get_timeline=lambda: getattr(self, "_timeline", None),
            session_closed_event=self._session_closed_event,
            on_idle_disconnect=self._on_idle_disconnect,
            on_session_end=self._on_session_end,
            disconnect_grace_sec=self._idle_disconnect_grace_sec,
            is_half_duplex=lambda: self._is_half_duplex,
            idle_end_reason=self._idle_end_reason,
            keep_alive_half_duplex=self._idle_keep_alive_half_duplex,
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
            on_hold=self._handle_hold_decision,
        )

    def _ensure_decision_effect_applier(self) -> None:
        handler = getattr(self, "_decision_effects", None)
        if (
            handler is None
            or getattr(handler, "_turn_runtime", None) is not self._turn_runtime
        ):
            self._decision_effects = self._build_decision_effect_applier()

    def _build_attention_effect_handler(self) -> AttentionEffectHandler:
        return AttentionEffectHandler(
            turn_policy=self._turn_policy,
            turn_runtime=self._turn_runtime,
            get_agent_speaking=lambda: self._agent_output_active_for_interrupts(),
            get_duck_active=lambda: self._ducking.is_suspended,
            latest_client_audio_state=lambda participant_identity: (
                self._latest_client_audio_state(
                    participant_identity=participant_identity,
                )
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

    def _ensure_turn_committer(self) -> None:
        if not hasattr(self, "_turn_committer"):
            self._turn_committer = UserTurnCommitter()

    def _build_user_turn_coordinator(self) -> UserTurnCoordinator:
        delay = min(
            max(self._turn_policy.eot.tail_hang_silence_ms / 1000.0, 0.0),
            _LOW_EOT_COMMIT_GRACE_MAX_SEC,
        )
        return UserTurnCoordinator(
            merge_grace_sec=delay,
            statement_deferred_merge_grace_sec=max(delay, 3.5),
            voiceprint_deferred_merge_grace_sec=max(delay, 4.0),
            low_eot_delay_sec=delay,
        )

    def _ensure_user_turn_coordinator(self) -> None:
        if not hasattr(self, "_user_turns"):
            self._user_turns = self._build_user_turn_coordinator()

    def _cancel_pending_voiceprint_commits(self, reason: str) -> None:
        tasks = getattr(self, "_pending_voiceprint_commit_tasks", set())
        for task in list(tasks):
            if not task.done():
                logger.info(
                    "[StreamingPipeline] cancelling pending voiceprint-gated "
                    "commit reason=%s",
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
        # Half-duplex (push-to-talk): button release is an explicit, final
        # end-of-turn — never defer for a continuation that won't come.
        if self._is_half_duplex:
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

    @staticmethod
    def _looks_like_short_statement_continuation(transcript: str) -> bool:
        text = transcript.strip()
        if not text:
            return False
        if any(mark in text for mark in ("？", "?", "！", "!")):
            return False
        cjk_chars = _count_cjk_chars(text)
        if cjk_chars <= 0:
            return False
        if cjk_chars > _SHORT_STATEMENT_DEFER_MAX_CJK_CHARS:
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
                _LOW_EOT_COMMIT_GRACE_MAX_SEC,
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
            "[StreamingPipeline] deferred low-EOT commit delay=%.3fs "
            "score=%s transcript=%r",
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
                    "[StreamingPipeline] deferred low-EOT commit skipped "
                    "reason=%s",
                    decision.reason,
                )
                return
            final_transcript = (
                decision.transcript.strip()
                or self._latest_asr_text.strip()
                or transcript
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
        if (
            not allowed
            and self._should_keep_waiting_merge_after_inconclusive_voiceprint(
                result,
                transcript=transcript,
            )
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
                "[StreamingPipeline] voiceprint gate blocked commit reason=%s "
                "transcript=%r",
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

    def _clear_session_user_turn(self, reason: str) -> None:
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

    async def _voiceprint_allows_completed_turn(self, *, new_message: Any) -> bool:
        """Gate LiveKit's final turn-completed hook with voiceprint ownership.

        This is the last public lifecycle boundary before LiveKit starts the
        LLM reply, so it catches both our explicit commit path and framework
        auto-EOU paths such as late STT FINAL delivery.
        """
        self._ensure_runtime_defaults()
        task = getattr(self, "_completed_turn_voiceprint_task", None)
        result = getattr(self, "_completed_turn_voiceprint_result", None)
        timeline = (
            getattr(self, "_completed_turn_voiceprint_timeline", None)
            or getattr(self, "_timeline", None)
        )
        completed_transcript = _message_text(new_message)
        if timeline is not None:
            timeline.set_attr(
                "framework_completed_turn",
                {
                    "text_preview": completed_transcript[:120],
                    "text_length": len(completed_transcript),
                },
            )
        if task is None and result is None:
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
                logger.exception(
                    "[StreamingPipeline] voiceprint gate failed in turn hook"
                )
                return False
            self._completed_turn_voiceprint_result = result

        allowed = bool(getattr(result, "commit_allowed", False))
        reason = str(getattr(result, "commit_reason", "") or "unknown")
        self._record_voiceprint_commit_gate(timeline, allowed=allowed, reason=reason)
        if allowed:
            stop_reason = self._non_semantic_completed_turn_reason(timeline)
            if stop_reason:
                self._cancel_deferred_low_eot_commit(stop_reason)
                self._ensure_user_turn_coordinator()
                self._user_turns.reject_active(stop_reason)
                self._clear_session_user_turn(stop_reason)
                self._flush_turn_timeline(timeline, stop_reason)
                logger.info(
                    "[StreamingPipeline] stopped framework completed turn "
                    "reason=%s transcript=%r",
                    stop_reason,
                    completed_transcript[:80],
                )
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
            "[StreamingPipeline] voiceprint gate stopped completed turn "
            "reason=%s transcript=%r",
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
        # Half-duplex (push-to-talk): button release is an explicit, final
        # end-of-turn — never hold it for a possible continuation.
        if self._is_half_duplex:
            return False
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
            "[StreamingPipeline] deferred inconclusive voiceprint result "
            "reason=%s transcript=%r",
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
            soft_interrupt_active=lambda: self._soft_interrupt_active,
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
                self._session is not None
                and self._session.user_state == "speaking"
            ),
            soft_interrupt_active=lambda: self._soft_interrupt_active,
            soft_interrupt_timeout=lambda: self._soft_interrupt_timeout,
            apply_decision=self._decision_effects.apply,
            record_decision_attrs=self._decision_effects.record_decision_attrs,
            publish_turn_control=self._decision_effects.publish_turn_control,
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
            apply_decision=self._decision_effects.apply,
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
                flush_timeline=lambda timeline, reason: self._flush_turn_timeline(
                    timeline,
                    reason,
                ),
                append_timeline_snapshot=lambda timeline, reason: (
                    self._append_turn_timeline_snapshot(timeline, reason)
                ),
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
        if not hasattr(self, "_is_half_duplex"):
            self._is_half_duplex = (
                self._interaction_mode == INTERACTION_MODE_HALF_DUPLEX
            )
        if not hasattr(self, "_suppress_transcripts_until_next_speech"):
            self._suppress_transcripts_until_next_speech = False
        if not hasattr(self, "_completed_turn_voiceprint_task"):
            self._completed_turn_voiceprint_task = None
        if not hasattr(self, "_completed_turn_voiceprint_result"):
            self._completed_turn_voiceprint_result = None
        if not hasattr(self, "_completed_turn_voiceprint_timeline"):
            self._completed_turn_voiceprint_timeline = None
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
        if not hasattr(self, "_voiceprint_turns"):
            factory = getattr(self, "_factory", None)
            self._voiceprint_turns = VoiceprintTurnObserver(
                service=getattr(factory, "voiceprint_service", None),
                runtime_admin=getattr(factory, "runtime_admin", None),
                sample_rate=getattr(self, "_audio_sample_rate", 16000),
                trust_paired_devices=getattr(
                    factory,
                    "voiceprint_trust_paired_devices",
                    True,
                ),
            )
        self._ensure_ducking_controller()
        self._ensure_decision_effect_applier()
        self._ensure_attention_effect_handler()
        self._ensure_session_signal_bridge()
        self._ensure_turn_committer()
        self._ensure_agent_state_effect_handler()
        self._ensure_semantic_interrupt_handler()
        self._ensure_duck_suspend_timeout_handler()
        self._ensure_room_data_handler()

    def _build_turn_handling(self) -> dict:
        """AgentSession ``turn_handling`` options (Round 8 R8.9: must live on the
        AgentSession, not the Agent — ``AgentActivity`` reads
        ``session._opts.turn_handling.interruption``).

        ``discard_audio_if_uninterruptible`` is **mode-aware**:
          - full_duplex: True (status quo). Speeches are interruptible
            (``enabled=True``) so this path rarely triggers; when an
            uninterruptible speech does occur, discarding open-mic audio is the
            right echo protection.
          - half_duplex: False. Here ``enabled=False`` makes every reply
            uninterruptible, so the default would discard the user's audio —
            including a DELIBERATE PTT tap-to-stop barge-in. The device gates the
            mic closed during playback, so any audio during playback is an
            intentional barge-in that MUST be captured (else the barge-in
            utterance is dropped → "empty_transcript" → no reply). The explicit
            PTT path (``_handle_explicit_client_interrupt``) does the actual cut;
            this just stops the framework from throwing the audio away.

        ``preemptive_generation`` (Phase 2, 2026-05-30): speculative brain
        generation gated via ``turn_policy.preemptive`` — hides the STT-final
        wait by starting the brain on a stable interim; framework reuses it if
        the final matches, else cancels via gRPC CancelTurn. ``preemptive_tts``
        stays gated by our commit so no partial audio leaks.
        """
        return {
            "interruption": {
                "enabled": self._allow_interruptions,
                "discard_audio_if_uninterruptible": not self._is_half_duplex,
                "false_interruption_timeout": self._false_interruption_timeout,
            },
            "preemptive_generation": {
                "enabled": self._turn_policy.preemptive.enabled,
                "preemptive_tts": self._turn_policy.preemptive.preemptive_tts,
            },
        }

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
            logger.info(
                "[StreamingPipeline] session closed event received, exiting run()"
            )
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
        """
        cb = getattr(self, "_on_session_closed", None)
        if cb is None:
            return
        try:
            await cb()
        except Exception:
            logger.exception(
                "[StreamingPipeline] on_session_closed (prompt room delete) failed"
            )

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
                "[StreamingPipeline] dropping proactive report (session closed) "
                "intent=%s",
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
        """Observe client-side audio hints and drive explicit (PTT) interrupts.

        The explicit-interrupt fast path is wired into the SAME registered
        ``data_received`` callback (via ``on_packet``) so a ``client.audio_state``
        ``ptt=True`` edge during playback actually interrupts the agent. (It used
        to be reachable only from ``_on_room_data_received``, which was never
        registered — dead code — so half-duplex barge-in never fired.)
        """
        self._ensure_room_data_handler()
        self._room_data.install(room, on_packet=self._on_client_room_packet)
        self._sync_room_data_compat_attrs()

    def _on_client_room_packet(self, packet: Any) -> None:
        # Runs after RoomDataHandler.handle_packet has stored the latest client
        # audio state (so do NOT handle_packet again here — that would double-count).
        self._sync_room_data_compat_attrs()
        self._handle_explicit_client_interrupt(packet)
        self._handle_ptt_turn_edges(packet)

    def _handle_ptt_turn_edges(self, packet: Any) -> None:
        """Half-duplex (manual) turn boundary: commit one turn on PTT release.

        With ``turn_detection="manual"`` nothing commits automatically; the PTT
        button is the sole turn boundary (plan §10). On the rising edge (press) we
        arm a fresh turn; on the falling edge (release) we commit exactly once —
        guarded by 守空 so a no-speech press doesn't fire an empty EOU.
        """
        if not self._is_half_duplex:
            return
        if getattr(packet, "topic", None) != CLIENT_AUDIO_STATE_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        identity = getattr(participant, "identity", "") or None
        state = self._latest_client_audio_state(participant_identity=identity)
        # Only react to a participant that actually publishes a ptt-bearing audio
        # state (the device). A stray packet from another identity yields no state
        # — ignore it rather than read it as ptt=False, which (mid-hold) would
        # fake a release edge and commit early. One voice room == one device (I4),
        # so a single held flag is sufficient.
        if state is None:
            return
        held = bool(state.ptt)
        was_held = self._last_ptt_held
        self._last_ptt_held = held
        if held and not was_held:
            # Press: start a fresh turn's 守空 accounting.
            self._ptt_turn_had_speech = False
            return
        if was_held and not held:
            self._commit_ptt_release_turn()

    def _commit_ptt_release_turn(self) -> None:
        """Commit the held turn on PTT release (manual turn_detection).

        Calls ``session.commit_user_turn`` exactly once, letting the framework
        wait (``transcript_timeout``) for a still-in-flight FINAL — so a tail like
        「北京的」 spoken just before release is included — and read its own
        complete accumulated transcript. NEVER calls ``clear_user_turn`` (that
        heavy reset is what dropped turns before, see §10). 守空: skip if no
        speech arrived this hold.
        """
        session = self._session
        if session is None:
            return
        if not self._ptt_turn_had_speech:
            logger.info(
                "[ptt-manual] release with no speech this hold; skipping commit (守空)"
            )
            return
        self._ptt_turn_had_speech = False
        try:
            self._get_eot_model().reset()
        except Exception:
            logger.debug("[ptt-manual] eot reset failed (non-fatal)", exc_info=True)
        try:
            session.commit_user_turn(
                transcript_timeout=self._stt_commit_transcript_timeout
            )
            logger.info(
                "[ptt-manual] PTT release → commit_user_turn(transcript_timeout=%.1fs)",
                self._stt_commit_transcript_timeout,
            )
        except Exception:
            logger.exception("[ptt-manual] commit_user_turn on release failed")

    def _on_room_data_received(self, packet: Any) -> None:
        # Full manual processing for direct callers/tests. The production path
        # registers handle_packet via RoomDataHandler.install and chains
        # _on_client_room_packet through its on_packet hook; this mirrors that.
        self._ensure_room_data_handler()
        self._room_data.handle_packet(packet)
        self._on_client_room_packet(packet)

    def _handle_explicit_client_interrupt(self, packet: Any) -> None:
        if getattr(packet, "topic", None) != CLIENT_AUDIO_STATE_TOPIC:
            return
        participant = getattr(packet, "participant", None)
        identity = getattr(participant, "identity", "") or None
        state = self._latest_client_audio_state(participant_identity=identity)
        # PTT (button press / tap-to-stop) is the only explicit client interrupt.
        # The device's energy-gate ``manual_interrupt`` guess was removed (it
        # falsely tripped on residual playback echo); barge-in for open-mic
        # full_duplex is now judged server-side from the clean transcript stream.
        if state is None or not state.ptt:
            return
        if not self._agent_output_active_for_interrupts(participant_identity=identity):
            return
        self._ensure_ducking_controller()
        if self._ducking.is_cancelled:
            return
        logger.info(
            "[StreamingPipeline] explicit client PTT interrupt received "
            "identity=%s playback=%s",
            state.participant_identity,
            state.playback_state,
        )
        if self._timeline is not None:
            self._timeline.set_attr(
                "explicit_client_interrupt",
                state.as_timeline_attr(),
            )
        # A deliberate button press — hard-cut immediately. ``force=True`` so it
        # works even in half_duplex, where the session runs with
        # allow_interruptions=False (no VAD/audio auto-interrupt) but tap-to-stop
        # must still cancel the agent's speech.
        self._duck_cancel_and_interrupt(force=True)

    def _turn_detection_for_mode(self) -> Any:
        """The Agent ``turn_detection`` for this session's mode (plan §10).

        half_duplex → ``"manual"`` (PTT owns the turn boundary; no auto EOU);
        full_duplex → the EOT model instance (VAD + semantic EOT, unchanged).
        """
        return "manual" if self._is_half_duplex else self._get_eot_model()

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

        # Bind welcome to a closure so the inner Agent class can read it
        # without us touching its constructor signature.
        welcome_message = self._welcome_message
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
                        "[lifecycle] welcome on_enter room=%s suppressed "
                        "(proactive=%s)",
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
                allowed = await pipeline._voiceprint_allows_completed_turn(
                    new_message=new_message
                )
                if not allowed:
                    raise StopResponse()

        # Round 8 R8.9 (re-fix): turn_handling config (including
        # false_interruption_timeout, preemptive_generation) lives on
        # AGENTSESSION, not Agent. Putting it here was silently ignored.
        # Agent only carries per-agent override of ``turn_detection`` (the
        # EOT model instance, which is per-agent semantic).
        # Turn detection by mode (plan §10):
        #   half_duplex (PTT) → "manual": the framework runs NO automatic
        #     end-of-utterance. The PTT button owns the turn boundary, so a
        #     mid-sentence pause can't make the framework commit early and drop
        #     the continuation ("…今天的天气 <pause> 北京的"). The whole hold
        #     accumulates into one transcript; PTT release commits it once.
        #   full_duplex → the EOT model instance (unchanged: VAD + semantic EOT).
        turn_detection = self._turn_detection_for_mode()
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
        if not hasattr(self, "_on_session_end"):
            self._on_session_end = None
        if not hasattr(self, "_idle_disconnect_grace_sec"):
            self._idle_disconnect_grace_sec = 0.3
        if not hasattr(self, "_idle_end_reason"):
            self._idle_end_reason = "idle_normal_end"
        if not hasattr(self, "_idle_keep_alive_half_duplex"):
            self._idle_keep_alive_half_duplex = True
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
                keep_alive_half_duplex=self._idle_keep_alive_half_duplex,
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
            "type": _COMPANION_UI_STATE_TOPIC,
            "state": state,
            "reason": reason,
            "ts_ms": int(time.time() * 1000),
        }

        async def _send() -> None:
            await local.publish_data(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=_COMPANION_UI_STATE_TOPIC,
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

    def _publish_client_control(self, op: str, *, reason: str) -> None:
        """Best-effort server-authoritative command for thin clients."""
        room = getattr(self, "_room", None)
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        timeline = self._timeline
        turn_id = getattr(timeline, "turn_id", "") if timeline is not None else ""
        payload = {
            "v": 1,
            "kind": "cmd",
            "id": f"{op}:{int(time.time() * 1000)}",
            "op": op,
            "payload": {
                "reason": reason,
                "turn_id": turn_id,
            },
            "ts": int(time.time() * 1000),
            "ttl_ms": 5000,
        }
        if timeline is not None:
            events = list(timeline.attrs.get("client_control_events") or ())
            events.append(
                {
                    "op": op,
                    "reason": reason,
                    "turn_id": turn_id,
                }
            )
            timeline.set_attr("client_control_events", events[-12:])

        async def _send() -> None:
            await local.publish_data(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                reliable=True,
                topic=_CLIENT_CONTROL_TOPIC,
            )

        task = loop.create_task(_send())

        def _log_failure(done: asyncio.Task[None]) -> None:
            try:
                done.result()
            except Exception:
                logger.debug(
                    "[StreamingPipeline] failed to publish client control",
                    exc_info=True,
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

    def _on_user_state_changed(self, event: Any) -> None:
        self._ensure_runtime_defaults()
        try:
            old = event.old_state
            new = event.new_state
            logger.info("[StreamingPipeline] user_state: %s -> %s", old, new)

            if new == "speaking":
                self._publish_companion_ui_state("listening", "user_state:speaking")
            elif old == "speaking" and new == "listening":
                self._publish_companion_ui_state("listening", "user_state:listening")
            elif new == "away":
                self._publish_companion_ui_state("idle", "user_state:away")

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
                self._session_signals.signal_stt_user_away()
            elif old == "away" and new in ("listening", "speaking"):
                self._session_signals.signal_stt_user_present()

            if new == "speaking":
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
                self._user_speaking_start_time = time.time()
                if not merge_continuation or self._timeline is None:
                    self._timeline = TurnTimeline(generate_turn_id())
                    self._timeline_debug_flushed = False
                    if self._room is not None:
                        self._timeline.set_attr("room_name", self._room.name or "")
                self._user_turns.start_speech(timeline=self._timeline)
                self._timeline.mark("speech_started_at")
                self._voiceprint_turns.start_turn(timeline=self._timeline)
                self._apply_pending_stt_provider_events()
                self._observe_stt_turn_audio()
                # Immediately clear stale text so EOT only sees text from THIS speech turn.
                self._latest_asr_text = ""

                # Feed VAD signal into EOT model so VADState reflects user activity.
                self._get_eot_model().update_vad(True)

                # Phase C: immediately fade agent output to silence and arm
                # the suspend-window fallback. EOT decisions in
                # the semantic interrupt handler will resolve SUSPENDED output
                # before the timeout fires in the typical case.
                self._attention_effects.handle_speaking_started()

                # If agent is speaking and interruptions are allowed, EOT check is
                # triggered synchronously in _on_user_transcribed as soon as STT
                # delivers the first transcript (INTERIM or FINAL) — no polling needed.

            elif old == "speaking" and new == "listening":
                self._user_speaking_start_time = None
                if self._timeline is not None:
                    self._timeline.mark("speech_stopped_at")
                voiceprint_task = self._voiceprint_turns.finish_turn()
                self._completed_turn_voiceprint_task = voiceprint_task
                self._completed_turn_voiceprint_result = None
                self._completed_turn_voiceprint_timeline = self._timeline

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
                    self._decision_effects.apply(
                        decision,
                        resolved_reason="user_silent",
                        transcript=self._latest_asr_text,
                        vad_active=False,
                    )

                self._callbacks.on_user_ended_speaking()
                self._skip_commit_after_interrupt_cancel = False
                if self._is_half_duplex:
                    # Manual turn_detection (plan §10): VAD silence is NOT a turn
                    # boundary in half_duplex — the PTT release is (see
                    # _handle_ptt_release_edge). A mid-utterance pause must NOT
                    # commit/defer here, or the turn would split and the
                    # continuation be dropped. Keep accumulating; do not clear
                    # _latest_asr_text.
                    logger.debug(
                        "[ptt-manual] VAD silence ignored, awaiting PTT release"
                    )
                elif self._session is not None:
                    transcript = (
                        self._user_turns.selected_text or self._latest_asr_text
                    )
                    if transcript:
                        self._remember_candidate_voiceprint_task(voiceprint_task)
                    if self._user_turns.active is None and transcript:
                        if self._timeline is None:
                            self._timeline = TurnTimeline(generate_turn_id())
                            self._timeline_debug_flushed = False
                        self._user_turns.start_speech(timeline=self._timeline)
                        self._user_turns.add_transcript(transcript, is_final=True)
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
                else:
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
        if (
            self._suppress_transcripts_until_next_speech
            and getattr(event, "transcript", "")
        ):
            logger.info(
                "[StreamingPipeline] dropping post-turn transcript after "
                "voiceprint ownership gate transcript=%r final=%s",
                event.transcript[:80],
                getattr(event, "is_final", None),
            )
            return
        # Content-based echo gate (full_duplex only): with an open mic during
        # playback, the hardware-AEC-cleaned mic can still leak residual-echo
        # spikes that STT transcribes as the agent's OWN words (energy can't
        # filter them — they exceed real speech). Drop a transcript contained in
        # what the agent is currently saying so it never starts a user turn
        # (which would churn the timeline and drop the real reply). Skipped in
        # half_duplex — the device gates the mic during playback, so there is no
        # echo to suppress.
        if (
            not self._is_half_duplex
            and self._agent_output_active_for_interrupts(
                participant_identity=getattr(event, "speaker_id", None),
            )
            and self._transcript_is_agent_echo(getattr(event, "transcript", ""))
        ):
            logger.info(
                "[StreamingPipeline] dropping agent-echo transcript during playback "
                "transcript=%r",
                getattr(event, "transcript", "")[:80],
            )
            return
        if event.transcript:
            # Real recognized speech (interim or final) — keeps the session
            # alive. Empty/noise transcripts deliberately don't, so a silent
            # room still trips the idle watchdog.
            self._mark_activity()
            self._latest_asr_text = event.transcript
            # Half-duplex 守空 guard: any recognized speech this PTT hold makes the
            # eventual release commit (an empty hold stays uncommitted).
            if self._is_half_duplex and event.transcript.strip():
                self._ptt_turn_had_speech = True
            self._ensure_user_turn_coordinator()
            self._user_turns.add_transcript(
                event.transcript,
                is_final=bool(getattr(event, "is_final", False)),
            )
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
        agent_is_speaking = self._agent_output_active_for_interrupts(
            participant_identity=getattr(event, "speaker_id", None),
        )
        interrupt_window_active = self._interrupt_window_active()
        if (
            self._allow_interruptions
            and event.transcript
            and (agent_is_speaking or interrupt_window_active)
        ):
            if self._interrupt_decision_suppressed():
                logger.debug(
                    "[StreamingPipeline] interrupt decision suppressed after cancel"
                )
                super()._on_user_transcribed(event)
                return
            if not self._attention_effects.allows_eot_check(
                event.transcript,
                speaker_id=getattr(event, "speaker_id", None),
            ):
                super()._on_user_transcribed(event)
                return
            self._semantic_interrupts.run(event.transcript, is_final=event.is_final)

        super()._on_user_transcribed(event)

    @staticmethod
    def _normalize_for_echo(text: str) -> str:
        # Keep CJK + alphanumerics (CJK is .isalnum()==True), drop punctuation /
        # spaces, lowercase — so echo matching ignores ASR punctuation noise.
        return "".join(c for c in text if c.isalnum()).lower()

    def _agent_recent_spoken_text(self) -> str:
        """The text the agent is currently synthesizing (in-flight TTS).

        Source of truth for content-based echo detection. Empty when the agent
        isn't speaking, so the echo gate is naturally scoped to playback.
        """
        factory = getattr(self, "_factory", None)
        try:
            if factory is not None and getattr(factory, "tts", None) is not None:
                return getattr(factory.tts.tts, "current_pushed_text", "") or ""
        except Exception:
            logger.debug(
                "[StreamingPipeline] could not read TTS current_pushed_text",
                exc_info=True,
            )
        return ""

    def _transcript_is_agent_echo(self, transcript: str) -> bool:
        """True when `transcript` is the agent's own current speech echoed back.

        Content-based, NOT energy-based: on this board the cleaned-mic residual
        echo spikes (RMS p99 ~1166) exceed real near-end speech (p50 ~67), so no
        energy threshold can separate them. But the echo IS the agent's words,
        which we know from the in-flight TTS text — so drop a (normalized)
        transcript that is contained in it. Caller gates on agent playback.
        """
        t = self._normalize_for_echo(transcript)
        if not t:
            return False
        agent = self._normalize_for_echo(self._agent_recent_spoken_text())
        if not agent:
            return False
        return t in agent

    def _interrupt_decision_suppressed(self) -> bool:
        """Ignore residual ASR after a confirmed interrupt cancel."""
        return time.monotonic() < self._suppress_commit_after_interrupt_until

    def _interrupt_window_active(self) -> bool:
        """Return true while an actual interrupt decision window is open."""
        return self._ducking.is_suspended or bool(
            getattr(self, "_soft_interrupt_active", False)
        )

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
        states = getattr(self, "_client_audio_states", {})
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

    def _interrupt_current_turn(self, *, force: bool = False) -> None:
        """Interrupt the currently in-progress agent turn via session.interrupt().

        ``force`` skips the ``allow_interruptions`` gate and passes through to
        ``session.interrupt(force=True)``, which cancels even a speech handle
        that was started with interruptions disabled. Used by the explicit PTT
        tap-to-stop path in half_duplex; policy-driven interrupts leave it False.
        """
        self._ensure_ducking_controller()
        if not force and not self._allow_interruptions:
            return
        # The cancelled-output short-circuit is only for the policy path (avoid a
        # redundant session.interrupt after cancel_output). The FORCED explicit
        # path (PTT tap-to-stop) MUST still call session.interrupt(force=True):
        # _duck_cancel_and_interrupt already set is_cancelled=True, but without
        # this the agent's (uninterruptible, half_duplex) speech handle is never
        # actually ended → agent_state stays "speaking" → the captured barge-in
        # turn can't commit → no reply. force must reach the framework.
        if not force and self._ducking.is_cancelled:
            logger.debug(
                "[StreamingPipeline] interrupt skipped — output already CANCELLED"
            )
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

    def _handle_hold_decision(
        self,
        decision: Decision,
        transcript: str,
        eot_score: float | None,
        vad_active: bool | None,
    ) -> None:
        if (
            decision.hold_recheck_ms is None
            and not decision.reason.startswith(STABLE_SIGNAL_WAIT_REASON_PREFIX)
        ):
            return
        if not self._ducking.is_suspended:
            return
        if not transcript.strip():
            return
        recheck_ms = decision.hold_recheck_ms
        if recheck_ms is None:
            recheck_ms = (
                self._turn_policy.interrupt.correction_topic_stability_window_ms
            )
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
    #   SemanticInterruptHandler.run (per STT interim/final):
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
            "[StreamingPipeline] duck armed  vad→duck=%.1fms  "
            "timeout=%.2fs  cooldown=%.2fs",
            vad_to_duck_ms, cfg.duck_suspend_timeout_sec, cfg.duck_cooldown_sec,
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
        client request (PTT tap-to-stop) interrupts even when the session
        disallows interruptions (half_duplex). Policy-driven callers leave it
        False so the ``allow_interruptions`` gate still applies.
        """
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
        self._cancel_stable_signal_timer()
        self._snapshot_interrupted_context()
        self._publish_client_control("playback.stop", reason="interrupt_cancel")
        self._ducking.cancel_output()
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
            time.monotonic() + _INTERRUPT_CANCEL_RESIDUAL_COMMIT_SUPPRESS_SEC
        )
        self._interrupt_current_turn(force=force)

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
            self._cancel_stable_signal_timer()
            self._ducking.unduck_if_suspended(drop_buffered=drop_buffered)
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
        context = getattr(self, "_last_interrupted_context", None)
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


def _count_cjk_chars(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _voiceprint_result_is_inconclusive(result: Any) -> bool:
    reason = str(getattr(result, "commit_reason", "") or "").lower()
    return reason in {"audio_too_short", "insufficient_audio", "too_short"}
