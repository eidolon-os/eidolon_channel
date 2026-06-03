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
from .turn_policy import (
    Action,
    Decision,
    InterruptIntent,
    TurnPolicyRuntime,
    eot_kwargs_from_turn_policy,
)
from .observability import TurnTimeline
from .output_controller import OutputController
from .factory import SharedStageFactory
from .filler import FillerManager
from .pipeline.base import BasePipeline
from .pipeline.types import PipelineCallbacks, PipelineState, generate_turn_id

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

        # DuckingMixer + suspend-window timeout fallback task. Both lazy.
        # Mixer is installed in ``run()`` after ``session.start()`` returns,
        # so the AudioOutput chain is fully assembled. The timeout task is
        # started on every VAD-start (user_state listening → speaking) and
        # cancelled the moment EOT decides to cancel/unduck.
        self._duck_mixer: OutputController | None = None
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
        _get_shared_eot_model(self._turn_policy)
        self._install_provider_observers()

    def _get_eot_model(self) -> Any:
        """Return the shared EOT model instance."""
        return _get_shared_eot_model(self._turn_policy)

    def _install_provider_observers(self) -> None:
        self._install_llm_metrics_observer()
        self._install_brain_provider_event_observer()
        self._install_stt_provider_event_observer()
        self._install_tts_provider_event_observer()

    def _install_llm_metrics_observer(self) -> None:
        """Bridge LiveKit LLM metrics into the active turn timeline."""
        if getattr(self, "_llm_metrics_observer_installed", False):
            return
        llm_plugin = getattr(getattr(self._factory, "llm", None), "llm", None)
        if llm_plugin is None or not hasattr(llm_plugin, "on"):
            return

        def _on_metrics_collected(metrics: Any) -> None:
            timeline = self._timeline
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
        self._llm_metrics_observer_installed = True

    def _install_brain_provider_event_observer(self) -> None:
        """Bridge provider-native brain RPC timing events into the timeline."""
        if getattr(self, "_brain_provider_observer_installed", False):
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
            timeline = self._timeline
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
        self._brain_provider_observer_installed = True

    def _install_tts_provider_event_observer(self) -> None:
        """Bridge provider-native TTS streaming timing events into timeline.

        These marks are provider truth (request opened, first audio byte from the
        TTS provider), distinct from ``tts_first_audio_at`` which is the
        agent-state experience mark. Together they split the brain-delta -> audio
        gap into TTS TTFB vs publish, for industry latency comparison.
        """
        if getattr(self, "_tts_provider_observer_installed", False):
            return
        tts_plugin = getattr(getattr(self._factory, "tts", None), "tts", None)
        if tts_plugin is None or not hasattr(tts_plugin, "on"):
            return

        mark_by_event = {
            "tts_request_started": "tts_request_started_at",
            "tts_provider_first_audio": "tts_provider_first_audio_at",
        }

        def _on_provider_event(event: Any) -> None:
            timeline = self._timeline
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
        self._tts_provider_observer_installed = True

    def _install_stt_provider_event_observer(self) -> None:
        """Bridge provider-native STT streaming timing events into timeline."""
        if getattr(self, "_stt_provider_observer_installed", False):
            return
        stt_plugin = getattr(getattr(self._factory, "stt", None), "stt", None)
        if stt_plugin is None or not hasattr(stt_plugin, "on"):
            return

        def _on_provider_event(event: Any) -> None:
            if not isinstance(event, dict):
                return
            if self._timeline is None:
                self._remember_pending_stt_provider_event(event)
                return
            self._record_stt_provider_event(event)

        stt_plugin.on("provider_event", _on_provider_event)
        self._stt_provider_observer_installed = True

    def _remember_pending_stt_provider_event(self, event: dict[str, Any]) -> None:
        self._ensure_runtime_defaults()
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return
        pending = self._pending_stt_provider_events
        pending.append(dict(event))
        cutoff = float(timestamp) - 2.0
        self._pending_stt_provider_events = [
            item
            for item in pending[-32:]
            if isinstance(item.get("timestamp"), (int, float))
            and float(item["timestamp"]) >= cutoff
        ]

    def _apply_pending_stt_provider_events(self) -> None:
        if self._timeline is None:
            return
        speech_started_at = self._timeline.timestamps.get("speech_started_at")
        if speech_started_at is None:
            return
        pending = list(self._pending_stt_provider_events)
        self._pending_stt_provider_events = []
        for event in pending:
            timestamp = event.get("timestamp")
            if not isinstance(timestamp, (int, float)):
                continue
            if float(timestamp) < speech_started_at - 0.5:
                continue
            self._record_stt_provider_event(event)

    def _record_stt_provider_event(self, event: dict[str, Any]) -> None:
        timeline = self._timeline
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

    def _observe_stt_turn_audio(self) -> None:
        timeline = self._timeline
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
                "[StreamingPipeline] failed to arm STT turn-audio observer"
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
        self._last_activity_monotonic = time.monotonic()

    def _start_idle_watchdog(self) -> None:
        if self._idle_timeout_sec <= 0:
            logger.info(
                "[StreamingPipeline] idle watchdog disabled "
                "(disconnect_after_idle_ms<=0)"
            )
            return
        self._mark_activity()
        self._idle_watchdog_task = asyncio.create_task(self._idle_watchdog())
        logger.info(
            "[StreamingPipeline] idle watchdog armed (timeout=%.0fs)",
            self._idle_timeout_sec,
        )

    def _stop_idle_watchdog(self) -> None:
        if self._idle_watchdog_task is not None:
            self._idle_watchdog_task.cancel()
            self._idle_watchdog_task = None

    async def _idle_watchdog(self) -> None:
        """Close the session once it has been idle past the configured timeout.

        Sleeps for the remaining time-to-deadline, then re-checks: activity
        resets ``_last_activity_monotonic`` without waking this task, so a
        wake that finds fresh activity simply sleeps again for the new
        remaining window (self-correcting, no inter-task signalling needed).
        """
        timeout = self._idle_timeout_sec
        try:
            while not self._session_closed_event.is_set():
                elapsed = time.monotonic() - self._last_activity_monotonic
                remaining = timeout - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    continue
                # Don't cut a turn that's live right now: state-change events
                # only fire on transitions, so a long continuous agent reply or
                # an in-progress user utterance wouldn't have refreshed
                # ``_last_activity_monotonic``. Treat active states as activity
                # and re-arm for another full window.
                if self._session is not None and (
                    self._session.agent_state in ("thinking", "speaking")
                    or self._session.user_state == "speaking"
                ):
                    self._mark_activity()
                    continue
                logger.info(
                    "[StreamingPipeline] idle for %.0fs ≥ %.0fs — "
                    "disconnecting session (no speech/agent activity)",
                    elapsed, timeout,
                )
                if self._timeline is not None:
                    self._timeline.mark("idle_timeout_triggered_at")
                await self._disconnect_idle()
                return
        except asyncio.CancelledError:
            pass

    async def _disconnect_idle(self) -> None:
        """Disconnect an idle session: notify the client, then destroy the room.

        Order matters:
          1. Publish a ``session_control`` data message so the client can show
             *why* it was dropped (a bare disconnect carries no reason).
          2. Brief grace so the reliable packet reaches the client before it is
             kicked.
          3. Delete the room via ``on_idle_disconnect`` — this actively
             disconnects the still-connected client (ROOM_DELETED) and resolves
             the job's shutdown future. ``session.aclose()`` alone would NOT:
             it leaves the client in a dead room and the job hanging.
          4. Set the closed event so ``run()`` exits → ``shutdown()`` closes
             STT/TTS (billing stops) even if room deletion is slow or fails.
        """
        await self._notify_client_idle_timeout()
        if self._idle_disconnect_grace_sec > 0:
            await asyncio.sleep(self._idle_disconnect_grace_sec)
        if self._on_idle_disconnect is not None:
            try:
                await self._on_idle_disconnect()
            except Exception:
                logger.exception(
                    "[StreamingPipeline] idle room-delete callback failed"
                )
        elif self._session is not None:
            # No room-delete wiring (e.g. tests / standalone) — at least close
            # the session so STT/TTS streaming and billing stop.
            try:
                await self._session.aclose()
            except Exception:
                logger.exception(
                    "[StreamingPipeline] error closing idle session"
                )
        self._session_closed_event.set()

    async def _notify_client_idle_timeout(self) -> None:
        """Best-effort: tell the client it is being dropped for inactivity.

        Sent on the ``session_control`` data topic; the web client surfaces it
        as a friendly message instead of a silent disconnect.
        """
        room = self._room
        local = getattr(room, "local_participant", None) if room else None
        if local is None:
            return
        try:
            payload = json.dumps(
                {"type": "idle_timeout", "reason": "idle_timeout"}
            ).encode("utf-8")
            await local.publish_data(
                payload, reliable=True, topic="session_control"
            )
            logger.info("[StreamingPipeline] notified client of idle timeout")
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to notify client of idle timeout",
                exc_info=True,
            )

    def _on_agent_state_changed(self, event: Any) -> None:
        """Forward agent state change + cancel pending soft interrupts.

        Round 8 cleanup: previously this also re-applied
        ``disable_audio_activity_interruption`` on every agent
        speech-start, because we only patched the runtime flag and
        framework restored it on each transition. Now ``_framework_patches``
        also patches the *default* flag, so the disable is permanent for
        the session and we no longer need this hot path.
        """
        self._ensure_runtime_defaults()
        super()._on_agent_state_changed(event)

        # Agent producing a reply (or speaking the welcome) is activity — keep
        # the idle watchdog from firing while the agent holds the turn.
        if event.new_state in ("thinking", "speaking"):
            self._mark_activity()

        # If agent starts thinking or speaking (e.g. after a new user transcript), cancel
        # any pending soft interrupt timer. The timer was set for the PREVIOUS turn's
        # interrupt; it must not fire and cancel the NEW agent response.
        if event.new_state in ("thinking", "speaking"):
            if self._filler is not None:
                self._filler.cancel()
        if self._timeline is not None:
            if event.new_state == "thinking":
                self._timeline.mark("llm_started_at")
            elif event.new_state == "speaking":
                self._timeline.mark("tts_first_audio_at")
            elif event.old_state == "speaking" and event.new_state in (
                "idle",
                "listening",
            ):
                self._timeline.mark("agent_audio_playback_done_at")
                self._append_timeline_debug("agent_audio_playback_done", clear=True)
        if event.new_state in ("thinking", "speaking") and self._soft_interrupt_active:
            logger.info(
                "[StreamingPipeline] agent started %s, cancelling pending soft interrupt timer",
                event.new_state,
            )
            self._cancel_soft_interrupt()

        # G22 (2026-05-18): on transition to SPEAKING, reset the duck mixer's
        # per-turn played-sample counter. This was previously reset inside
        # ``DuckingMixer.duck()`` — that fired multiple times per turn
        # (backchannels, echo) and clobbered the count, so the interrupted
        # context snapshot under-reported "how much the user heard". The
        # speaking-transition is the true turn boundary.
        if event.new_state == "speaking" and self._duck_mixer is not None:
            try:
                self._duck_mixer.on_agent_started_speaking()
            except AttributeError:
                # Older DuckingMixer build without the new hook — silently
                # tolerate; observed in tests that pin a frozen mixer.
                pass

        # G22-fix (2026-05-18): when a new agent turn starts (transition to
        # ``thinking``), reset the mixer state if it was left in CANCELLED
        # by a previous interrupt. Without this, OutputController.cancel()
        # leaves state="CANCELLED" forever — every subsequent TTS frame
        # gets dropped, agent_state never transitions to "speaking" (the
        # framework gates that on first audio frame reaching the inner
        # sink), and we deadlock with TTS generating audio that the user
        # never hears.
        #
        # Production manifestation (round-2 interrupt log 2026-05-18):
        # 1st interrupt cancelled TTS; 2nd user utterance produced FINAL +
        # LLM response + 17s of TTS audio, but no `agent_state → speaking`
        # and no audio playback. Symptom: "second interrupt then TTS not
        # playing".
        if event.new_state == "thinking" and self._duck_mixer is not None:
            try:
                if self._duck_mixer.state == "CANCELLED":
                    self._duck_mixer.reset()
                    logger.info(
                        "[StreamingPipeline] OutputController CANCELLED→NORMAL "
                        "(new turn starting — clearing prior-interrupt state)"
                    )
            except AttributeError:
                pass

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
            # G16 (2026-05-17): also bridge VAD signal to the STT plugin
            # if it exposes a notify_vad_state hook. Used by Bailian STT's
            # cost-saving gate (BAILIAN_STT_GATE_ENABLED=true). Duck-typed —
            # other STT plugins are unaffected.
            stt_stage = getattr(self._factory, "stt", None)
            stt_plugin = getattr(stt_stage, "stt", None) if stt_stage else None
            stt_notify = (
                getattr(stt_plugin, "notify_vad_state", None)
                if stt_plugin is not None
                else None
            )

            def _on_inference(probability: float, speaking: bool) -> None:
                # Inference thread → keep this lightweight.
                try:
                    eot_model.update_vad_probability(probability)
                except Exception:
                    # Don't let pipeline state errors stall VAD inference.
                    pass
                # G16: forward to STT gate. ``speaking`` is the boolean
                # threshold-pass result from FireRed pVAD; we synthesize an
                # RMS estimate by mapping probability to a coarse fallback
                # (the gate has its own RMS threshold for ground-truth audio,
                # but the VAD callback doesn't carry raw frame bytes — so we
                # pass 0.0 and rely on the gate's vad_high primary path).
                if stt_notify is not None:
                    try:
                        stt_notify(probability, 0.0)
                    except Exception:
                        pass

            raw_vad.register_inference_callback(_on_inference)
            logger.info(
                "[StreamingPipeline] VAD inference callback registered "
                "(per-frame probability → EOT state%s)",
                " + STT gate" if stt_notify else "",
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
                self._duck_and_arm_timeout()

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
                if (
                    self._duck_mixer is not None
                    and self._duck_mixer.state == "SUSPENDED"
                ):
                    decision = self._turn_runtime.decider.on_user_silent(
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
                    # G8 fix (2026-05-17): only commit a user turn if STT
                    # actually produced text. Without this guard, every
                    # VAD-end (including AEC-warmup-suppressed audio, brief
                    # noise, or STT hiccups) triggers commit_user_turn → the
                    # framework waits ``transcript_timeout`` for a FINAL that
                    # will never come, then promotes whatever INTERIM is
                    # currently in ``_audio_interim_transcript`` (a global
                    # string, not VAD-segment-scoped). That INTERIM is often
                    # from the NEXT user utterance, producing a ghost LLM
                    # call with cross-segment-contaminated text.
                    if self._latest_asr_text:
                        # Record turn BEFORE reset so dialogue history captures it.
                        eot_model.record_turn(
                            self._latest_asr_text,
                            is_complete=True,
                            eot_score=eot_model._current_eot_score,
                        )
                        eot_model.reset()
                        self._inject_interrupted_context()
                        if self._timeline is not None:
                            self._timeline.mark("turn_committed_at")
                        # F1 fix (2026-05-16): framework default is 2.0s, too
                        # short for Bailian FunASR FINAL on long Chinese
                        # sentences. Pass our configured timeout.
                        self._session.commit_user_turn(
                            transcript_timeout=self._stt_commit_transcript_timeout,
                        )
                        if (
                            self._filler is not None
                            and self._session.output.audio is not None
                        ):
                            self._filler.inject(self._session.output.audio)
                    else:
                        # G8 (2026-05-17): VAD-end with no STT text. Reset
                        # EOT state but skip commit — see comment above.
                        eot_model.reset()
                        logger.info(
                            "[StreamingPipeline] VAD-end with empty ASR — "
                            "skipping commit_user_turn (AEC window / noise / "
                            "STT hiccup)"
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
        self._ensure_runtime_defaults()

        eot_model = self._get_eot_model()

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
            decision = self._turn_runtime.decider.on_strong_intent()
            signal = self._turn_runtime.control_signal_from_decision(decision)
            self._publish_turn_control(signal.as_metadata())
            if self._timeline is not None:
                self._record_decision_attrs(
                    decision,
                    source="strong_intent",
                    transcript=text,
                    vad_active=True,
                )
                self._timeline.set_attr("turn_control", signal.as_metadata())
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
        # Duck-active path: delegate to InterruptDecider (G18b)
        # ──────────────────────────────────────────────────────────
        if duck_active:
            decision = self._turn_runtime.decide_from_transcript(
                text,
                score,
                vad_active=vad_active,
                agent_speaking=True,
            )
            suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
            logger.info(
                "[StreamingPipeline] EOT(duck): decision=%s reason=%s "
                "suspend_ms=%.0f score=%.2f text=%r",
                decision.action.value, decision.reason,
                suspend_ms, score, text[:80],
            )
            self._apply_decision(
                decision,
                eot_score=score,
                transcript=text,
                vad_active=vad_active,
            )
            return

        # ──────────────────────────────────────────────────────────
        # Fallback path (mixer not installed, or duck disabled):
        # use the soft/hard interrupt machinery without output ducking.
        # ──────────────────────────────────────────────────────────
        semantic_decision = self._turn_runtime.decide_from_transcript(
            text,
            score,
            vad_active=vad_active,
            agent_speaking=True,
        )
        if semantic_decision.intent in (
            InterruptIntent.HARD_STOP,
            InterruptIntent.TOPIC_SWITCH,
            InterruptIntent.CORRECTION,
            InterruptIntent.BACKCHANNEL,
            InterruptIntent.NOISE,
        ):
            logger.info(
                "[StreamingPipeline] EOT(fallback semantic): decision=%s "
                "reason=%s score=%.2f text=%r",
                semantic_decision.action.value,
                semantic_decision.reason,
                score,
                text[:80],
            )
            self._apply_decision(
                semantic_decision,
                eot_score=score,
                transcript=text,
                vad_active=vad_active,
            )
            return

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
                if self._timeline is not None:
                    self._timeline.record_decision(
                        action="cancel",
                        reason="fallback_eot_hard_score",
                        rollback_drop_buffered=False,
                        source="eot_fallback",
                        eot_score=score,
                        transcript_preview=text[:120],
                        vad_active=vad_active,
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
                if self._timeline is not None:
                    self._timeline.record_decision(
                        action="hold",
                        reason="fallback_eot_soft_interrupt",
                        rollback_drop_buffered=False,
                        source="eot_fallback",
                        eot_score=score,
                        transcript_preview=text[:120],
                        vad_active=vad_active,
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
        mixer = OutputController(
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
        self._duck_timeout_task = asyncio.create_task(
            self._duck_suspend_timeout_fallback(cfg.duck_suspend_timeout_sec)
        )

    async def _duck_suspend_timeout_fallback(self, timeout_sec: float) -> None:
        """500ms decision-budget deadline. Delegates the branch decision
        to :class:`InterruptDecider` so the policy stays in one place
        (G18b)."""
        try:
            self._ensure_runtime_defaults()
            await asyncio.sleep(timeout_sec)
            if self._duck_mixer is not None and self._duck_mixer.state == "SUSPENDED":
                suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
                buffered = self._duck_mixer.buffered_frames
                buffered_sec = self._duck_mixer.buffered_sec

                vad_still_active = (
                    self._session is not None
                    and self._session.user_state == "speaking"
                )
                latest_asr_text = self._latest_asr_text.strip()
                decision = self._turn_runtime.decider.on_decision_deadline(
                    vad_still_active,
                    has_transcript=bool(latest_asr_text),
                    transcript=latest_asr_text,
                )
                logger.info(
                    "[StreamingPipeline] duck resolved  reason=deadline  "
                    "decision=%s decider_reason=%s  has_transcript=%s  "
                    "suspend_ms=%.0f  buffered=%d frames (%.3fs)  timeout=%.2fs",
                    decision.action.value, decision.reason,
                    bool(latest_asr_text), suspend_ms, buffered, buffered_sec,
                    timeout_sec,
                )
                self._apply_decision(
                    decision,
                    resolved_reason="timeout",
                    transcript=latest_asr_text,
                    vad_active=vad_still_active,
                )
        except asyncio.CancelledError:
            pass

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
        self._record_decision_attrs(
            decision,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript=transcript,
            vad_active=vad_active,
        )
        if decision.action is Action.CANCEL:
            # Real interrupt: snapshot context, cancel mixer, interrupt TTS.
            signal = self._turn_runtime.control_signal_from_decision(decision)
            self._publish_turn_control(signal.as_metadata())
            if self._timeline is not None:
                self._timeline.set_attr("turn_control", signal.as_metadata())
            self._duck_cancel_and_interrupt()
            return
        if decision.action is Action.ROLLBACK:
            signal = self._turn_runtime.control_signal_from_decision(decision)
            self._publish_turn_control(signal.as_metadata())
            if self._timeline is not None:
                self._timeline.set_attr("turn_control", signal.as_metadata())
            self._duck_unduck_if_suspended(
                reason=resolved_reason or decision.reason,
                drop_buffered=decision.rollback_drop_buffered,
            )
            return
        # HOLD / NONE — no-op; let next interim or deadline drive.
        return

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
        if self._timeline is None:
            return
        self._timeline.record_decision(
            action=decision.action.value,
            reason=decision.reason,
            rollback_drop_buffered=decision.rollback_drop_buffered,
            intent=decision.intent.value if decision.intent is not None else None,
            intent_source=decision.intent_source,
            intent_confidence=decision.intent_confidence,
            topic_switch_hint=decision.topic_switch_hint,
            correction_hint=decision.correction_hint,
            source=source,
            resolved_reason=resolved_reason,
            eot_score=eot_score,
            transcript_preview=transcript[:120],
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
        try:
            llm_plugin = getattr(self._factory.llm, "llm", None)
            setter = getattr(llm_plugin, "set_turn_control_metadata", None)
            if setter is not None:
                setter(metadata)
        except Exception:
            logger.debug(
                "[StreamingPipeline] failed to publish turn_control metadata",
                exc_info=True,
            )

    def _cancel_duck_timeout(self) -> None:
        """Cancel the suspend-window fallback task if active. Safe to call any time."""
        if self._duck_timeout_task is not None and not self._duck_timeout_task.done():
            self._duck_timeout_task.cancel()
        self._duck_timeout_task = None

    def _duck_cancel_and_interrupt(self) -> None:
        """Confirm interrupt: discard buffer + cancel TTS generation."""
        self._ensure_runtime_defaults()
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
        if self._duck_mixer is None:
            return
        self._cancel_duck_timeout()
        if self._duck_mixer.state == "SUSPENDED":
            suspend_ms = (time.monotonic() - self._duck_suspend_start) * 1000
            buffered = self._duck_mixer.buffered_frames
            buffered_sec = self._duck_mixer.buffered_sec
            logger.info(
                "[StreamingPipeline] duck resolved  reason=%s  "
                "action=unduck(drop_buffered=%s)  suspend_ms=%.0f  "
                "buffered=%d frames (%.3fs)",
                reason, drop_buffered,
                suspend_ms, buffered, buffered_sec,
            )
            self._duck_mixer.unduck(drop_buffered=drop_buffered)
            self._last_unduck_time = time.monotonic()
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
        cfg = self._get_eot_model()._config
        if not cfg.interrupted_context_enabled:
            return
        if self._session is None:
            return
        try:
            played_sec = (
                self._duck_mixer.played_seconds
                if self._duck_mixer is not None
                else None
            )

            # G21: primary path — read in-flight TTS text directly from
            # the plugin instance. Both our TTS plugins (Bailian, SenseTime)
            # expose ``current_pushed_text``; plugins that don't (e.g.
            # third-party) fall through to the history-based fallback.
            in_flight_text = ""
            tts_plugin = None
            try:
                if self._factory is not None and self._factory.tts is not None:
                    tts_plugin = self._factory.tts.tts
                    in_flight_text = getattr(
                        tts_plugin, "current_pushed_text", ""
                    ) or ""
            except Exception:
                logger.debug(
                    "[StreamingPipeline] could not read TTS current_pushed_text",
                    exc_info=True,
                )

            if in_flight_text and in_flight_text.strip():
                self._last_interrupted_context = {
                    "text": in_flight_text,
                    "timestamp": time.monotonic(),
                    "played_seconds": played_sec,
                    "source": "tts_in_flight",
                }
                logger.info(
                    "[StreamingPipeline] interrupted context captured "
                    "(source=tts_in_flight): text=%r played=%.2fs",
                    in_flight_text[:80],
                    played_sec or 0.0,
                )
                return

            # Fallback: walk session.history for the most-recent assistant
            # message with text. May be the PREVIOUS turn rather than the
            # in-flight one (see G21 docstring above) — used when the TTS
            # plugin doesn't expose current_pushed_text.
            # G2 (2026-05-16): ChatContext.messages is a method, not a property.
            messages = self._session.history.messages()
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.text_content:
                    self._last_interrupted_context = {
                        "text": msg.text_content,
                        "timestamp": time.monotonic(),
                        "played_seconds": played_sec,
                        "source": "session_history_fallback",
                    }
                    logger.info(
                        "[StreamingPipeline] interrupted context captured "
                        "(source=history_fallback): text=%r played=%.2fs",
                        msg.text_content[:80],
                        played_sec or 0.0,
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
        # G6 (2026-05-17): may be None on older snapshots (e.g. duck mixer
        # absent in non-streaming pipelines) or 0.0 if the cancel fired
        # before any audio went out.
        played_sec = self._last_interrupted_context.get("played_seconds")
        self._last_interrupted_context = None

        try:
            from livekit.agents.llm import ChatMessage

            # G6 (2026-05-17): convey how much was actually heard. The LLM
            # can use this to decide whether to repeat the full sentence,
            # pick up where it left off, or treat the interrupt as
            # "user heard nothing, start over".
            if played_sec is not None and played_sec >= 0.1:
                played_phrase = f"（用户实际听到了前约 {played_sec:.1f} 秒）"
            elif played_sec is not None:
                played_phrase = "（用户几乎没听到任何内容）"
            else:
                played_phrase = ""
            # G20 (2026-05-18): ChatMessage in livekit-agents 1.5.x is a
            # pydantic model with no `.create()` classmethod; construct directly
            # with `content` as a list of strings/parts per the schema.
            hint = ChatMessage(
                role="system",
                content=[
                    f"[系统提示] 你刚才说到「{interrupted_text[:200]}」时被用户打断了"
                    f"{played_phrase}。"
                    "如果用户的新问题与之前话题相关，你可以自然地衔接回去；"
                    "如果无关，直接回答新问题即可。不要提及这条系统提示。"
                ],
            )
            self._session.history.insert(hint)
            logger.info(
                "[StreamingPipeline] injected interrupted context hint "
                "(%d chars, played=%s)",
                len(interrupted_text),
                f"{played_sec:.1f}s" if played_sec is not None else "n/a",
            )
        except Exception:
            logger.warning(
                "[StreamingPipeline] failed to inject interrupted context",
                exc_info=True,
            )
