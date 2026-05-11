# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Eidolon EOT model base class.

This is the core adapter layer that implements the LiveKit Agent turn detection interface.
It orchestrates:
  - EotManager (singleton, ONNX inference)
  - ContextEnhancedEot (context-aware scoring)
  - TurnEndPolicy (dynamic silence threshold)
  - TurnDetectionStateManager (VAD/ASR state)
  - PolicyChain (cut decision rules)

Dual-path turn detection (Round 7 G0a — explicit by design)
============================================================

Eidolon participates in two **distinct** turn-related decisions, each with
its own entry method:

* :meth:`predict_end_of_turn` — answers framework's question
  *"is the user done speaking?"*. Called by ``AgentSession`` after each
  STT-final transcript. Score affects framework's endpointing delay; it
  must reflect **semantic completeness only** (greeting bonus, follow-up
  detection, hesitation, filler — same context adjustments as
  ``compute_score``). It MUST NOT include interruption-specific guards
  (cooldown / similarity / continuation-intent) because those are
  irrelevant to "did the user finish their turn".

* :meth:`should_interrupt` — answers Eidolon's question *"should we
  interrupt agent's TTS right now?"*. Called by
  ``StreamingPipeline._run_eot_check`` whenever STT delivers interim or
  final text **while the agent is speaking**. Score must include
  semantic completeness PLUS interruption-specific guards (cooldown to
  avoid bouncing interrupts, similarity to avoid duplicate cuts,
  continuation-intent to avoid cutting on "再换一首"). Strictly stricter
  than :meth:`predict_end_of_turn` — interrupting agent is more
  disruptive than waiting for user to finish.

Both paths share the same underlying ONNX call via
:meth:`_raw_eot_score` (the ``EotManager`` 50ms TTL cache guarantees a
single physical inference per text). The difference is in which
adjustments / guards each path layers on top.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from typing import TYPE_CHECKING, Optional

from ..config import EidolonEOTConfig
from ..impl.eot_manager import EotManager
from ..impl.context_enhanced_eot import ContextEnhancedEot
from ..impl.conversation_phase import (
    ConversationPhase,
    ConversationPhaseDetector,
)
from ..impl.turn_end_policy import TurnEndPolicy
from ..impl.state import TurnDetectionStateManager
from ..impl.eot_policy import PolicyChain
from ..log import logger

_DEBUG_LOG_PATH = os.environ.get("EIDOLON_EOT_DEBUG_LOG", "")


def _eot_log(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    """Append a NDJSON log entry for EOT model debugging. No-op unless EIDOLON_EOT_DEBUG_LOG is set."""
    if not _DEBUG_LOG_PATH:
        return
    entry = {
        "id": f"log_{int(time.time() * 1000)}",
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "hypothesisId": hypothesis_id,
    }
    if data:
        entry["data"] = data
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

if TYPE_CHECKING:
    from livekit.agents import llm


class EidolonEOTModel(ABC):
    """
    Eidolon EOT detection model base class.

    This class implements the LiveKit Agent turn detection contract.
    Subclasses configure the model type (Chinese vs Multilingual).

    Usage::

        from eidolon.channel.livekit.plugins.eot import ChineseModel

        session = AgentSession(turn_detection=ChineseModel())

        # Or directly use the model:
        model = ChineseModel()
        score = await model.predict_eou(chat_ctx)
        threshold = model.get_dynamic_silence_threshold(text, score, is_final=False)
    """

    def __init__(self, config: EidolonEOTConfig):
        self._config = config

        # EOT singleton manager (config passed so the singleton backend respects prefer_multilingual)
        self._eot_manager = EotManager(config)

        # Context-enhanced EOT scorer
        self._context_eot = ContextEnhancedEot(
            base_eot=self._eot_manager,
            max_history=config.utterance_end_max_history,
            enable_user_profile=config.enable_user_profile,
            enable_temporary_compensations=config.enable_temporary_compensations,
            cooldown_period=config.context_eot_cooldown_sec,
        )

        # Dynamic threshold policy
        self._turn_end_policy = TurnEndPolicy(
            t_min=config.min_threshold,
            t_max=config.max_threshold,
            t_urgent=config.urgent_threshold,
            enable_semantic_tail_hang=config.enable_semantic_tail_hang,
            t_fast=config.t_fast,
            t_mid=config.t_mid,
            t_deep=config.t_deep,
            t_tail_hang=config.tail_hang_silence_sec,
            is_final_reduction=config.is_final_threshold_reduction,
        )

        # State manager
        self._state = TurnDetectionStateManager(
            max_sentence_duration=config.max_sentence_second,
            vad_stale_timeout=config.vad_stale_timeout_sec,
            min_cut_interval=config.min_cut_interval,
        )

        # Policy chains
        self._policy_chain = PolicyChain.for_normal_turn_end(
            final_cut_min_silence_sec=config.final_cut_min_silence_sec,
            stability_short_sec=config.asr_stability_short_sec,
            stability_long_sec=config.asr_stability_long_sec,
            stability_long_char_threshold=config.asr_stability_long_char_threshold,
            vad_flip_window_sec=config.vad_flip_window_sec,
            vad_flip_count_threshold=config.vad_flip_count_threshold,
            min_speech_duration_sec=config.min_speech_duration_sec,
            similarity_threshold=config.similarity_threshold,
            min_avg_vad_confidence=config.min_avg_vad_confidence,
            confidence_window_sec=config.vad_confidence_window_sec,
        )
        # Dedicated chain for semantic interruption (agent speaking, user starts speaking).
        self._semantic_policy_chain = PolicyChain.for_semantic_interruption(
            base_threshold=config.streaming_eot_base_threshold,
            weak_threshold=config.streaming_eot_weak_threshold,
            vad_active_delta=config.semantic_threshold_vad_active_delta,
            vad_flip_window_sec=config.vad_flip_window_sec,
            vad_flip_count_threshold=config.vad_flip_count_threshold,
            min_speech_duration_sec=config.min_speech_duration_sec,
            similarity_threshold=config.similarity_threshold,
            min_avg_vad_confidence=config.min_avg_vad_confidence,
            confidence_window_sec=config.vad_confidence_window_sec,
        )

        # EOT score cache for the current streaming text
        self._current_eot_score: float = 0.0
        self._last_asr_update_time: float = 0.0

        # Round 7 G5: conversation-phase detector (stateless). Called on
        # update_asr / record_turn to refresh state.conversation_phase so
        # TurnEndPolicy can scale silence thresholds by phase.
        self._phase_detector: ConversationPhaseDetector = ConversationPhaseDetector()

        # NOTE: Round 8 R8.13 turn-merge dampener was removed when STT
        # was moved back to Bailian streaming. The dampener existed to
        # work around SenseAudio's server-VAD splitting one utterance
        # into multiple ``result_final`` segments — Bailian streaming
        # gives clean incremental interim → one final per utterance, so
        # there is nothing to merge. Keeping the dampener active was
        # actively harmful: every clean final got dampened to ~0.55,
        # forcing framework to wait its full ``max_endpointing_delay``
        # (3 s) before committing the turn, adding 2-3 s of latency to
        # every reply. See git log around 2026-05-08 for context.

    # ------------------------------------------------------------------
    # _TurnDetector Protocol metadata (livekit-agents 1.5.6+)
    #
    # Framework reads these in ``agent_activity._user_turn_completed_task``
    # (~line 2069) to construct EOUMetrics. Missing them raises
    # ``AttributeError`` in framework's metrics path on every turn end.
    # See ``livekit/agents/voice/turn.py:_TurnDetector``.
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """ONNX model identifier (chinese / multilingual)."""
        return "multilingual" if self._config.prefer_multilingual else "chinese"

    @property
    def provider(self) -> str:
        """Plugin/provider name reported in EOUMetrics."""
        return "eidolon"

    @property
    def soft_interrupt_timeout(self) -> float:
        """Soft interrupt timeout in seconds, read from config."""
        return self._config.soft_interrupt_timeout_sec

    @property
    def soft_interrupt_score_threshold(self) -> float:
        """Tiered interrupt: score floor that triggers a soft interrupt."""
        return self._config.soft_interrupt_score_threshold

    @property
    def hard_interrupt_score_threshold(self) -> float:
        """Tiered interrupt: score floor that bypasses soft → fires hard."""
        return self._config.hard_interrupt_score_threshold

    async def predict_eou(self, chat_ctx: "llm.ChatContext") -> float:
        """
        LiveKit Agent entry point: given a ChatContext, return EOT probability [0,1].

        Extracts the last user text from chat_ctx and computes the context-enhanced
        EOT score.

        Args:
            chat_ctx: LiveKit ChatContext containing the conversation history.

        Returns:
            EOT probability in [0, 1].
        """
        text = self._extract_last_user_text(chat_ctx)
        if not text:
            return 0.0

        return self._context_eot.p_complete_score(text)

    # -------------------------------------------------------------------------
    # LiveKit _TurnDetector protocol methods
    # -------------------------------------------------------------------------

    async def unlikely_threshold(self, language: "str | None") -> "float | None":
        """
        Implement LiveKit _TurnDetector protocol.

        Returns the unlikely threshold — scores above this mean the user is unlikely
        to continue, i.e., the turn is probably complete.
        """
        return self._config.eot_unlikely_threshold

    async def supports_language(self, language: "str | None") -> bool:
        """EOT model supports all languages (token-level scoring)."""
        return True

    async def predict_end_of_turn(
        self, chat_ctx: "llm.ChatContext", *, timeout: "float | None" = None
    ) -> float:
        """Path A — framework's question: "is the user done speaking?".

        Implements the LiveKit ``_TurnDetector`` protocol. Called by
        ``AgentSession`` after each STT-final transcript. The score affects
        framework's endpointing delay and ultimately when ``commit_user_turn``
        fires (and therefore when the LLM is invoked).

        **Score composition** (semantic completeness only):

        * Raw ONNX score from FireRed Chat Turn Detector
        * Context adjustments via ``ContextEnhancedEot._get_context_adjustment``
          (greeting bonus, follow-up, hesitation, filler, user profile, etc.)

        **Explicitly NOT included** — these belong to :meth:`should_interrupt`:

        * Cooldown after a recent interrupt
        * Similarity to last-interrupt text
        * Continuation-intent guards ("再换一首")

        Returns a score in [0, 1]; framework treats ``score >= unlikely_threshold``
        as "user done".
        """
        text = self._extract_last_user_text(chat_ctx)
        if not text:
            logger.debug("[EOT] predict_end_of_turn: no user text, returning 0.0")
            return 0.0

        # Use the shared ``semantic_completeness_score`` so this path agrees
        # with ``compute_score``'s gates on continuation-intent and too-short
        # filler — only cooldown/similarity (interrupt-specific) live in
        # compute_score. Without this, backchannels like "嗯" scored ~0.7
        # here while compute_score returned 0.0 → framework committed the
        # turn anyway and fired LLM. Production bug 2026-05-07.
        score = self._context_eot.semantic_completeness_score(text)

        _eot_log("H2/H5", "base.py:EidolonEOTModel.predict_end_of_turn",
                 f"EOT CALLED! text={text[:80]!r} score={score:.3f}",
                 {"text": text, "score": score})
        logger.info(
            "[EOT] predict_end_of_turn text=%r score=%.3f",
            text[:80], score,
        )
        return score

    def _extract_last_user_text(self, chat_ctx: "llm.ChatContext") -> str:
        """
        Extract the last user text from ChatContext.

        Round 8 R8.13 cleanup: previously this only handled content
        blocks with ``.text`` attributes; missed the simpler case where
        ``content`` is a list of plain strings (which is what
        ``ChatContext.add_message(role="user", content="...")`` creates).
        Now uses ``text_content`` (the framework's own helper that
        flattens text-like content into a string) as the primary path.

        Args:
            chat_ctx: LiveKit ChatContext.

        Returns:
            Last user message text, or empty string if none.
        """
        try:
            if not (hasattr(chat_ctx, "items") and chat_ctx.items):
                return ""
            for item in reversed(chat_ctx.items):
                if not (hasattr(item, "role") and str(item.role) == "user"):
                    continue
                # Primary path: text_content is the framework's official
                # API for "give me this message's text in string form,
                # whatever content blocks it has".
                tc = getattr(item, "text_content", None)
                if tc:
                    return tc
                # Fallback path: walk content blocks manually for
                # exotic ChatItem shapes.
                if hasattr(item, "content") and item.content:
                    for block in item.content:
                        if isinstance(block, str):
                            if block:
                                return block
                        elif hasattr(block, "text") and block.text:
                            return block.text
                # Last resort: legacy item.text attribute
                if hasattr(item, "text") and item.text:
                    return item.text
                return ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract last user text from chat_ctx: {e}")
            return ""

    def update_vad(self, is_active: bool) -> None:
        """
        Update VAD state for streaming use.

        Delegates to :meth:`TurnDetectionStateManager.update_vad` so that
        ``active_since`` / ``transition_times`` / ``has_active_in_segment``
        are all maintained properly. Earlier this assigned to the
        ``vad_active`` property setter, which only flipped ``_vad.active``
        and silently dropped the timing fields — that left
        ``active_since=None`` forever, so :class:`MinSpeakingDurationPolicy`
        always read ``get_speech_duration()`` as ``0.0`` and reported
        "Speech too short 0.000s" for every interim cut. Was masked by
        R8.13 dampener (now removed); rediscovered when streaming-STT
        revealed every interim cut being suppressed.

        Args:
            is_active: Whether voice activity is currently detected.
        """
        self._state.update_vad(is_active)

    def update_vad_probability(self, probability: float) -> None:
        """Round 7 G6: feed per-frame VAD probability into state.

        Wired by ``StreamingPipeline`` to the FireRed VAD's per-frame
        inference callback. Policies can then read
        ``state.recent_avg_vad_confidence(window)`` to discriminate real
        speech from echo/noise — useful for confidence-gated cut decisions.

        Args:
            probability: VAD inference probability ∈ [0, 1] for this frame.
        """
        self._state.update_vad_probability(probability)

    # Lowered 4 → 3 when streaming STT (Bailian) became the primary
    # backend: incremental interims like "你给我" (3 chars, score 0.55)
    # carry useful information once the model has at least 3 tokens.
    # 200 ms debounce protects against ONNX call-rate inflation when
    # Bailian fires interim every ~100 ms.
    _MIN_INFERENCE_CHARS: int = 3
    _MIN_INFERENCE_INTERVAL: float = 0.2

    def update_asr(self, text: str, is_final: bool) -> float:
        """
        Update ASR text and recompute EOT score.

        Skips ONNX inference for very short interim transcripts and applies a
        200ms time debounce to avoid excessive inference during rapid ASR updates.
        Final transcripts always trigger inference regardless of debounce.

        Args:
            text: Current ASR text.
            is_final: Whether ASR has returned the final result for this segment.

        Returns:
            Current EOT score [0, 1].
        """
        self._state.update_asr(text, is_final)
        now = time.time()

        should_infer = (
            len(text.strip()) >= self._MIN_INFERENCE_CHARS
            and (is_final or (now - self._last_asr_update_time) >= self._MIN_INFERENCE_INTERVAL)
        )
        self._last_asr_update_time = now

        if should_infer:
            self._current_eot_score = self._context_eot.p_complete_score(text)
            self._state.update_eot_score(self._current_eot_score)
        elif len(text.strip()) < self._MIN_INFERENCE_CHARS:
            self._current_eot_score = 0.0

        # Round 7 G5: refresh conversation phase (cheap stateless detection).
        # Done on every update so transitions like "...好" → "...好的就这样吧"
        # (CLOSING) are caught as soon as the ASR delivers the trigger word.
        history_length = len(self._context_eot._dialogue_history)
        phase = self._phase_detector.detect(text, history_length)
        self._state.update_conversation_phase(phase)

        logger.info("[EOT] update_asr text=%r is_final=%s score=%.3f phase=%s",
                    text[:80], is_final, self._current_eot_score, phase.name)
        return self._current_eot_score

    def should_interrupt(self, text: str, vad_active: bool, is_final: bool = False) -> bool:
        """Path B — Eidolon's question: "should we interrupt agent now?".

        Called by :meth:`StreamingPipeline._run_eot_check` whenever STT
        delivers interim/final text **while the agent is speaking**.
        Stricter than :meth:`predict_end_of_turn` — must include interruption-
        specific guards because cutting agent's TTS is more disruptive than
        waiting for user to finish their turn.

        **Score composition** (via ``ContextEnhancedEot.compute_score``):

        * Continuation-intent guard ("再换一首" → score=0)
        * Valid-speech guard (too short / pure filler → score=0)
        * Cooldown guard (recent interrupt → score=0)
        * Similarity guard (duplicate of last cut text → score=0)
        * Raw ONNX score + context adjustments (same as Path A)

        **Then delegates to** the semantic-interruption ``PolicyChain``
        (cooldown, intent, MinSpeakingDuration, VADStability,
        DuplicateText, MaxDuration, VADStale, EOTScoreSemantic) for the
        final cut/block decision.

        Strong interrupt intent ("停", "闭嘴", etc.) is handled by
        ``InterruptIntentPolicy`` in the chain. A separate fast-path check
        in ``streaming.py`` covers the immediate hard-interrupt case
        (bypasses the two-stage soft-interrupt flow entirely).

        Args:
            text: Current streaming ASR text.
            vad_active: Whether VAD is currently active.
            is_final: Whether ASR has confirmed this as the final transcript
                for the current utterance.

        Returns:
            True if playback should be interrupted.
        """
        # Update ASR state so all policies have consistent, up-to-date data.
        # VAD state is already updated by the caller (streaming.py
        # _on_user_state_changed) via update_vad() — do NOT call update_vad
        # again here to avoid inflating transition counts in VADStabilityPolicy.
        self._state.update_asr(text, is_final=is_final)

        # Compute EOT score with interruption-specific guards layered on.
        # Cache the score on the model so callers (StreamingPipeline) can
        # branch on it (tiered hard vs. soft interrupt) without redoing
        # the work. Read via :pyattr:`current_eot_score`.
        if text.strip():
            eot_score = self._context_eot.compute_score(text, is_final=False)
            self._state.update_eot_score(eot_score)
            self._current_eot_score = eot_score

        # Delegate entirely to the semantic interruption policy chain.
        decision = self._semantic_policy_chain.check(self._state, self._turn_end_policy)

        if decision is not None and decision.should_cut:
            # Record cut so subsequent calls see cooldown / similarity guards.
            # Use the public record_interrupt() API instead of writing private
            # fields directly (G0a in Round 7).
            self._context_eot.record_interrupt(text)
            logger.info(
                "[EOT] should_interrupt: cut=%s — %s",
                decision.should_cut,
                decision.reason,
            )
            return True

        if decision is not None:
            logger.info(
                "[EOT] should_interrupt: cut=%s — %s",
                decision.should_cut,
                decision.reason,
            )
        return False

    def get_dynamic_silence_threshold(
        self, text: str, p_complete: float, is_final: bool
    ) -> float:
        """
        Called by AgentSession during silence: return how many seconds to wait before breaking.

        Args:
            text: Current ASR text.
            p_complete: Current EOT completeness probability.
            is_final: Whether ASR has returned final result.

        Returns:
            Silence threshold in seconds.
        """
        return self._turn_end_policy.get_dynamic_threshold(
            text=text,
            p_complete=p_complete,
            is_final=is_final,
            # Round 7 G5: scale threshold by current conversation phase.
            phase=self._state.conversation_phase,
        )

    def start_session(self, session_id: str) -> None:
        """Initialize session context for multi-turn awareness.

        Must be called once when the pipeline starts (e.g. after session.start()).
        Enables user profile tracking and dialogue history.
        """
        self._context_eot.start_session(session_id)
        logger.info("[EOT] session started: %s", session_id)

    def end_session(self, session_id: "str | None" = None) -> None:
        """Round 8 P2.L8: clean up session-scoped state on room close.

        Removes the session's ``UserProfile`` and clears the active
        session-id pointer. Should be called from
        ``StreamingPipeline._on_session_close`` so long-running
        daemons don't accumulate per-session state forever.

        ``session_id=None`` clears the currently-active session.
        """
        self._context_eot.end_session(session_id)
        logger.info(
            "[EOT] session ended: %s",
            session_id or "(current)",
        )

    def record_turn(self, text: str, is_complete: bool, eot_score: float) -> None:
        """Record a completed user turn into dialogue history.

        Must be called at turn boundaries (user stops speaking) BEFORE reset().
        Enables multi-turn features: follow-up detection, hesitation patterns,
        first-turn greeting detection.
        """
        if text and text.strip():
            self._context_eot.record_turn(text, is_complete, eot_score)
            logger.debug("[EOT] recorded turn: text=%r score=%.3f", text[:40], eot_score)

    def reset(self) -> None:
        """Reset per-turn state (ASR, scores). Preserves VAD activity
        AND dialogue history.

        Crucially this does **not** rebuild :class:`VADState` (as the
        previous ``reset_session("", 0)`` did) — in streaming-STT mode
        the next utterance can begin within ~200 ms of the previous one,
        and wiping ``vad.active_since`` causes
        :class:`MinSpeakingDurationPolicy` to read ``Speech too short
        0.000s`` for every interim cut on that next utterance,
        suppressing all early interrupts. VAD state is managed
        end-to-end by ``streaming.py``'s ``update_vad(True/False)`` calls
        on framework user_state transitions.
        """
        self._context_eot.reset_turn()
        self._state.reset_turn()
        self._current_eot_score = 0.0
        self._last_asr_update_time = 0.0
        # Round 7 G5: clear conversation phase between turns. Will be
        # re-detected on the next update_asr.
        self._state.update_conversation_phase(None)

    @property
    def current_eot_score(self) -> float:
        """Most recent EOT score from ``update_asr`` / ``should_interrupt``.

        Used by :class:`StreamingPipeline._run_eot_check` to branch
        between hard and soft interrupt tiers based on confidence.
        """
        return self._current_eot_score
