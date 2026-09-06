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
  - ContextEnhancedEot (learned scoring and interruption state guards)
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
  must reflect **semantic completeness only**, using the learned model's
  score for the current user text. It MUST NOT include interruption-specific
  guards (cooldown / similarity) because those are
  irrelevant to "did the user finish their turn".

* :meth:`should_interrupt` — answers Eidolon's question *"should we
  interrupt agent's TTS right now?"*. Called by
  ``SemanticInterruptHandler`` whenever STT delivers interim or
  final text **while the agent is speaking**. Score must include
  semantic completeness PLUS interruption-specific guards (cooldown to
  avoid bouncing interrupts, similarity to avoid duplicate cuts). Strictly stricter
  than :meth:`predict_end_of_turn` — interrupting agent is more
  disruptive than waiting for user to finish.

Both paths share the same underlying ONNX call via
:meth:`_raw_eot_score` (the ``EotManager`` 50ms TTL cache reuses the
last text's score after inference completes). The difference is in which
adjustments / guards each path layers on top.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import EidolonEOTConfig
from ..impl.eot_manager import EotManager
from ..impl.context_enhanced_eot import ContextEnhancedEot
from ..impl.turn_end_policy import TurnEndPolicy
from ..impl.state import TurnDetectionStateManager
from ..impl.eot_policy import PolicyChain
from ..log import logger


def _resolve_debug_log_path() -> str:
    raw = os.environ.get("EIDOLON_EOT_DEBUG_LOG", "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    log_dir = os.environ.get("LOG_DIR", "").strip()
    if log_dir:
        return str(Path(log_dir).expanduser() / path)
    log_root = os.environ.get("EIDOLON_LOG_ROOT", "").strip()
    base = Path(log_root).expanduser() if log_root else Path.home() / "eidolon" / "logs"
    return str(base / "channel" / path)


_DEBUG_LOG_PATH = _resolve_debug_log_path()


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

        from eidolon.livekit.plugins.eot import ChineseModel

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
            cooldown_period=config.context_eot_cooldown_sec,
            similarity_threshold=config.similarity_threshold,
        )

        # Dynamic threshold policy
        self._turn_end_policy = TurnEndPolicy(
            t_max=config.max_threshold,
            t_urgent=config.urgent_threshold,
            t_fast=config.t_fast,
            t_mid=config.t_mid,
            t_deep=config.t_deep,
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
        )
        # Dedicated chain for semantic interruption (agent speaking, user starts speaking).
        self._semantic_policy_chain = PolicyChain.for_semantic_interruption(
            base_threshold=config.streaming_eot_base_threshold,
            vad_active_delta=config.semantic_threshold_vad_active_delta,
            vad_flip_window_sec=config.vad_flip_window_sec,
            vad_flip_count_threshold=config.vad_flip_count_threshold,
            min_speech_duration_sec=config.min_speech_duration_sec,
            similarity_threshold=config.similarity_threshold,
        )

        # EOT score cache for the current streaming text
        self._current_eot_score: float = 0.0
        self._current_eot_text = ""
        self._last_asr_inference_time: float | None = None

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

        Extracts the last user text from chat_ctx and computes the learned
        EOT score. Conversation history is not part of this model's input.

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

        **Score composition**: the learned FireRed Chat Turn Detector score.
        Fixed transcript phrases do not modify the production score.

        **Explicitly NOT included** — these belong to :meth:`should_interrupt`:

        * Cooldown after a recent interrupt
        * Similarity to last-interrupt text
        * Interruption cooldown and duplicate-transcript suppression

        Returns a score in [0, 1]; framework treats ``score >= unlikely_threshold``
        as "user done".
        """
        text = self._extract_last_user_text(chat_ctx)
        if not text:
            logger.debug("[EOT] predict_end_of_turn: no user text, returning 0.0")
            return 0.0

        # Use the same learned scorer as the interruption path; only
        # cooldown/similarity (interrupt-specific) live in compute_score.
        score = self._context_eot.semantic_completeness_score(text)

        _eot_log(
            "H2/H5",
            "base.py:EidolonEOTModel.predict_end_of_turn",
            f"EOT CALLED! text={text[:80]!r} score={score:.3f}",
            {"text": text, "score": score},
        )
        logger.info(
            "[EOT] predict_end_of_turn text=%r score=%.3f",
            text[:80],
            score,
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
        200ms inference throttle to avoid excessive inference during rapid ASR updates.
        Nonempty final transcripts are scored regardless of length or cadence.

        Args:
            text: Current ASR text.
            is_final: Whether ASR has returned the final result for this segment.

        Returns:
            Current EOT score [0, 1].
        """
        self._state.update_asr(text, is_final)
        now = time.monotonic()
        stripped = text.strip()
        # Throttle from the last inference, not the last ASR event. A stream
        # of 90ms interims must continue making progress without going quiet.
        due = (
            self._last_asr_inference_time is None
            or now - self._last_asr_inference_time >= self._MIN_INFERENCE_INTERVAL
        )
        should_infer = bool(stripped) and (
            is_final or (len(stripped) >= self._MIN_INFERENCE_CHARS and due)
        )
        if should_infer:
            self._current_eot_score = self._context_eot.p_complete_score(text)
            self._current_eot_text = stripped
            self._last_asr_inference_time = now
            self._state.update_eot_score(self._current_eot_score)
        elif len(stripped) < self._MIN_INFERENCE_CHARS or stripped != self._current_eot_text:
            # A throttled revision has no score yet. The preceding hypothesis'
            # score must not authorize an interruption of this different text.
            self._current_eot_score = 0.0
            self._current_eot_text = ""
            self._state.update_eot_score(0.0)

        logger.info(
            "[EOT] update_asr text=%r is_final=%s score=%.3f",
            text[:80],
            is_final,
            self._current_eot_score,
        )
        return self._current_eot_score

    def should_interrupt(self, text: str, vad_active: bool, is_final: bool = False) -> bool:
        """Path B — Eidolon's question: "should we interrupt agent now?".

        Called by ``SemanticInterruptHandler`` whenever STT
        delivers interim/final text **while the agent is speaking**.
        Stricter than :meth:`predict_end_of_turn` — must include interruption-
        specific guards because cutting agent's TTS is more disruptive than
        waiting for user to finish their turn.

        **Score composition** (via ``ContextEnhancedEot.compute_score``):

        * Cooldown guard (recent interrupt → score=0)
        * Similarity guard (duplicate of last cut text → score=0)
        * EOT model score and provider-neutral acoustic/finality state

        **Then delegates to** the semantic-interruption ``PolicyChain``
        (cooldown, MinSpeakingDuration, VADStability,
        DuplicateText, MaxDuration, VADStale, EOTScoreSemantic) for the
        final cut/block decision.

        No fixed transcript phrase receives control authority. Immediate
        hard interruption is reserved for explicit client control such as PTT;
        spoken interruption uses EOT/VAD/finality evidence.

        Args:
            text: Current streaming ASR text.
            vad_active: Whether VAD is currently active.
            is_final: Whether ASR has confirmed this as the final transcript
                for the current utterance.

        Returns:
            True if playback should be interrupted.
        """
        # Update ASR state so all policies have consistent, up-to-date data.
        # VAD state is already updated by the caller (full-duplex pipeline
        # _on_user_state_changed) via update_vad() — do NOT call update_vad
        # again here to avoid inflating transition counts in VADStabilityPolicy.
        self._state.update_asr(text, is_final=is_final)

        # Compute EOT score with interruption-specific guards layered on.
        # Cache the score on the model so callers (StreamingPipeline) can
        # branch on it (tiered hard vs. soft interrupt) without redoing
        # the work. Read via :pyattr:`current_eot_score`.
        if text.strip():
            eot_score = self._context_eot.compute_score(text)
            self._state.update_eot_score(eot_score)
            self._current_eot_score = eot_score
            self._current_eot_text = text.strip()

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

    def get_dynamic_silence_threshold(self, text: str, p_complete: float, is_final: bool) -> float:
        """
        Return the product silence policy's threshold.

        The Agent adapter binds its fast/deep bounds through SDK endpointing
        options. AgentSession does not call this helper for every transcript.

        Args:
            text: Current ASR text.
            p_complete: Current EOT completeness probability.
            is_final: Whether ASR has returned final result.

        Returns:
            Silence threshold in seconds.
        """
        del text
        return self._turn_end_policy.get_dynamic_threshold(p_complete, is_final)

    def start_session(self, session_id: str) -> None:
        """Initialize session context for multi-turn awareness.

        Must be called once when the pipeline starts (e.g. after session.start()).
        Starts bounded diagnostic history for the room.
        """
        self._context_eot.start_session(session_id)
        logger.info("[EOT] session started: %s", session_id)

    def end_session(self, session_id: "str | None" = None) -> None:
        """Round 8 P2.L8: clean up session-scoped state on room close.

        Clears the active session and bounded diagnostic history. Called from
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
        The history is diagnostic only and never changes the learned score.
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
        end-to-end by the full-duplex pipeline's ``update_vad(True/False)`` calls
        on framework user_state transitions.
        """
        self._context_eot.reset_turn()
        self._state.reset_turn()
        self._current_eot_score = 0.0
        self._current_eot_text = ""
        self._last_asr_inference_time = None

    @property
    def current_eot_score(self) -> float:
        """Most recent EOT score from ``update_asr`` / ``should_interrupt``.

        Used by ``SemanticInterruptHandler`` to branch
        between hard and soft interrupt tiers based on confidence.
        """
        return self._current_eot_score
