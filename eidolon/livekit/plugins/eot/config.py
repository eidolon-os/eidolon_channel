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

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EidolonEOTConfig:
    """All configuration options for the Eidolon EOT plugin."""

    # EOT / turn-end threshold
    min_threshold: float = 0.4
    max_threshold: float = 2.2
    urgent_threshold: float = 0.18
    eot_unlikely_threshold: float = 0.5
    enable_semantic_tail_hang: bool = True

    # Dynamic threshold steps
    t_fast: float = 0.25
    t_mid: float = 1.0
    t_deep: float = 2.0

    # Tail-hang silence (thinking / conjunction endings)
    tail_hang_silence_sec: float = 2.5

    # VAD related
    vad_stale_timeout_sec: float = 0.35
    max_sentence_second: float = 15.0
    min_cut_interval: float = 0.5

    # Interrupt protection
    interrupt_cooldown_seconds: float = 0.8
    # Min speech duration before MinSpeakingDurationPolicy allows a cut.
    # Lowered from 0.1 → 0.05 in the streaming-STT regime: Bailian feeds
    # interim every ~100 ms, and the VAD-active timer is now correctly
    # preserved across turn boundaries (see ``state.reset_turn()``), so a
    # 50 ms floor is enough to filter genuine coughs / clicks while letting
    # high-confidence interim cuts through immediately.
    min_speech_duration_sec: float = 0.05

    # Round 7 G2b: VAD probability gate.
    # When > 0, MinSpeakingDurationPolicy blocks cuts if recent average VAD
    # probability is below this threshold (likely echo / noise rather than
    # real speech). Requires the VAD inference callback to be wired (G6).
    # 0.0 disables the gate (default for backward compatibility).
    # Recommended value for noisy environments after pilot testing: 0.50–0.65.
    min_avg_vad_confidence: float = 0.55

    vad_confidence_window_sec: float = 2.0

    # Semantic interruption: EOT score thresholds (used by EOTScoreSemanticPolicy)
    streaming_eot_base_threshold: float = 0.7
    streaming_eot_weak_threshold: float = 0.5
    semantic_threshold_vad_active_delta: float = 0.1
    is_final_threshold_reduction: float = 0.2

    # Tiered interrupt thresholds (streaming STT mode).
    #
    # When a streaming interim transcript reaches a high EOT score we want
    # different reaction speeds depending on how confident we are:
    #
    #   score ≥ hard_interrupt_score_threshold (default 0.9)
    #     Very confident the user wants to interrupt — bypass soft stage,
    #     hard-cut the agent immediately. Used for "停, I 不想听" and
    #     similar where the linguistic + VAD evidence is overwhelming.
    #
    #   soft_interrupt_score_threshold ≤ score < hard_interrupt_score_threshold
    #     Likely interrupt but worth a brief grace window in case the
    #     speech turns out to be a backchannel or noise. Enter soft
    #     interrupt with ``soft_interrupt_timeout_sec`` (default 0.5 s).
    #     If the user falls silent during the window we cancel; if not we
    #     upgrade to hard.
    #
    # The previous SenseAudio-era setup used a single 2.0 s timeout because
    # we only had ``result_final`` to react to and needed time to confirm.
    # In streaming mode we already see the score climb through interims,
    # so the grace window can be much shorter.
    soft_interrupt_score_threshold: float = 0.8
    hard_interrupt_score_threshold: float = 0.9
    soft_interrupt_timeout_sec: float = 0.5

    # ──────────────────────────────────────────────────────────────────
    # Ducking mixer + early-resume watcher
    # ──────────────────────────────────────────────────────────────────
    # Phase C architecture: VAD start → DuckingMixer.duck() (50 ms fade-out
    # to silence). TTS keeps generating frames; after fade-out completes,
    # the mixer **buffers** subsequent frames in memory (pause at our
    # layer). During the suspend window an "early-resume watcher" listens
    # to STT interim and decides:
    #
    #   strong intent / score ≥ ``early_cancel_score_threshold``
    #     → mixer.cancel() + session.interrupt()
    #       Buffer is discarded (real interrupt, no resume).
    #
    #   semantic score 0.0 (filler/too-short) OR
    #   score ≤ ``early_resume_score_threshold``
    #     → mixer.unduck()
    #       Buffered frames are drained with fade-in ramp — user hears
    #       the agent's speech from the suspension point with no content
    #       loss (false interruption, smooth resume).
    #
    #   neither → wait until ``duck_suspend_timeout_sec`` then default to
    #             unduck (graceful fallback if no STT interim arrives)
    #
    # If you need to re-test the SenseAudio-style "framework auto-interrupt"
    # path, set ``duck_enabled = False``; the rest of the pipeline still
    # works using soft/hard interrupt fallback.

    duck_enabled: bool = True
    """Master switch. False → keep current (post-2ec587b) EOT-only path."""

    duck_fade_ms: int = 50
    """Linear-ramp duration for fade-out (NORMAL→SUSPENDED). 50 ms is
    short enough not to feel like a "dip" yet long enough to avoid
    clicks from instantaneous gain change."""

    duck_fade_in_ms: int = 200
    """Linear-ramp duration for fade-in (SUSPENDED→NORMAL). Longer than
    ``duck_fade_ms`` for a more natural, gradual recovery — mimics how
    humans raise their voice back after yielding ("slow in, fast out").
    200 ms smooths the unduck without feeling sluggish."""

    duck_suspend_volume: float = 0.0
    """Target volume at the end of fade-out. ``0.0`` = full silence.
    After fade-out completes, frames are buffered (not forwarded)
    regardless of this value."""

    duck_buffer_max_sec: float = 2.0
    """Maximum audio duration to buffer during SUSPENDED. Safety cap to
    bound memory. In normal operation the suspend window (0.8 s) is well
    below this limit. If exceeded, excess frames are dropped."""

    duck_suspend_timeout_sec: float = 0.8
    """Maximum time to stay in SUSPENDED before auto-unduck if early-resume
    watcher emits no decision. Should be ≥ Bailian interim latency P95
    (typically 200-400 ms after VAD start). 0.8 s gives ASR more room to
    deliver a first interim before the timeout fires, reducing spurious
    "no-interim → default unduck" cycles. Below 0.3 s risks "always
    default-resume"."""

    duck_early_cancel_score_threshold: float = 0.7
    """Score threshold during the suspend window above which we call
    ``cancel()`` immediately (don't wait for final). Lower than the
    non-ducking ``hard_interrupt_score_threshold`` because we already
    paid the duck overhead — once we're suspended, we want decisive
    cancel rather than another wait cycle."""

    duck_early_resume_score_threshold: float = 0.2
    """Score threshold during the suspend window below which we call
    ``unduck()`` immediately, classifying as false interrupt. Backchannels
    ("嗯", "好的") usually score < 0.1 via semantic_completeness_score's
    too-short / filler gates; this catches partial-low-confidence
    transcripts too."""

    duck_cooldown_sec: float = 0.8
    """Minimum interval between unduck and the next duck trigger. Prevents
    "volume yo-yo" when the user gives frequent backchannels or in noisy
    environments where VAD rapidly toggles. During cooldown, new duck
    requests are silently skipped."""

    # ASR final-cut policy: minimum silence after final transcript before
    # allowing a cut. Filters jittery finals.
    final_cut_min_silence_sec: float = 0.2

    # Context enhancement
    utterance_end_max_history: int = 3
    enable_user_profile: bool = True
    enable_temporary_compensations: bool = True
    similarity_threshold: float = 0.85

    # ContextEnhancedEot cooldown (seconds between cuts)
    # Uses the same value as interrupt_cooldown_seconds for consistency.
    context_eot_cooldown_sec: float = 0.8

    # VAD stability: rapid toggle detection
    vad_flip_window_sec: float = 1.0
    vad_flip_count_threshold: int = 3

    # ASR stability: requires ASR text to be stable before allowing cut
    asr_stability_short_sec: float = 0.10
    asr_stability_long_sec: float = 0.40
    asr_stability_long_char_threshold: int = 10

    # Noise mode
    noise_mode_duration_sec: float = 2.0
    noise_mode_silence_threshold_sec: float = 0.6

    # ──────────────────────────────────────────────────────────────────
    # Filler word injection (latency masking)
    # ──────────────────────────────────────────────────────────────────
    filler_enabled: bool = False
    """Pre-synthesize short filler phrases at startup and inject them
    between user-done and agent-starts to mask the ASR+LLM latency gap.
    Disabled by default — pre-recorded filler clips sound mechanical;
    revisit when TTS supports real-time prosody-matched fillers."""

    filler_phrases: tuple[str, ...] = ("嗯...", "好的...", "让我想想...")
    """Phrases to pre-synthesize at pipeline warmup."""

    # ──────────────────────────────────────────────────────────────────
    # Interrupted content tracking
    # ──────────────────────────────────────────────────────────────────
    interrupted_context_enabled: bool = True
    """When the agent is interrupted mid-speech, track what it was saying
    and inject context into the next LLM turn so it can optionally
    reference the interrupted content."""

    interrupted_context_max_age_sec: float = 30.0
    """Maximum age of interrupted context before it's discarded."""

    # Model path (default points to the bundled FireRed model inside the plugin)
    # Can be overridden via EIDOLON_EOT_MODEL_DIR env var or model_dir argument
    model_dir: str | None = None
    prefer_multilingual: bool = False
