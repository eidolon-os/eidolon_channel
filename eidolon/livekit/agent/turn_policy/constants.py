"""Constants shared by the realtime turn-policy hot path.

The values here are algorithmic/runtime-internal constants. Deployment knobs
belong in ``config/settings.yaml`` and ``InterruptPolicyConfig`` instead.
"""

from __future__ import annotations

INTERRUPT_TEXT_TRAILING_CHARS = "。.!？?！,， "

ASR_EXACT_CANONICALIZATIONS: dict[str, str] = {
    # Short correction fragments are ambiguous; canonicalize them so the
    # decider holds for a follow-up interim instead of treating them as normal
    # speech.
    "是我": "我刚",
}

ASR_PREFIX_CANONICALIZATIONS: tuple[tuple[str, str], ...] = (
    # Bailian/FunASR can briefly hear "换个..." as "半个..." in topic-switch
    # clips. Timeline logging still keeps the original transcript text.
    ("半个", "换个"),
    # Correction clips such as "不是，我刚才..." may briefly lose the leading
    # negation and arrive as "是我刚...".
    ("是我刚", "我刚才"),
)

REPEATED_NOISE_CHARS = "啊嗯哈咳哎哦唉"

TRANSCRIPT_PREVIEW_MAX_CHARS = 80

TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX = "transcript_evidence_hold:"
DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX = "deadline_wait_for_better_transcript:"
SEMANTIC_SCORE_WAIT_REASON_PREFIX = "semantic_score_wait"
STABLE_SIGNAL_WAIT_REASON_PREFIX = "stable_signal_wait"
STABLE_NORMAL_INTERRUPT_REASON_PREFIX = "stable_normal_interrupt"

WEAK_SIGNAL_HOLD_REASON_PREFIXES: tuple[str, ...] = (
    TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX,
    DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX,
    "intent:noise_",
    "intent:backchannel_",
)
