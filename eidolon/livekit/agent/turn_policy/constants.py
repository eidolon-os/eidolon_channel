"""Constants shared by the realtime turn-policy hot path.

The values here are algorithmic/runtime-internal constants. Deployment knobs
belong in ``config/settings.yaml`` and ``InterruptPolicyConfig`` instead.
"""

from __future__ import annotations

TRANSCRIPT_PREVIEW_MAX_CHARS = 80

TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX = "transcript_evidence_hold:"
DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX = "deadline_wait_for_better_transcript:"
SEMANTIC_SCORE_WAIT_REASON_PREFIX = "semantic_score_wait"
STABLE_SIGNAL_WAIT_REASON_PREFIX = "stable_signal_wait"
STABLE_NORMAL_INTERRUPT_REASON_PREFIX = "stable_normal_interrupt"
WEAK_SIGNAL_SHORT_TRANSCRIPT_REASON_PREFIX = "weak_signal_short_transcript"

WEAK_SIGNAL_HOLD_REASON_PREFIXES: tuple[str, ...] = (
    TRANSCRIPT_EVIDENCE_HOLD_REASON_PREFIX,
    DEADLINE_BETTER_TRANSCRIPT_REASON_PREFIX,
    WEAK_SIGNAL_SHORT_TRANSCRIPT_REASON_PREFIX,
    "intent:noise_",
    "intent:backchannel_",
)
