"""Tier labels for realtime interruption policy.

The tier model is intentionally small: it names the policy layer that produced
or dominated a decision without owning LiveKit side effects. The first
migration step uses it as structured observability; later steps can move each
tier's implementation behind the same names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    TIER0_HARD_STOP = "tier0_hard_stop"
    TIER1_REDIRECT = "tier1_redirect"
    TIER2_INTERRUPTION = "tier2_interruption"
    TIER3_BACKCHANNEL_NOISE = "tier3_backchannel_noise"
    TIER4_ATTENTION = "tier4_attention"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TierEvidence:
    tier: Tier
    reason: str
