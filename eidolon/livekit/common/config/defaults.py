"""Shared default values for realtime interrupt policy assets.

The lexicons themselves now live in ``eidolon_sdk.biz.dialogue_control`` so
that eidolon_agent's reflex layer and this hot path can never drift apart.
This module re-exports them under their historical names for all channel
consumers.
"""

from __future__ import annotations

from eidolon_sdk.biz.dialogue_control import (
    DEFAULT_ATTENTION_EARLY_DUCK_PREFIX_LEXICON,
    DEFAULT_CORRECTION_EXCLUSION_LEXICON,
    DEFAULT_CORRECTION_LEXICON,
    DEFAULT_HARD_STOP_CONTROL_SUFFIXES,
    DEFAULT_HARD_STOP_LEXICON,
    DEFAULT_HARD_STOP_NEGATION_PREFIXES,
    DEFAULT_HARD_STOP_PREFIX_LEXICON,
    DEFAULT_HARD_STOP_SPEECH_VERBS,
    DEFAULT_TOPIC_SWITCH_LEXICON,
)

__all__ = [
    "DEFAULT_ATTENTION_EARLY_DUCK_PREFIX_LEXICON",
    "DEFAULT_CORRECTION_EXCLUSION_LEXICON",
    "DEFAULT_CORRECTION_LEXICON",
    "DEFAULT_HARD_STOP_CONTROL_SUFFIXES",
    "DEFAULT_HARD_STOP_LEXICON",
    "DEFAULT_HARD_STOP_NEGATION_PREFIXES",
    "DEFAULT_HARD_STOP_PREFIX_LEXICON",
    "DEFAULT_HARD_STOP_SPEECH_VERBS",
    "DEFAULT_TOPIC_SWITCH_LEXICON",
]
