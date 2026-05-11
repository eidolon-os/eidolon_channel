# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Backward-compatibility shim — the canonical module is now
``eidolon.livekit.plugins.tts._aggregator``.

This file re-exports :class:`SentenceAggregator` so existing imports
like ``from sensetime._aggregator import SentenceAggregator`` keep
working without code changes in downstream consumers.
"""

from eidolon.livekit.plugins.tts._aggregator import SentenceAggregator

__all__ = ["SentenceAggregator"]
