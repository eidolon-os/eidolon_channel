"""Eidolon LiveKit Agent framework."""

from __future__ import annotations

from . import pipeline
from .factory import SharedStageFactory
from .streaming import StreamingPipeline
from .batch import BatchPipeline
from .half_duplex import HalfDuplexPttPipeline

__all__ = [
    "pipeline",
    "SharedStageFactory",
    "StreamingPipeline",
    "BatchPipeline",
    "HalfDuplexPttPipeline",
]
