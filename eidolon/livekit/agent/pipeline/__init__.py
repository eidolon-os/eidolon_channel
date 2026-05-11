"""Eidolon Voice Agent Pipeline.

A complete production-ready voice agent pipeline built on LiveKit Agents,
supporting both manual (one-shot audio) and streaming (continuous audio)
modes with real-time interruption.
"""

from __future__ import annotations

from .base import BasePipeline
from .llm import LlmInput, LlmOutput, LlmParams, LlmStage, LivekitLlmStage
from .stt import SttParams, SttStage
from .tts import TtsParams, TtsStage
from .types import PipelineCallbacks, PipelineState, generate_turn_id
from .vad import VadStage

__all__ = [
    # Base
    "BasePipeline",
    # Pipeline
    "PipelineState",
    "PipelineCallbacks",
    # VAD
    "VadStage",
    # STT
    "SttStage",
    "SttParams",
    # LLM
    "LlmStage",
    "LivekitLlmStage",
    "LlmInput",
    "LlmOutput",
    "LlmParams",
    # TTS
    "TtsStage",
    "TtsParams",
    # Utilities
    "generate_turn_id",
]
