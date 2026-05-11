# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""In-process mock plugins conforming to livekit-agents protocols.

Each mock subclasses the appropriate ABC from livekit-agents
(LLM/STT/TTS/VAD) and implements only what's needed to drive a
deterministic, scripted scenario test. They share two design goals:

1. **Faithful protocol behavior**: emit the same event sequence and
   types a real plugin would, so tests catch protocol-level bugs.
2. **Deterministic + fast**: no asyncio sleeps unless explicitly
   requested via ``delayed_ms``; no network; no ONNX.
"""

from .mock_llm import MockLLM, ScriptedReply
from .mock_stt import MockSTT, ScriptedTranscript
from .mock_tts import MockTTS
from .mock_vad import MockVAD, MockVADEvent

__all__ = [
    "MockLLM",
    "ScriptedReply",
    "MockSTT",
    "ScriptedTranscript",
    "MockTTS",
    "MockVAD",
    "MockVADEvent",
]
