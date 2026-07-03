# Copyright 2025 Eidolon Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Test harness for LiveKit voice agent pipeline tests.

This package provides:
  - audio: synthetic PCM generators + AudioFrame helpers
  - mocks: in-process mock plugins (LLM/VAD/STT/TTS/EOT) that conform to
    the livekit-agents plugin protocols, suitable for headless / scenario
    testing without external services
  - headless: AgentSession driver that runs in-memory (no LiveKit room),
    feeds scripted audio in, captures audio out (built in P2)
  - conversation: high-level scenario DSL (built in P3)

Layered design (see plan):
  L0 (existing) : per-plugin unit tests in tests/{stt,tts,vad,eot}
  L1 (new)      : stage wrapper + mock backend
  L2 (new)      : full-duplex / half-duplex pipelines + all-mock
  L3 (existing) : real-API smoke (gated by @pytest.mark.live)

This module is `tests/_harness/`. All exports are reusable across the
test tree; production code MUST NOT import from here.
"""
