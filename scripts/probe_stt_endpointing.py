#!/usr/bin/env python3
"""Probe FunASR sentence-endpointing sensitivity to ``max_sentence_silence``.

Validates whether lowering the STT endpointing silence (the ``eos`` knob that
cuts first-audio latency) changes mid-utterance / hesitation splitting. It
synthesizes hesitant utterances via the configured TTS (a fragment, a spoken
filler like 嗯/那个, then a completion, separated by short gaps), saves them as
reusable WAV fixtures under ``benchmarks/audio/generated/``, then streams each
through the real STT at several ``max_sentence_silence`` values and reports how
many FINAL transcripts come back (>1 = a premature split).

Finding (2026-05-31): splits are identical across eos=400/600/800 — the silence
knob governs only trailing-silence finalization (the latency win), not the
internal ~300ms mid-utterance segmentation. So eos=400 is safe vs the 800
default. Run against your own provider before trusting in production:

    ./.venv/bin/python scripts/probe_stt_endpointing.py
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import wave
from pathlib import Path

from livekit.agents.stt import SpeechEventType

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.plugins.stt.bailian.stt import BailianFunASRSTT
from eidolon.livekit.tests._harness.audio import frames_from_pcm, synth_silence

SR = 16_000
OUTDIR = Path("benchmarks/audio/generated")
EOS_VALUES = (400, 600, 800)

# (label, fragment_a, spoken_filler, gap_ms, fragment_b) — one intended utterance.
CASES = [
    ("hesit_um", "帮我详细介绍一下", "嗯", 200, "这个方案的实时控制部分"),
    ("hesit_nage", "我想问一下", "那个", 250, "时间安排是怎么样的"),
    ("hesit_e", "这个", "呃", 300, "大概需要多长时间"),
]


async def _synth(factory: SharedStageFactory, text: str) -> bytes:
    chunks: list[bytes] = []
    async for frame in factory.tts.synthesize(text):
        chunks.append(bytes(frame.data))
    return b"".join(chunks)


def _save_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


async def _finals(base_cfg, eos: int, clip: bytes) -> list[str]:
    cfg = dataclasses.replace(base_cfg, max_sentence_silence_ms=eos)
    stt = BailianFunASRSTT(config=cfg)
    stream = stt.stream()
    for frame in frames_from_pcm(clip, sample_rate=SR, frame_ms=20):
        stream.push_frame(frame)
    stream.end_input()
    finals: list[str] = []

    async def read() -> None:
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                finals.append(ev.alternatives[0].text)

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(read(), timeout=20)
    with contextlib.suppress(Exception):
        await stream.aclose()
    return finals


async def _main() -> int:
    cfg = load_effective_config()
    factory = SharedStageFactory.from_config(cfg)
    await factory.tts.warmup()
    try:
        for label, a_txt, filler, gap, b_txt in CASES:
            a = await _synth(factory, a_txt)
            f = await _synth(factory, filler)
            b = await _synth(factory, b_txt)
            gap_pcm = synth_silence(gap / 1000.0)
            clip = a + gap_pcm + f + gap_pcm + b
            _save_wav(OUTDIR / f"{label}.wav", clip)
            print(f"\n=== {label}: '{a_txt}…{filler}…{b_txt}' ({gap}ms gaps) ===")
            for eos in EOS_VALUES:
                finals = await _finals(cfg.bailian_stt, eos, clip)
                tag = "SPLIT" if len(finals) > 1 else "ok(1 final)"
                print(f"  eos={eos:>4} -> {len(finals)} final {finals}  {tag}")
    finally:
        await factory.tts.shutdown()
        await factory.stt.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
