#!/usr/bin/env python3
"""Generate benchmark WAV clips from the configured TTS provider."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.benchmarks.audio_assets import write_wav
from eidolon.livekit.common.config import load_effective_config


DEFAULT_CLIPS: dict[str, str] = {
    "normal_ask_intro": "帮我详细介绍一下这个方案。",
    "normal_followup": "那它的主要风险是什么？",
    "hard_stop_stop": "停一下。",
    "hard_stop_dont": "别说了。",
    "topic_switch": "换个话题，我们聊点别的。",
    "correction": "不是，我刚才说错了。",
    "backchannel_en": "嗯。",
    "backchannel_ok": "好。",
    "noise_like_cough": "咳。",
}


async def _synthesize_clip(factory: SharedStageFactory, text: str) -> bytes:
    frames = []
    async for frame in factory.tts.synthesize(text):
        frames.append(bytes(frame.data))
    if not frames:
        raise RuntimeError(f"TTS returned no audio for {text!r}")
    return b"".join(frames)


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="benchmarks/audio/generated")
    parser.add_argument("--manifest", default="benchmarks/audio/generated/manifest.yaml")
    args = parser.parse_args()

    cfg = load_effective_config()
    factory = SharedStageFactory.from_config(cfg)
    await factory.tts.warmup()

    out_dir = Path(args.out_dir)
    manifest: dict[str, dict[str, str]] = {}
    try:
        for clip_id, text in DEFAULT_CLIPS.items():
            pcm = await _synthesize_clip(factory, text)
            path = out_dir / f"{clip_id}.wav"
            write_wav(path, pcm, sample_rate=cfg.bailian_tts.sample_rate)
            manifest[clip_id] = {"text": text, "path": str(path)}
            print(f"generated {clip_id}: {path}")
    finally:
        await factory.tts.shutdown()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump({"clips": manifest}, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
