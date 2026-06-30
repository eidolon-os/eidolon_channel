#!/usr/bin/env python3
"""Generate benchmark WAV clips from the configured TTS provider."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from eidolon.livekit.agent.factory import SharedStageFactory
from benchmark.audio_assets import (
    CompositeParts,
    synthesize_composite_pcm,
    synthesize_pcm,
    write_wav,
)
from eidolon.livekit.common.config import load_effective_config


DEFAULT_CLIPS: dict[str, str] = {
    "normal_ask_intro": "帮我详细介绍一下这个方案。",
    "normal_followup": "那它的主要风险是什么？",
    "normal_followup_deadline": "那大概多久能做完？",
    "statement_plan_intro": "我在做一个新项目。",
    "statement_followup_risk": "想先把风险理清楚。",
    "hard_stop_stop": "停一下。",
    "hard_stop_dont": "别说了。",
    "hard_stop_ok_dont": "行，不要说了。",
    "topic_switch": "换个话题，我们聊点别的。",
    "correction": "不是，我刚才说错了。",
    "owner_followup_capability": "那你现在能帮我做什么？",
    "language_switch_english": "我们换成英文，然后聊。",
    "wait_one_second": "等一秒。",
    "backchannel_en": "嗯。",
    "backchannel_mm": "嗯嗯。",
    "backchannel_ok": "好。",
    "noise_like_cough": "咳。",
}

# Speech segments joined by trailing silence — utterances with natural internal
# pauses for turn-merge and hesitation scenarios.
COMPOSITE_CLIPS: dict[str, CompositeParts] = {
    "pause_plan_two_parts": [
        ("今天想讨论一下那个方案。", 900),
        ("就是上次说的实时语音方案。", 0),
    ],
    "hesitation_weather": [
        ("嗯，那个。", 600),
        ("帮我查一下明天的天气。", 0),
    ],
}


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="benchmark/audio/generated")
    parser.add_argument("--manifest", default="benchmark/audio/generated/manifest.yaml")
    args = parser.parse_args()

    cfg = load_effective_config()
    stages = SharedStageFactory.components_from_config(cfg)
    await stages.tts.warmup()

    out_dir = Path(args.out_dir)
    manifest: dict[str, dict[str, str]] = {}
    try:
        for clip_id, text in DEFAULT_CLIPS.items():
            pcm, sample_rate = await synthesize_pcm(stages.tts.synthesize, text)
            path = out_dir / f"{clip_id}.wav"
            write_wav(path, pcm, sample_rate=sample_rate)
            manifest[clip_id] = {"text": text, "path": str(path)}
            print(f"generated {clip_id}: {path}")
        for clip_id, parts in COMPOSITE_CLIPS.items():
            pcm, sample_rate = await synthesize_composite_pcm(
                stages.tts.synthesize, parts
            )
            path = out_dir / f"{clip_id}.wav"
            write_wav(path, pcm, sample_rate=sample_rate)
            manifest[clip_id] = {
                "text": "".join(text for text, _silence in parts),
                "path": str(path),
            }
            print(f"generated {clip_id}: {path}")
    finally:
        await stages.tts.shutdown()

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
