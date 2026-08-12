#!/usr/bin/env python
"""Benchmark channel-style voiceprint verification against cached embeddings."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
import wave
from pathlib import Path

from eidolon.livekit.agent.speaker_verification import (
    SpeakerVerificationService,
    VoiceprintStore,
    default_voiceprint_root,
)
from eidolon.livekit.plugins.speaker_verification import (
    default_campplus_model_dir,
    ModelScopeCampPlusSpeakerVerificationProvider,
)


def _crop_wav(src: Path, dst: Path, seconds: int) -> int:
    with wave.open(str(src), "rb") as reader:
        params = reader.getparams()
        frames = min(reader.getnframes(), int(seconds * reader.getframerate()))
        data = reader.readframes(frames)
    with wave.open(str(dst), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(data)
    return int(frames * 1000 / params.framerate)


async def _run(args: argparse.Namespace) -> dict[str, object]:
    store = VoiceprintStore(args.voiceprint_root)
    provider = ModelScopeCampPlusSpeakerVerificationProvider(
        model_dir=args.model_dir,
        voiceprint_root=args.voiceprint_root,
        threshold=args.threshold,
        min_audio_ms=args.min_audio_ms,
    )
    service = SpeakerVerificationService(provider=provider, store=store)
    profile = store.load_profile(tenant_id=args.tenant_id, user_id=args.user_id)
    if profile is None:
        raise SystemExit(f"profile not found for {args.tenant_id}/{args.user_id}")
    embedding = store.load_embedding(profile)
    if embedding is None:
        raise SystemExit(f"profile embedding not found: {profile.embedding_ref}")

    with tempfile.TemporaryDirectory(prefix="eidolon-channel-vp-bench-") as tmp:
        tmp_dir = Path(tmp)
        clips: list[tuple[int, Path, int]] = []
        for seconds in args.durations:
            path = tmp_dir / f"turn_{seconds}s.wav"
            audio_ms = _crop_wav(args.sample, path, seconds)
            clips.append((seconds, path, audio_ms))

        cold_seconds, cold_path, cold_audio_ms = clips[min(1, len(clips) - 1)]
        cold_started = time.perf_counter()
        cold_signal = await service.verify_turn(
            tenant_id=args.tenant_id,
            user_id=args.user_id,
            audio=cold_path.read_bytes(),
            sample_rate=args.sample_rate,
            audio_ms=cold_audio_ms,
        )
        cold_wall_ms = (time.perf_counter() - cold_started) * 1000

        rows = []
        for seconds, path, audio_ms in clips:
            samples = []
            for _ in range(args.rounds):
                started = time.perf_counter()
                signal = await service.verify_turn(
                    tenant_id=args.tenant_id,
                    user_id=args.user_id,
                    audio=path.read_bytes(),
                    sample_rate=args.sample_rate,
                    audio_ms=audio_ms,
                )
                wall_ms = (time.perf_counter() - started) * 1000
                samples.append(
                    {
                        "wall_ms": wall_ms,
                        "latency_ms": signal.latency_ms or wall_ms,
                        "score": signal.score,
                        "known": signal.known,
                        "error": signal.error,
                    }
                )
            kept = samples[args.drop_first :]
            latencies = [float(item["latency_ms"]) for item in kept]
            scores = [float(item["score"]) for item in kept if item["score"] is not None]
            rows.append(
                {
                    "seconds": seconds,
                    "audio_ms": audio_ms,
                    "bytes": path.stat().st_size,
                    "known": kept[-1]["known"],
                    "score_mean": round(statistics.mean(scores), 6) if scores else None,
                    "score_min": round(min(scores), 6) if scores else None,
                    "latency_ms": {
                        "min": round(min(latencies), 1),
                        "p50": round(statistics.median(latencies), 1),
                        "mean": round(statistics.mean(latencies), 1),
                        "max": round(max(latencies), 1),
                    },
                }
            )

    return {
        "tenant_id": args.tenant_id,
        "user_id": args.user_id,
        "provider": provider.provider,
        "model": provider.model,
        "profile_id": profile.profile_id,
        "embedding_ref": profile.embedding_ref,
        "embedding_dim": embedding.dim,
        "threshold": args.threshold,
        "cold_call": {
            "seconds": cold_seconds,
            "wall_ms": round(cold_wall_ms, 1),
            "latency_ms": round(cold_signal.latency_ms or cold_wall_ms, 1),
            "score": cold_signal.score,
            "known": cold_signal.known,
            "error": cold_signal.error,
        },
        "hot_path": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--user-id", default="manson")
    parser.add_argument("--voiceprint-root", type=Path, default=default_voiceprint_root())
    parser.add_argument("--model-dir", type=Path, default=default_campplus_model_dir())
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="enrolled WAV to benchmark; pass an explicit Host-local sample",
    )
    parser.add_argument("--durations", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--drop-first", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.31)
    parser.add_argument("--min-audio-ms", type=int, default=1500)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
