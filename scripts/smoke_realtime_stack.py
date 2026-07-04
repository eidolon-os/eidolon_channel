#!/usr/bin/env python3
"""Smoke-test the configured real-time provider stack.

This script intentionally uses the same config and factory path as the worker.
It does not mock providers and it exits non-zero when a selected check fails.
Secrets are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from livekit.agents.types import APIConnectOptions

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.agent.providers.llm import LlmInput
from eidolon.livekit.common.config import EffectiveAgentConfig, load_effective_config


@dataclass
class SmokeResult:
    name: str
    ok: bool
    elapsed_ms: int
    detail: dict[str, object]


def _pcm_sine(duration_ms: int = 300, sample_rate: int = 16_000) -> bytes:
    samples = int(sample_rate * duration_ms / 1000)
    return b"".join(
        struct.pack(
            "<h",
            int(16_000 * math.sin(2 * math.pi * 440 * i / sample_rate)),
        )
        for i in range(samples)
    )


async def _timed(
    name: str,
    fn: Callable[[], Awaitable[dict[str, object]]],
    *,
    timeout_sec: float | None = None,
) -> SmokeResult:
    started = time.monotonic()
    try:
        detail = await asyncio.wait_for(fn(), timeout=timeout_sec)
        return SmokeResult(
            name=name,
            ok=True,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            detail=detail,
        )
    except Exception as exc:
        message = str(exc).splitlines()[0][:300] if str(exc).splitlines() else repr(exc)
        return SmokeResult(
            name=name,
            ok=False,
            elapsed_ms=round((time.monotonic() - started) * 1000),
            detail={
                "error_type": type(exc).__name__,
                "error": message,
            },
        )


async def _check_llm(factory: SharedStageFactory, timeout: float) -> dict[str, object]:
    out = await factory.llm.chat(
        LlmInput(text="请用中文简短回复：ping"),
        conn_options=APIConnectOptions(max_retry=0, timeout=timeout),
    )
    return {"chars": len(out.text), "preview": out.text[:80]}


async def _check_tts(factory: SharedStageFactory) -> dict[str, object]:
    await factory.tts.warmup()
    frames = []
    async for frame in factory.tts.synthesize("你好，测试。"):
        frames.append(frame)
        if len(frames) >= 2:
            break
    if not frames:
        raise RuntimeError("TTS returned no audio frames")
    return {
        "frames": len(frames),
        "sample_rate": frames[0].sample_rate,
        "samples_per_channel": frames[0].samples_per_channel,
    }


async def _check_stt(factory: SharedStageFactory) -> dict[str, object]:
    text = await factory.stt.recognize_streaming(_pcm_sine())
    return {"chars": len(text), "text": text[:80]}


def _config_summary(cfg: EffectiveAgentConfig) -> dict[str, object]:
    return {
        "brain_provider": cfg.providers.brain_provider,
        "stt_provider": cfg.providers.stt_provider,
        "tts_provider": cfg.providers.tts_provider,
        "vad_provider": cfg.providers.vad_provider,
        "turn_policy_profile": cfg.turn_policy.profile,
        "remote_agent_configured": bool(cfg.remote_agent_rpc.target),
        "llm_base_url_configured": bool(cfg.llm.base_url),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checks",
        default="llm,tts,stt",
        help="Comma-separated checks: llm,tts,stt. LLM uses eidolon_agent when configured.",
    )
    parser.add_argument("--llm-timeout-sec", type=float, default=20.0)
    parser.add_argument("--stt-timeout-sec", type=float, default=30.0)
    parser.add_argument("--tts-timeout-sec", type=float, default=45.0)
    args = parser.parse_args()

    cfg = load_effective_config()
    factory = SharedStageFactory.from_config(cfg)

    requested = {x.strip() for x in args.checks.split(",") if x.strip()}
    checks: dict[str, Callable[[], Awaitable[dict[str, object]]]] = {
        "llm": lambda: _check_llm(factory, args.llm_timeout_sec),
        "tts": lambda: _check_tts(factory),
        "stt": lambda: _check_stt(factory),
    }
    unknown = requested - set(checks)
    if unknown:
        raise ValueError(f"unknown checks: {sorted(unknown)}")

    try:
        timeouts = {
            "llm": args.llm_timeout_sec + 5.0,
            "stt": args.stt_timeout_sec,
            "tts": args.tts_timeout_sec,
        }
        results = [
            await _timed(name, checks[name], timeout_sec=timeouts[name])
            for name in sorted(requested)
        ]
    finally:
        await factory.stt.shutdown()
        await factory.tts.shutdown()
        if hasattr(factory.llm.llm, "aclose"):
            await factory.llm.llm.aclose()

    payload = {
        "config": _config_summary(cfg),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
