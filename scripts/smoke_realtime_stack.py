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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable

from benchmark.provider_smoke import (
    DEFAULT_STT_AUDIO as _DEFAULT_STT_AUDIO,
    check_stt as _check_stt,
    check_tts as _check_tts,
)
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
    parser.add_argument(
        "--stt-audio-file",
        type=Path,
        default=_DEFAULT_STT_AUDIO,
        help="PCM16 mono 16kHz WAV containing speech for the real STT check.",
    )
    args = parser.parse_args()

    requested = {x.strip() for x in args.checks.split(",") if x.strip()}
    cfg = load_effective_config()
    # Audio-only component checks must not construct the room-bound Brain or
    # require its device-token authority. Reuse the factory's dedicated
    # component path so provider selection remains identical to production.
    factory = (
        SharedStageFactory.from_config(cfg)
        if "llm" in requested
        else SharedStageFactory.components_from_config(cfg)
    )
    checks: dict[str, Callable[[], Awaitable[dict[str, object]]]] = {
        "llm": lambda: _check_llm(factory, args.llm_timeout_sec),
        "tts": lambda: _check_tts(factory),
        "stt": lambda: _check_stt(factory, args.stt_audio_file),
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
        llm_stage = getattr(factory, "llm", None)
        llm = getattr(llm_stage, "llm", None)
        if llm is not None and hasattr(llm, "aclose"):
            await llm.aclose()

    payload = {
        "config": _config_summary(cfg),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
