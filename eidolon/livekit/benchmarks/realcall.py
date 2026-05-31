"""Real-provider call verification for the voice benchmark.

A benchmark can only claim "top-tier" if it provably exercised real STT / TTS /
LLM(brain) calls — not mocks, silence, or a dead worker. This module turns that
into hard evidence: provider identity, minimum audio volume, and (for the real
room) brain-gRPC / STT-stream markers in the per-turn timeline. A run that
cannot prove real calls is marked failed rather than allowed to masquerade as a
pass.

It mirrors the provider/factory path used by ``scripts/smoke_realtime_stack.py``
(``SharedStageFactory.from_config``) for the pre-flight check, and reuses the
existing timeline JSONL attrs (``brain_rpc``, ``stt_stream``, audio-byte counts)
as evidence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from .schema import CaseResult, RunResult
from .timeline import load_timeline_records

# Real provider identifiers used by SharedStageFactory.from_config. Anything
# carrying a mock marker (or absent from this set) is treated as not-real.
REAL_PROVIDER_NAMES = frozenset(
    {
        "eidolon_agent",
        "eidolon_agent_rpc",
        "direct_llm",
        "bailian",
        "sensetime",
        "firered",
        "firered_pvad",
        "silero",
    }
)
MOCK_MARKERS = ("mock", "noop", "fake", "stub", "scripted", "dummy")

# A real Bailian TTS clip for a short sentence is well over this; a mock/empty
# synthesizer is not. 0.25s of 16kHz mono PCM16 is 8000 bytes.
MIN_COMPONENT_TTS_BYTES = 2000
MIN_ROOM_AGENT_AUDIO_BYTES = 8000

_ROOM_CASE_RE = re.compile(r"^voice-bench-(?P<case>.+)-[0-9a-f]{8}$")


def _is_mock(name: str) -> bool:
    lowered = (name or "").lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in MOCK_MARKERS):
        return True
    return lowered not in REAL_PROVIDER_NAMES


def verify_provider_config(provider_config: dict[str, str]) -> list[str]:
    """Fail if any configured provider is a mock / unknown (non-real) provider."""

    failures: list[str] = []
    for role, name in sorted((provider_config or {}).items()):
        if not name:
            continue
        if _is_mock(name):
            failures.append(f"{role} provider {name!r} is not a real provider")
    return failures


def _base_case_id(case_id: str) -> str:
    # component results use "<case>:<component>"; rooms use the bare case id.
    return case_id.split(":", 1)[0]


def group_records_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        room_name = attrs.get("room_name")
        if not isinstance(room_name, str):
            continue
        match = _ROOM_CASE_RE.match(room_name)
        if not match:
            continue
        grouped.setdefault(match.group("case"), []).append(record)
    return grouped


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_brain_rpc_evidence(record: dict[str, Any]) -> bool:
    """Whether a turn record proves a real eidolon_agent brain gRPC stream."""

    attrs = _mapping(record.get("attrs"))
    brain_rpc = _mapping(attrs.get("brain_rpc"))
    timestamps = _mapping(record.get("timestamps"))
    return (
        str(brain_rpc.get("provider") or "").startswith("eidolon_agent")
        and bool(brain_rpc.get("request_id"))
        and "brain_request_sent_at" in timestamps
        and "brain_first_delta_at" in timestamps
    )


def _verify_room_timeline(
    provider_config: dict[str, str],
    case_records: list[dict[str, Any]],
) -> list[str]:
    """Per-case room evidence: a worker turn record with the right STT stream.

    Brain-RPC evidence is intentionally NOT checked here. Interrupt / rollback
    cases capture the *decision* turn (action=cancel/rollback), which has no
    brain generation, so requiring brain marks per case is wrong. The brain is
    proven once per run instead (see ``apply_real_call_verification``); per case
    we rely on agent-audio bytes + the STT stream that transcribed the user.
    """

    failures: list[str] = []
    if not case_records:
        failures.append("no worker timeline records (cannot prove real call)")
        return failures

    stt_provider = (provider_config or {}).get("stt", "")
    if stt_provider:
        stt_ok = any(
            str(_mapping(_mapping(r.get("attrs")).get("stt_stream")).get("provider") or "")
            == stt_provider
            for r in case_records
        )
        if not stt_ok:
            failures.append(f"no STT stream evidence for provider {stt_provider!r}")

    return failures


def verify_real_call(
    case_result: CaseResult,
    *,
    runner: str,
    provider_config: dict[str, str],
    case_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return real-call failures for one case (empty means proven-real)."""

    failures = verify_provider_config(provider_config)
    metrics = case_result.metrics

    if runner == "component":
        component = case_result.case_id.rpartition(":")[2]
        if component == "tts":
            audio_bytes = metrics.get("tts_audio_bytes")
            if not isinstance(audio_bytes, (int, float)) or audio_bytes < MIN_COMPONENT_TTS_BYTES:
                failures.append(
                    f"tts_audio_bytes={audio_bytes} < {MIN_COMPONENT_TTS_BYTES} "
                    "(mock/empty TTS?)"
                )
        elif component == "stt":
            if not metrics.get("stt_nonempty"):
                failures.append("stt produced empty transcript (no real STT output)")
        elif component == "vad":
            inference = metrics.get("vad_inference_count")
            if not isinstance(inference, (int, float)) or inference <= 0:
                failures.append("vad emitted no inference events")
    elif runner == "livekit_room":
        audio_bytes = metrics.get("agent_audio_bytes")
        if not isinstance(audio_bytes, (int, float)) or audio_bytes < MIN_ROOM_AGENT_AUDIO_BYTES:
            failures.append(
                f"agent_audio_bytes={audio_bytes} < {MIN_ROOM_AGENT_AUDIO_BYTES} "
                "(dead/mock agent?)"
            )
        failures.extend(_verify_room_timeline(provider_config, case_records or []))

    return failures


def apply_real_call_verification(
    run: RunResult,
    *,
    strict: bool,
    timeline_path: Path | None = None,
) -> None:
    """Annotate each case with a real-call verdict.

    ``strict`` (default for real-provider runners) turns failures into case
    errors so the run fails. Otherwise failures are recorded advisory in
    ``metrics['real_call_warnings']``. Every case gets a boolean
    ``metrics['real_call_verified']``.
    """

    records_by_case: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    if timeline_path is not None:
        all_records = load_timeline_records(timeline_path)
        records_by_case = group_records_by_case(all_records)

    # Run-level brain proof: the eidolon_agent gRPC only needs to be shown to
    # really stream once across the run (interrupt/rollback turns legitimately
    # have none). If it never appears, every case shares the failure.
    brain_provider = (run.provider_config or {}).get("brain", "")
    run_brain_failure: str | None = None
    if run.runner == "livekit_room" and brain_provider == "eidolon_agent":
        if not any(_has_brain_rpc_evidence(record) for record in all_records):
            run_brain_failure = (
                "no real eidolon_agent brain RPC evidence in the whole run "
                "(provider/request_id + brain_request_sent/first_delta marks)"
            )

    for case in run.cases:
        case_records = records_by_case.get(_base_case_id(case.case_id), [])
        failures = verify_real_call(
            case,
            runner=run.runner,
            provider_config=run.provider_config,
            case_records=case_records,
        )
        if run_brain_failure:
            failures.append(run_brain_failure)
        case.metrics["real_call_verified"] = not failures
        if not failures:
            continue
        if strict:
            case.errors.extend(f"real-call: {failure}" for failure in failures)
            case.passed = False
        else:
            case.metrics["real_call_warnings"] = "; ".join(failures)


async def preflight_real_stack(
    checks: tuple[str, ...] = ("llm", "stt", "tts"),
    *,
    llm_timeout_sec: float = 20.0,
    stt_timeout_sec: float = 30.0,
    tts_timeout_sec: float = 45.0,
    attempts: int = 2,
) -> dict[str, Any]:
    """Exercise the configured real provider stack once before benchmarking.

    Uses the same ``SharedStageFactory.from_config`` path as the worker (no
    mocks). Returns ``{"ok": bool, "config": {...}, "results": [...]}`` so the
    benchmark can abort early with a clear message if a provider is unreachable.

    Each check gets up to ``attempts`` tries: a cold connection to an upstream
    model gateway can spike (e.g. a one-off 28s TCP stall) even when the steady
    state is ~130ms, so the first attempt warms the connection and a transient
    spike does not false-abort the whole run.
    """

    import asyncio
    import math
    import struct
    import time

    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.pipeline.llm import LlmInput
    from eidolon.livekit.common.config import load_effective_config

    def _pcm_sine(duration_ms: int = 300, sample_rate: int = 16_000) -> bytes:
        samples = int(sample_rate * duration_ms / 1000)
        return b"".join(
            struct.pack("<h", int(16_000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(samples)
        )

    async def _check_llm(factory: Any) -> dict[str, Any]:
        out = await factory.llm.chat(
            LlmInput(text="请用中文简短回复：ping"),
            conn_options=APIConnectOptions(max_retry=0, timeout=llm_timeout_sec),
        )
        return {"chars": len(out.text), "preview": out.text[:80]}

    async def _check_tts(factory: Any) -> dict[str, Any]:
        await factory.tts.warmup()
        frames = []
        async for frame in factory.tts.synthesize("你好，测试。"):
            frames.append(frame)
            if len(frames) >= 2:
                break
        if not frames:
            raise RuntimeError("TTS returned no audio frames")
        return {"frames": len(frames), "sample_rate": frames[0].sample_rate}

    async def _check_stt(factory: Any) -> dict[str, Any]:
        text = await factory.stt.recognize_streaming(_pcm_sine())
        return {"chars": len(text), "text": text[:80]}

    cfg = load_effective_config()
    factory = SharedStageFactory.from_config(cfg)
    fns: dict[str, tuple[Callable[[Any], Awaitable[dict[str, Any]]], float]] = {
        "llm": (_check_llm, llm_timeout_sec + 5.0),
        "tts": (_check_tts, tts_timeout_sec),
        "stt": (_check_stt, stt_timeout_sec),
    }
    results: list[dict[str, Any]] = []
    try:
        for name in checks:
            fn, timeout = fns[name]
            started = time.monotonic()
            ok = False
            detail: dict[str, Any] = {}
            used_attempts = 0
            for attempt in range(max(1, attempts)):
                used_attempts = attempt + 1
                try:
                    detail = await asyncio.wait_for(fn(factory), timeout=timeout)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 - surface as data, never crash preflight
                    detail = {"error_type": type(exc).__name__, "error": str(exc)[:300]}
            results.append(
                {
                    "name": name,
                    "ok": ok,
                    "attempts": used_attempts,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "detail": detail,
                }
            )
    finally:
        await factory.stt.shutdown()
        await factory.tts.shutdown()
        if hasattr(factory.llm.llm, "aclose"):
            await factory.llm.llm.aclose()

    return {
        "ok": all(result["ok"] for result in results),
        "config": {
            "brain_provider": cfg.providers.brain_provider,
            "stt_provider": cfg.providers.stt_provider,
            "tts_provider": cfg.providers.tts_provider,
            "vad_provider": cfg.providers.vad_provider,
        },
        "results": results,
    }


def provider_config_from_cfg(cfg: Any) -> dict[str, str]:
    """Extract the {brain,stt,tts,vad} provider names from effective config."""

    providers = cfg.providers
    return {
        "brain": providers.brain_provider,
        "stt": providers.stt_provider,
        "tts": providers.tts_provider,
        "vad": providers.vad_provider,
    }
