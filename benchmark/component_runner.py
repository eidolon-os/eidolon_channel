"""Real provider component benchmark runner.

This runner validates the actual VAD / EOT / STT / TTS components outside
LiveKit room wiring. It is intentionally separate from the deterministic
policy/headless runners because external providers are slower and can be
flaky; the value here is honest visibility into real component health.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from livekit.agents import vad as lk_vad

from eidolon.livekit.agent.factory import RealtimeStageBundle, SharedStageFactory
from eidolon.livekit.agent.turn_policy.eot_config import eot_kwargs_from_turn_policy
from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.plugins.eot import ChineseModel
from eidolon.livekit.tests._harness.audio import frames_from_pcm

from .audio_assets import load_clip_pcm
from .device_envelope import render_device_envelope_mic_pcm
from .realcall import provider_config_from_cfg
from .schema import AudioClip, BenchmarkCase, BenchmarkSuite, CaseResult, RunResult, UserStep


@dataclass(frozen=True)
class ComponentTimeouts:
    vad_sec: float = 15.0
    eot_sec: float = 5.0
    stt_sec: float = 35.0
    tts_warmup_sec: float = 25.0
    tts_sec: float = 35.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _unique_clips(suites: list[BenchmarkSuite]) -> list[tuple[BenchmarkCase, AudioClip]]:
    seen: set[str] = set()
    clips: list[tuple[BenchmarkCase, AudioClip]] = []
    for suite in suites:
        for case in suite.cases:
            for clip in case.audio_clips:
                key = clip.path
                if key in seen:
                    continue
                seen.add(key)
                clips.append((case, clip))
    return clips


async def _run_vad_clip(
    stages: RealtimeStageBundle,
    *,
    case: BenchmarkCase,
    clip: AudioClip,
    root: Path,
    timeouts: ComponentTimeouts,
) -> CaseResult:
    started = time.monotonic()
    metrics: dict[str, float | int | str | bool | None] = {}
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if stages.vad is None:
        errors.append("VAD provider is disabled")
        return _component_result(case, "vad", False, started, metrics, events, errors)

    stream = stages.vad.vad.stream()

    async def consume() -> None:
        async for ev in stream:
            event_type = _event_type_name(ev.type)
            payload = {
                "type": event_type,
                "timestamp_ms": round(float(getattr(ev, "timestamp", 0.0)) * 1000),
                "probability": getattr(ev, "probability", None),
                "speaking": getattr(ev, "speaking", None),
            }
            events.append(payload)

    task = asyncio.create_task(consume())
    try:
        pcm, sample_rate = load_clip_pcm(root / clip.path)
        pcm = render_device_envelope_mic_pcm(
            case,
            _step_for_clip(case, clip),
            pcm,
            sample_rate=sample_rate,
        )
        for frame in frames_from_pcm(pcm, sample_rate=sample_rate, frame_ms=20):
            stream.push_frame(frame)
        stream.end_input()
        await asyncio.wait_for(task, timeout=timeouts.vad_sec)

        probabilities = [
            float(ev["probability"])
            for ev in events
            if isinstance(ev.get("probability"), (int, float))
        ]
        start_count = sum(
            1 for ev in events if ev["type"] == lk_vad.VADEventType.START_OF_SPEECH.name
        )
        end_count = sum(
            1 for ev in events if ev["type"] == lk_vad.VADEventType.END_OF_SPEECH.name
        )
        inference_count = sum(
            1 for ev in events if ev["type"] == lk_vad.VADEventType.INFERENCE_DONE.name
        )
        metrics.update(
            {
                "vad_elapsed_ms": _elapsed_ms(started),
                "vad_event_count": len(events),
                "vad_start_count": start_count,
                "vad_end_count": end_count,
                "vad_inference_count": inference_count,
                "vad_max_probability": max(probabilities) if probabilities else None,
                "vad_detected_speech": start_count > 0,
            }
        )
        if inference_count <= 0:
            errors.append("VAD emitted no inference events")
        if clip.intent not in {"noise", "noise_like"} and start_count <= 0:
            errors.append("VAD did not emit START_OF_SPEECH for speech clip")
    except Exception as exc:
        metrics["vad_elapsed_ms"] = _elapsed_ms(started)
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if hasattr(stream, "aclose"):
            with contextlib.suppress(Exception):
                await stream.aclose()

    return _component_result(case, "vad", not errors, started, metrics, events, errors)


async def _run_eot_case(
    model: ChineseModel,
    *,
    case: BenchmarkCase,
    timeouts: ComponentTimeouts,
) -> CaseResult:
    started = time.monotonic()
    metrics: dict[str, float | int | str | bool | None] = {}
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []

    async def score_steps() -> None:
        for step in case.user_steps:
            step_started = time.monotonic()
            should_interrupt = model.should_interrupt(
                step.text,
                vad_active=step.agent_speaking,
                is_final=False,
            )
            decisions.append(
                {
                    "text": step.text,
                    "agent_speaking": step.agent_speaking,
                    "should_interrupt": should_interrupt,
                    "score": model.current_eot_score,
                    "elapsed_ms": _elapsed_ms(step_started),
                }
            )

    try:
        await asyncio.wait_for(score_steps(), timeout=timeouts.eot_sec)
        scores = [
            float(d["score"])
            for d in decisions
            if isinstance(d.get("score"), (int, float))
        ]
        metrics.update(
            {
                "eot_elapsed_ms": _elapsed_ms(started),
                "eot_step_count": len(decisions),
                "eot_max_score": max(scores) if scores else None,
                "eot_interrupt_count": sum(
                    1 for d in decisions if d["should_interrupt"]
                ),
                "eot_expected_cancel": case.expectations.action == "cancel",
            }
        )
        if not decisions:
            errors.append("EOT had no user steps to score")
    except Exception as exc:
        metrics["eot_elapsed_ms"] = _elapsed_ms(started)
        errors.append(f"{type(exc).__name__}: {exc}")

    return _component_result(
        case,
        "eot",
        not errors,
        started,
        metrics,
        [],
        errors,
        decisions=decisions,
    )


async def _run_stt_clip(
    stages: RealtimeStageBundle,
    *,
    case: BenchmarkCase,
    clip: AudioClip,
    root: Path,
    timeouts: ComponentTimeouts,
) -> CaseResult:
    started = time.monotonic()
    metrics: dict[str, float | int | str | bool | None] = {}
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        pcm, _sample_rate = load_clip_pcm(root / clip.path)
        pcm = render_device_envelope_mic_pcm(
            case,
            _step_for_clip(case, clip),
            pcm,
            sample_rate=_sample_rate,
        )
        text = await asyncio.wait_for(
            stages.stt.recognize_streaming(pcm),
            timeout=timeouts.stt_sec,
        )
        normalized_expected = _normalize_text(clip.text)
        normalized_actual = _normalize_text(text)
        empty_allowed = clip.intent in {"noise", "noise_like"}
        metrics.update(
            {
                "stt_elapsed_ms": _elapsed_ms(started),
                "stt_output_chars": len(normalized_actual),
                "stt_expected_chars": len(normalized_expected),
                "stt_exact_match": normalized_actual == normalized_expected,
                "stt_nonempty": bool(normalized_actual),
                "stt_empty_allowed": empty_allowed,
            }
        )
        events.append(
            {
                "type": "stt_transcript",
                "expected": clip.text,
                "actual": text,
            }
        )
        if not normalized_actual and not empty_allowed:
            errors.append("STT returned an empty transcript")
    except Exception as exc:
        metrics["stt_elapsed_ms"] = _elapsed_ms(started)
        errors.append(f"{type(exc).__name__}: {exc}")

    return _component_result(case, "stt", not errors, started, metrics, events, errors)


async def _run_tts_text(
    stages: RealtimeStageBundle,
    *,
    case: BenchmarkCase,
    text: str,
    timeouts: ComponentTimeouts,
) -> CaseResult:
    started = time.monotonic()
    metrics: dict[str, float | int | str | bool | None] = {}
    events: list[dict[str, Any]] = []
    errors: list[str] = []

    async def synthesize() -> list[Any]:
        frames: list[Any] = []
        async for frame in stages.tts.synthesize(text):
            frames.append(frame)
        return frames

    try:
        frames = await asyncio.wait_for(synthesize(), timeout=timeouts.tts_sec)
        audio_bytes = sum(len(bytes(frame.data)) for frame in frames)
        sample_rates = sorted(
            {int(getattr(frame, "sample_rate", 0)) for frame in frames if frame}
        )
        metrics.update(
            {
                "tts_elapsed_ms": _elapsed_ms(started),
                "tts_frame_count": len(frames),
                "tts_audio_bytes": audio_bytes,
                "tts_first_sample_rate": sample_rates[0] if sample_rates else None,
                "tts_nonempty_audio": audio_bytes > 0,
            }
        )
        events.append({"type": "tts_text", "text": text})
        if audio_bytes <= 0:
            errors.append("TTS returned no audio bytes")
    except Exception as exc:
        metrics["tts_elapsed_ms"] = _elapsed_ms(started)
        errors.append(f"{type(exc).__name__}: {exc}")

    return _component_result(case, "tts", not errors, started, metrics, events, errors)


async def run_component_suite(
    suites: list[BenchmarkSuite],
    *,
    root: Path,
    run_id: str | None = None,
    timeouts: ComponentTimeouts | None = None,
) -> RunResult:
    timeouts = timeouts or ComponentTimeouts()
    cfg = load_effective_config()
    stages = SharedStageFactory.components_from_config(cfg)
    eot_model = ChineseModel(**eot_kwargs_from_turn_policy(cfg.turn_policy))
    results: list[CaseResult] = []

    try:
        await asyncio.wait_for(stages.stt.warmup(), timeout=timeouts.stt_sec)
        await asyncio.wait_for(stages.tts.warmup(), timeout=timeouts.tts_warmup_sec)

        for case, clip in _unique_clips(suites):
            results.append(
                await _run_vad_clip(
                    stages,
                    case=case,
                    clip=clip,
                    root=root,
                    timeouts=timeouts,
                )
            )
            results.append(
                await _run_stt_clip(
                    stages,
                    case=case,
                    clip=clip,
                    root=root,
                    timeouts=timeouts,
                )
            )
            results.append(
                await _run_tts_text(
                    stages,
                    case=case,
                    text=clip.text,
                    timeouts=timeouts,
                )
            )

        for suite in suites:
            for case in suite.cases:
                results.append(
                    await _run_eot_case(eot_model, case=case, timeouts=timeouts)
                )
    finally:
        await stages.stt.shutdown()
        await stages.tts.shutdown()
        if stages.vad is not None:
            await stages.vad.shutdown()

    return RunResult(
        run_id=run_id or time.strftime("%Y%m%d-%H%M%S"),
        git_sha=_git_sha(),
        runner="component",
        profile=(
            "real_components:"
            f"vad={cfg.providers.vad_provider},"
            f"stt={cfg.providers.stt_provider},"
            f"tts={cfg.providers.tts_provider},"
            f"eot={cfg.turn_policy.profile}"
        ),
        cases=results,
        provider_config=provider_config_from_cfg(cfg),
    )


def write_component_outputs(run: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run.write_jsonl(output_dir / "component_results.jsonl")


def _component_result(
    case: BenchmarkCase,
    component: str,
    passed: bool,
    started: float,
    metrics: dict[str, float | int | str | bool | None],
    events: list[dict[str, Any]],
    errors: list[str],
    *,
    decisions: list[dict[str, Any]] | None = None,
) -> CaseResult:
    metrics.setdefault("elapsed_ms", _elapsed_ms(started))
    return CaseResult(
        case_id=f"{case.case_id}:{component}",
        suite=f"{case.suite}:{component}",
        runner="component",
        passed=passed,
        metrics=metrics,
        decisions=decisions or [],
        events=events,
        errors=errors,
    )


def _event_type_name(event_type: Any) -> str:
    return getattr(event_type, "name", str(event_type))


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _step_for_clip(case: BenchmarkCase, clip: AudioClip) -> UserStep:
    for step in case.user_steps:
        if step.audio == clip.id:
            return step
    if case.user_steps:
        return case.user_steps[0]
    return UserStep(
        text=clip.text,
        audio=clip.id,
        start_ms=0,
        duration_ms=700,
        agent_speaking=False,
    )


def _normalize_text(text: str) -> str:
    return (
        text.replace("。", "")
        .replace("，", "")
        .replace(",", "")
        .replace(".", "")
        .replace(" ", "")
        .strip()
    )
