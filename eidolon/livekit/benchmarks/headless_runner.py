"""Headless audio replay benchmark runner."""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

from livekit.agents import llm as lk_llm

from eidolon.livekit.agent.factory import SharedStageFactory
from eidolon.livekit.common.config import load_effective_config
from eidolon.livekit.tests._harness.audio import synth_silence
from eidolon.livekit.tests._harness.headless import headless_session
from eidolon.livekit.tests._harness.mocks import (
    MockLLM,
    MockSTT,
    MockTTS,
    MockVAD,
    MockVADEvent,
    ScriptedReply,
    ScriptedTranscript,
)

from .audio_assets import load_clip_pcm
from .schema import BenchmarkCase, BenchmarkSuite, CaseResult, RunResult


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _clip_map(case: BenchmarkCase) -> dict[str, str]:
    return {clip.id: clip.path for clip in case.audio_clips}


def _build_stt_scripts(case: BenchmarkCase) -> list[ScriptedTranscript]:
    scripts: list[ScriptedTranscript] = []
    for step in case.user_steps:
        scripts.append(
            ScriptedTranscript(
                text=step.text,
                interims=list(step.interims),
                interim_gap_ms=50 if step.interims else 0,
                trigger_after_ms=step.start_ms + step.duration_ms + step.final_delay_ms,
            )
        )
    return scripts


def _build_vad_events(case: BenchmarkCase) -> list[MockVADEvent]:
    events: list[MockVADEvent] = []
    for step in case.user_steps:
        events.append(
            MockVADEvent("start", at_ms=step.start_ms, probability=step.vad_probability)
        )
        events.append(
            MockVADEvent(
                "end",
                at_ms=step.start_ms + step.duration_ms,
                probability=0.1,
            )
        )
    return events


async def _run_case(case: BenchmarkCase, root: Path) -> CaseResult:
    errors: list[str] = []
    events: list[dict] = []
    started = time.monotonic()
    clip_paths = _clip_map(case)

    replies = [
        ScriptedReply(when=reply.when, reply=reply.reply)
        for reply in case.agent_replies
    ]
    llm = MockLLM.scripted(replies, default_reply="好的。")
    stt = MockSTT.scripted(_build_stt_scripts(case))
    tts = MockTTS(char_seconds=0.04, chunk_ms=60)
    vad = MockVAD.scripted(_build_vad_events(case))

    try:
        async with headless_session(
            llm=llm,
            stt=stt,
            tts=tts,
            vad=vad,
            real_time_audio=True,
        ) as h:
            cursor_ms = 0
            for step in sorted(case.user_steps, key=lambda s: s.start_ms):
                if step.start_ms > cursor_ms:
                    h.audio_in.feed_pcm(synth_silence((step.start_ms - cursor_ms) / 1000))
                rel = clip_paths.get(step.audio)
                if not rel:
                    raise ValueError(f"{case.case_id}: missing audio clip id {step.audio!r}")
                pcm, _sample_rate = load_clip_pcm(root / rel)
                h.audio_in.feed_pcm(pcm)
                cursor_ms = step.start_ms + step.duration_ms
            h.audio_in.feed_pcm(synth_silence(0.8))

            for step in case.user_steps:
                await h.events.wait_for(
                    lambda e, text=step.text: e.type == "user_input_transcribed"
                    and e.payload.is_final
                    and e.payload.transcript == text,
                    timeout=case.timeout_sec,
                )
            if case.expectations.min_agent_messages:
                await h.events.wait_for(
                    lambda e: e.type == "conversation_item_added"
                    and getattr(e.payload.item, "role", None) == "assistant",
                    timeout=case.timeout_sec,
                )
            await asyncio.sleep(0.25)

            finals = [x.transcript for x in h.events.user_finals()]
            agent_messages = h.events.agent_messages()
            cleared_segments = sum(1 for seg in h.audio_out.segments if seg.cleared)
            if len(finals) < case.expectations.min_user_finals:
                errors.append(
                    f"expected at least {case.expectations.min_user_finals} finals, got {len(finals)}"
                )
            if len(agent_messages) < case.expectations.min_agent_messages:
                errors.append(
                    "expected at least "
                    f"{case.expectations.min_agent_messages} agent messages, got {len(agent_messages)}"
                )
            if case.expectations.agent_audio_cancelled and cleared_segments == 0:
                errors.append("expected interrupted/cleared agent audio segment")

            events = [
                {
                    "type": ev.type,
                    "timestamp_ms": round((ev.timestamp - started) * 1000),
                }
                for ev in h.events.all
            ]
            metrics = {
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "user_final_count": len(finals),
                "agent_message_count": len(agent_messages),
                "audio_bytes": h.audio_out.captured_bytes,
                "cleared_segments": cleared_segments,
                "llm_call_count": llm.call_count,
            }
    except Exception as exc:
        metrics = {"elapsed_ms": round((time.monotonic() - started) * 1000)}
        errors.append(f"{type(exc).__name__}: {exc}")

    return CaseResult(
        case_id=case.case_id,
        suite=case.suite,
        runner="headless",
        passed=not errors,
        metrics=metrics,
        events=events,
        errors=errors,
    )


async def run_headless_suite(
    suites: list[BenchmarkSuite],
    *,
    root: Path,
    run_id: str | None = None,
    llm_mode: str = "mock",
) -> RunResult:
    direct_llm: lk_llm.LLM | None = None
    if llm_mode == "direct":
        cfg = load_effective_config()
        direct_llm = SharedStageFactory.from_config(cfg).llm.llm
    elif llm_mode != "mock":
        raise ValueError("llm_mode must be 'mock' or 'direct'")

    results: list[CaseResult] = []
    for suite in suites:
        for case in suite.cases:
            results.append(await _run_case_with_llm(case, root, direct_llm))
    return RunResult(
        run_id=run_id or time.strftime("%Y%m%d-%H%M%S"),
        git_sha=_git_sha(),
        runner="headless",
        profile=f"headless_audio_replay:{llm_mode}",
        cases=results,
    )


async def _run_case_with_llm(
    case: BenchmarkCase,
    root: Path,
    direct_llm: lk_llm.LLM | None,
) -> CaseResult:
    if direct_llm is None:
        return await _run_case(case, root)

    errors: list[str] = []
    events: list[dict] = []
    started = time.monotonic()
    clip_paths = _clip_map(case)
    stt = MockSTT.scripted(_build_stt_scripts(case))
    tts = MockTTS(char_seconds=0.04, chunk_ms=60)
    vad = MockVAD.scripted(_build_vad_events(case))

    try:
        async with headless_session(
            llm=direct_llm,
            stt=stt,
            tts=tts,
            vad=vad,
            real_time_audio=True,
        ) as h:
            cursor_ms = 0
            for step in sorted(case.user_steps, key=lambda s: s.start_ms):
                if step.start_ms > cursor_ms:
                    h.audio_in.feed_pcm(synth_silence((step.start_ms - cursor_ms) / 1000))
                rel = clip_paths.get(step.audio)
                if not rel:
                    raise ValueError(f"{case.case_id}: missing audio clip id {step.audio!r}")
                pcm, _sample_rate = load_clip_pcm(root / rel)
                h.audio_in.feed_pcm(pcm)
                cursor_ms = step.start_ms + step.duration_ms
            h.audio_in.feed_pcm(synth_silence(0.8))

            for step in case.user_steps:
                await h.events.wait_for(
                    lambda e, text=step.text: e.type == "user_input_transcribed"
                    and e.payload.is_final
                    and e.payload.transcript == text,
                    timeout=case.timeout_sec,
                )
            if case.expectations.min_agent_messages:
                await h.events.wait_for(
                    lambda e: e.type == "conversation_item_added"
                    and getattr(e.payload.item, "role", None) == "assistant",
                    timeout=case.timeout_sec,
                )
            await asyncio.sleep(0.25)

            finals = [x.transcript for x in h.events.user_finals()]
            agent_messages = h.events.agent_messages()
            cleared_segments = sum(1 for seg in h.audio_out.segments if seg.cleared)
            if len(finals) < case.expectations.min_user_finals:
                errors.append(
                    f"expected at least {case.expectations.min_user_finals} finals, got {len(finals)}"
                )
            if len(agent_messages) < case.expectations.min_agent_messages:
                errors.append(
                    "expected at least "
                    f"{case.expectations.min_agent_messages} agent messages, got {len(agent_messages)}"
                )
            events = [
                {
                    "type": ev.type,
                    "timestamp_ms": round((ev.timestamp - started) * 1000),
                }
                for ev in h.events.all
            ]
            metrics = {
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "user_final_count": len(finals),
                "agent_message_count": len(agent_messages),
                "audio_bytes": h.audio_out.captured_bytes,
                "cleared_segments": cleared_segments,
                "llm_call_count": None,
            }
    except Exception as exc:
        metrics = {"elapsed_ms": round((time.monotonic() - started) * 1000)}
        errors.append(f"{type(exc).__name__}: {exc}")

    return CaseResult(
        case_id=case.case_id,
        suite=case.suite,
        runner="headless",
        passed=not errors,
        metrics=metrics,
        events=events,
        errors=errors,
    )


def write_headless_outputs(run: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run.write_jsonl(output_dir / "headless_results.jsonl")
