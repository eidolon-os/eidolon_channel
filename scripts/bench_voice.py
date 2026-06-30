#!/usr/bin/env python3
"""Run Eidolon realtime voice baseline benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

from benchmark.component_runner import (
    ComponentTimeouts,
    run_component_suite,
    write_component_outputs,
)
from benchmark.headless_runner import (
    run_headless_suite,
    write_headless_outputs,
)
from benchmark.livekit_room_runner import (
    LiveKitRoomOptions,
    run_livekit_room_suite,
    write_livekit_room_outputs,
)
from benchmark.policy_runner import (
    run_policy_suite,
    write_policy_outputs,
)
from benchmark.realcall import (
    apply_real_call_verification,
    preflight_real_stack,
)
from benchmark.compare import load_metrics
from benchmark.report import write_repeated_reports
from benchmark.schema import load_suites
from benchmark.slo import enforcement_failures, evaluate_slo_gates
from benchmark.timeline import (
    TimelineCapture,
    load_timeline_records,
    summarize_timeline_records,
)
from benchmark.timeline_expectations import apply_timeline_expectations
from eidolon.livekit.common.config import load_effective_config


def _default_cases() -> list[str]:
    return [
        str(p)
        for p in sorted(Path("benchmark/cases").glob("*.yaml"))
        if not p.name.endswith("_enforced.yaml")
    ]


async def _preflight_gate(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    checks: tuple[str, ...],
) -> None:
    """Prove the real provider stack is reachable before benchmarking.

    Writes ``preflight.json`` and aborts the run if any provider check fails,
    so a benchmark cannot silently "pass" against a dead/misconfigured stack.
    """

    if args.skip_preflight:
        return
    result = await preflight_real_stack(checks=checks)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not result["ok"]:
        failed = [r["name"] for r in result["results"] if not r["ok"]]
        raise SystemExit(
            f"preflight real-call check failed for {failed}; "
            f"see {output_dir / 'preflight.json'} (use --skip-preflight to bypass)"
        )


def _repeat_dir(output_dir: Path, index: int, repeat: int) -> Path:
    """Raw-output directory for one repeat.

    A single run writes raw output at the top level (unchanged behavior); only
    multi-repeat runs get ``repeat-NN`` subdirectories so the merged report at
    the top level stays the canonical artifact.
    """

    return output_dir if repeat <= 1 else output_dir / f"repeat-{index:02d}"


async def _run_headless(args: argparse.Namespace, suites) -> Path:
    output_dir = Path(args.output_dir) / args.run_id / "headless"
    runs = []
    for index in range(args.repeat):
        run = await run_headless_suite(
            suites,
            root=Path("."),
            run_id=args.run_id,
            llm_mode=args.llm_mode,
        )
        write_headless_outputs(run, _repeat_dir(output_dir, index, args.repeat))
        runs.append(run)
    write_repeated_reports(runs, output_dir)
    return output_dir


def _run_policy(args: argparse.Namespace, suites) -> Path:
    output_dir = Path(args.output_dir) / args.run_id / "policy"
    turn_policy = None
    if args.attention_enforce:
        cfg = load_effective_config()
        turn_policy = replace(
            cfg.turn_policy,
            attention=replace(cfg.turn_policy.attention, enforce=True),
        )
    runs = []
    for index in range(args.repeat):
        run = run_policy_suite(suites, run_id=args.run_id, turn_policy=turn_policy)
        write_policy_outputs(run, _repeat_dir(output_dir, index, args.repeat))
        runs.append(run)
    write_repeated_reports(runs, output_dir)
    return output_dir


async def _run_component(args: argparse.Namespace, suites) -> Path:
    output_dir = Path(args.output_dir) / args.run_id / "component"
    await _preflight_gate(args, output_dir, checks=("stt", "tts"))
    runs = []
    for index in range(args.repeat):
        run = await run_component_suite(
            suites,
            root=Path("."),
            run_id=args.run_id,
            timeouts=ComponentTimeouts(
                vad_sec=args.vad_timeout_sec,
                eot_sec=args.eot_timeout_sec,
                stt_sec=args.stt_timeout_sec,
                tts_warmup_sec=args.tts_warmup_timeout_sec,
                tts_sec=args.tts_timeout_sec,
            ),
        )
        apply_real_call_verification(run, strict=not args.lenient_realcall)
        write_component_outputs(run, _repeat_dir(output_dir, index, args.repeat))
        runs.append(run)
    write_repeated_reports(runs, output_dir)
    return output_dir


def _isolate_loopback_proxy(livekit_url: str) -> None:
    """Match eidolon_admin's supervisor proxy isolation for loopback LiveKit.

    The LiveKit rust FFI client routes ws:// through HTTP(S)_PROXY and does NOT
    honor NO_PROXY, so a proxied shell makes a 127.0.0.1 room join hang. The
    admin supervisor starts every sub-project (incl. channel) with the proxy
    vars wiped + NO_PROXY=loopback; we do the same in-process for a loopback
    URL so the benchmark "starts like admin does" regardless of the shell.
    """

    host = (urlparse(livekit_url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return
    cleared = [
        var
        for var in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
        if os.environ.pop(var, None)
    ]
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1,*.local"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    if cleared:
        print(f"[bench] loopback LiveKit -> proxy isolated (cleared {', '.join(cleared)})")


async def _run_livekit_room(args: argparse.Namespace, suites) -> Path:
    output_dir = Path(args.output_dir) / args.run_id / "livekit_room"
    cfg = load_effective_config()
    _isolate_loopback_proxy(cfg.core.livekit_url)
    preflight_checks = (
        ("stt", "tts")
        if cfg.providers.brain_provider == "eidolon_agent"
        else ("llm", "stt", "tts")
    )
    await _preflight_gate(args, output_dir, checks=preflight_checks)
    if _suite_requires_runtime_identity(suites) and not args.livekit_participant_identity:
        raise SystemExit(
            "livekit_room cases expecting agent replies require "
            "--livekit-participant-identity (or "
            "EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY). The identity must "
            "resolve through admin /api/resolve/{kind}/{identity}."
        )
    timeline_source = cfg.observability.timeline_debug_path
    runs = []
    for index in range(args.repeat):
        # Capture per repeat so each repeat's timeline expectations are checked
        # against only that repeat's worker records, not a pooled snapshot.
        timeline_capture = TimelineCapture.start(timeline_source)
        run = await run_livekit_room_suite(
            suites,
            root=Path("."),
            run_id=args.run_id,
            options=LiveKitRoomOptions(
                timeout_sec=args.livekit_room_timeout_sec,
                settle_after_first_audio_sec=args.livekit_room_settle_sec,
                agent_ready_timeout_sec=args.livekit_agent_ready_timeout_sec,
                agent_name=args.livekit_agent_name,
                participant_identity=args.livekit_participant_identity,
                participant_kind=args.livekit_participant_kind,
            ),
        )
        if args.livekit_timeline_flush_grace_sec > 0:
            await asyncio.sleep(args.livekit_timeline_flush_grace_sec)
        repeat_dir = _repeat_dir(output_dir, index, args.repeat)
        timeline_path = repeat_dir / "turn_timeline.jsonl"
        timeline_capture.write_new_lines(timeline_path)
        apply_timeline_expectations(run, suites, timeline_path)
        apply_real_call_verification(
            run,
            strict=not args.lenient_realcall,
            timeline_path=timeline_path,
            require_brain_evidence=_suite_requires_runtime_identity(suites),
        )
        write_livekit_room_outputs(run, repeat_dir)
        runs.append(run)
    write_repeated_reports(runs, output_dir)
    return output_dir


def _suite_requires_runtime_identity(suites) -> bool:
    return any(
        case.expectations.min_agent_messages > 0
        for suite in suites
        for case in suite.cases
    )


def _slo_enforcement_failures(output_dir: Path) -> list[dict]:
    """Evaluate required SLO gates for a livekit_room run directory.

    Builds the same runner payload the dashboard uses (summary metrics +
    timeline latencies) and returns the required gates that hard-failed.
    """

    metrics = load_metrics(output_dir)
    payload = {
        "name": "livekit_room",
        "summary": metrics.get("summary", {}),
        "timeline": summarize_timeline_records(load_timeline_records(output_dir)),
    }
    return enforcement_failures(evaluate_slo_gates(payload))


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        choices=["policy", "headless", "component", "livekit_room", "all"],
        default="all",
    )
    parser.add_argument("--llm-mode", choices=["mock", "direct"], default="mock")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--output-dir", default="benchmark/runs")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--attention-enforce",
        action="store_true",
        help=(
            "Policy-runner only: evaluate benchmark cases with "
            "turn_policy.attention.enforce=true without changing settings.yaml."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the whole suite N times so per-case p50/p95/jitter come from a "
            "real distribution. Use >=20 before trusting real-provider SLO gates."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the real-provider preflight smoke check (component/livekit_room).",
    )
    parser.add_argument(
        "--lenient-realcall",
        action="store_true",
        help=(
            "Record real-call verification failures as advisory warnings instead "
            "of failing cases. Default is strict for real-provider runners."
        ),
    )
    parser.add_argument(
        "--enforce-slo",
        action="store_true",
        help=(
            "Exit non-zero if any required SLO gate hard-fails on the "
            "livekit_room run (CI / release gate). Advisory and unmet "
            "Phase-2 gates never block."
        ),
    )
    parser.add_argument("--vad-timeout-sec", type=float, default=15.0)
    parser.add_argument("--eot-timeout-sec", type=float, default=5.0)
    parser.add_argument("--stt-timeout-sec", type=float, default=35.0)
    parser.add_argument("--tts-warmup-timeout-sec", type=float, default=25.0)
    parser.add_argument("--tts-timeout-sec", type=float, default=35.0)
    parser.add_argument("--livekit-room-timeout-sec", type=float, default=45.0)
    parser.add_argument("--livekit-room-settle-sec", type=float, default=2.0)
    parser.add_argument("--livekit-agent-ready-timeout-sec", type=float, default=12.0)
    parser.add_argument(
        "--livekit-timeline-flush-grace-sec",
        type=float,
        default=5.0,
        help=(
            "After a real-room case disconnects, wait briefly before capturing "
            "worker timeline JSONL so session_closed flushes are included."
        ),
    )
    parser.add_argument("--livekit-agent-name", default="eidolon")
    parser.add_argument(
        "--livekit-participant-identity",
        default=os.getenv("EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY"),
        help=(
            "Registered admin user/device identity for real-room benchmarks "
            "that trigger eidolon_agent replies."
        ),
    )
    parser.add_argument(
        "--livekit-participant-kind",
        choices=["user", "device"],
        default=os.getenv("EIDOLON_BENCH_LIVEKIT_PARTICIPANT_KIND", "user"),
        help="LiveKit participant metadata.kind used by channel runtime resolver.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    case_paths = args.cases or _default_cases()
    if not case_paths:
        raise SystemExit("no benchmark case files found")
    suites = load_suites(case_paths)

    outputs: list[Path] = []
    room_output: Path | None = None
    if args.runner in ("policy", "all"):
        outputs.append(_run_policy(args, suites))
    if args.runner in ("headless", "all"):
        outputs.append(await _run_headless(args, suites))
    if args.runner in ("component", "all"):
        outputs.append(await _run_component(args, suites))
    if args.runner == "livekit_room":
        room_output = await _run_livekit_room(args, suites)
        outputs.append(room_output)

    for path in outputs:
        print(path)

    if args.enforce_slo and room_output is not None:
        failures = _slo_enforcement_failures(room_output)
        if failures:
            print("SLO ENFORCEMENT FAILED (required gates):")
            for f in failures:
                value = f.get("value")
                value_str = f"{value:.1f}" if isinstance(value, (int, float)) else str(value)
                print(
                    f"  {f['name']} [{f['tier']}] "
                    f"{f['metric']}.{f['statistic']}={value_str} > {f['max_value']:.1f}"
                )
            return 1
        print("SLO enforcement: all required gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
