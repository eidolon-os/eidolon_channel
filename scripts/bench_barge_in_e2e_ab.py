#!/usr/bin/env python3
"""Run real-room barge-in A/B validation with real audio fixtures.

This script is the E2E counterpart to ``bench_barge_in_ab.py``. It exercises
the actual LiveKit room boundary, real provider stack, real WAV input assets,
agent audio subscription, worker timeline capture, and timeline expectations.
It deliberately keeps LiveKit as the media engine and varies only the Channel
interruption owner profile.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from eidolon.livekit.benchmarks.livekit_room_runner import (
    LiveKitRoomOptions,
    run_livekit_room_suite,
    write_livekit_room_outputs,
)
from eidolon.livekit.benchmarks.realcall import (
    apply_real_call_verification,
    preflight_real_stack,
)
from eidolon.livekit.benchmarks.report import write_repeated_reports
from eidolon.livekit.benchmarks.schema import BenchmarkSuite, load_suites
from eidolon.livekit.benchmarks.timeline import (
    TimelineCapture,
    load_timeline_records,
    summarize_timeline_records,
)
from eidolon.livekit.benchmarks.timeline_expectations import apply_timeline_expectations
from eidolon.livekit.common.config import load_effective_config


OWNER_PROFILES = ("channel", "livekit_native_adaptive")
DEFAULT_CASES = (
    "benchmarks/cases/barge_in_ab_matrix_enforced.yaml",
    "benchmarks/cases/v1_interrupt_tiers_enforced.yaml",
    "benchmarks/cases/v1_realistic_interaction_flows_enforced.yaml",
    "benchmarks/cases/dogfood_box3_audio_first_enforced.yaml",
)
KEY_LATENCY_METRICS = (
    "timeline_interrupt_speech_to_started_ms",
    "timeline_interrupt_speech_to_first_transcript_ms",
    "timeline_stt_speech_to_actionable_transcript_ms",
    "timeline_interrupt_first_transcript_to_resolved_ms",
    "timeline_interrupt_speech_to_resolved_ms",
    "timeline_vad_start_to_interrupt_resolved",
    "user_done_to_agent_audio_after_user_done_ms",
)
WORKER_READY_MARKERS = (
    "[Server] starting",
    "starting worker",
    "registered worker",
    "starting livekit worker",
)


@dataclass(frozen=True)
class ProfileResult:
    profile: str
    output_dir: Path
    overlay_path: Path | None
    worker_log_path: Path | None
    metrics: dict[str, Any]
    timeline: dict[str, Any]


class ManagedWorker:
    """Own a temporary Channel worker process for one A/B profile."""

    def __init__(
        self,
        *,
        root: Path,
        env: dict[str, str],
        log_path: Path,
        ready_timeout_sec: float,
        stop_timeout_sec: float,
    ) -> None:
        self._root = root
        self._env = env
        self._log_path = log_path
        self._ready_timeout_sec = ready_timeout_sec
        self._stop_timeout_sec = stop_timeout_sec
        self._process: subprocess.Popen[str] | None = None
        self._log_file: Any = None

    @property
    def log_path(self) -> Path:
        return self._log_path

    def __enter__(self) -> "ManagedWorker":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [sys.executable, "-m", "eidolon.livekit.agent.server"],
            cwd=str(self._root),
            env=self._env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._wait_until_ready()

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=self._stop_timeout_sec)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
        self._process = None
        self._log_file = None

    def _wait_until_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self._ready_timeout_sec
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                tail = _tail_text(self._log_path)
                raise RuntimeError(
                    "managed worker exited before readiness; "
                    f"see {self._log_path}\n{tail}"
                )
            text = _tail_text(self._log_path)
            if any(marker in text for marker in WORKER_READY_MARKERS):
                return
            time.sleep(0.25)
        raise TimeoutError(
            "managed worker did not become ready within "
            f"{self._ready_timeout_sec:.1f}s; see {self._log_path}"
        )


def _overlay_payload(
    *,
    interruption_owner: str,
    server_port: int,
    timeline_path: Path,
) -> dict[str, Any]:
    return {
        "core": {"port": server_port},
        "worker": {"num_idle_processes": 0},
        "turn_policy": {
            "interruption_owner": interruption_owner,
            "attention": {"enforce": True},
        },
        "observability": {"timeline_debug_path": str(timeline_path)},
    }


def _write_overlay(
    *,
    profile: str,
    profile_dir: Path,
    server_port: int,
) -> Path:
    overlay_path = profile_dir / "settings.overlay.yaml"
    payload = _overlay_payload(
        interruption_owner=profile,
        server_port=server_port,
        timeline_path=profile_dir / "worker-turn-timeline.jsonl",
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return overlay_path


def _worker_env(overlay_path: Path, profile_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["EIDOLON_ENV"] = env.get("EIDOLON_ENV", "dev")
    env["EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML"] = str(overlay_path)
    env["EIDOLON_CHANNEL_NUM_IDLE_PROCESSES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LOG_DIR", str(profile_dir / "worker_logs"))
    _isolate_loopback_proxy_in_env(env)
    return env


@contextlib.contextmanager
def _temporary_overlay_env(overlay_path: Path):
    keys = ("EIDOLON_ENV", "EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["EIDOLON_ENV"] = os.environ.get("EIDOLON_ENV", "dev")
    os.environ["EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML"] = str(overlay_path)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _isolate_loopback_proxy(livekit_url: str) -> None:
    host = (urlparse(livekit_url).hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return
    _isolate_loopback_proxy_in_env(os.environ)


def _isolate_loopback_proxy_in_env(env: dict[str, str]) -> None:
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(var, None)
    env["NO_PROXY"] = "127.0.0.1,localhost,::1,*.local"
    env["no_proxy"] = env["NO_PROXY"]


async def _preflight_gate(output_dir: Path, *, checks: tuple[str, ...], skip: bool) -> None:
    if skip:
        return
    result = await preflight_real_stack(checks=checks)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not result["ok"]:
        failed = [r["name"] for r in result["results"] if not r["ok"]]
        raise SystemExit(
            f"preflight real-call check failed for {failed}; "
            f"see {output_dir / 'preflight.json'}"
        )


def _suite_requires_runtime_identity(suites: list[BenchmarkSuite]) -> bool:
    return any(
        case.expectations.min_agent_messages > 0
        for suite in suites
        for case in suite.cases
    )


def _repeat_dir(output_dir: Path, index: int, repeat: int) -> Path:
    return output_dir if repeat <= 1 else output_dir / f"repeat-{index:02d}"


async def _run_profile(
    args: argparse.Namespace,
    *,
    profile: str,
    profile_index: int,
    root: Path,
    suites: list[BenchmarkSuite],
    run_root: Path,
) -> ProfileResult:
    profile_dir = run_root / profile
    output_dir = profile_dir / "livekit_room"
    overlay_path: Path | None = None
    worker_log_path: Path | None = None

    if args.manage_worker:
        overlay_path = _write_overlay(
            profile=profile,
            profile_dir=profile_dir,
            server_port=args.worker_base_port + profile_index,
        )
        worker_log_path = profile_dir / "worker.stdout.log"
        overlay_ctx = _temporary_overlay_env(overlay_path)
    else:
        if len(args.profiles) > 1:
            raise SystemExit(
                "A/B with multiple profiles requires --manage-worker so each "
                "owner runs in an isolated worker process."
            )
        overlay_ctx = contextlib.nullcontext()

    with overlay_ctx:
        cfg = load_effective_config()
        _isolate_loopback_proxy(cfg.core.livekit_url)
        if args.manage_worker and str(cfg.turn_policy.interruption_owner) != profile:
            raise SystemExit(
                "profile overlay did not take effect: expected "
                f"{profile}, got {cfg.turn_policy.interruption_owner}"
            )
        preflight_checks = (
            ("stt", "tts")
            if cfg.providers.brain_provider == "eidolon_agent"
            else ("llm", "stt", "tts")
        )
        await _preflight_gate(
            output_dir,
            checks=preflight_checks,
            skip=args.skip_preflight,
        )
        if _suite_requires_runtime_identity(suites) and not args.livekit_participant_identity:
            raise SystemExit(
                "real-room cases expecting agent replies require "
                "--livekit-participant-identity or "
                "EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY."
            )

        worker_ctx: contextlib.AbstractContextManager[Any]
        if args.manage_worker:
            assert overlay_path is not None
            assert worker_log_path is not None
            worker_ctx = ManagedWorker(
                root=root,
                env=_worker_env(overlay_path, profile_dir),
                log_path=worker_log_path,
                ready_timeout_sec=args.worker_ready_timeout_sec,
                stop_timeout_sec=args.worker_stop_timeout_sec,
            )
        else:
            worker_ctx = contextlib.nullcontext()

        runs = []
        with worker_ctx:
            for index in range(args.repeat):
                timeline_capture = TimelineCapture.start(
                    cfg.observability.timeline_debug_path
                )
                run = await run_livekit_room_suite(
                    suites,
                    root=root,
                    run_id=f"{args.run_id}-{profile}",
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

        metrics = write_repeated_reports(runs, output_dir)
        timeline = summarize_timeline_records(load_timeline_records(output_dir))
        return ProfileResult(
            profile=profile,
            output_dir=output_dir,
            overlay_path=overlay_path,
            worker_log_path=worker_log_path,
            metrics=metrics,
            timeline=timeline,
        )


def _existing_worker_pids() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "aux"], text=True)
    except Exception:
        return []
    current_pid = str(os.getpid())
    pids: list[str] = []
    for line in output.splitlines():
        if "eidolon.livekit.agent.server" not in line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        if parts[1] == current_pid:
            continue
        if "bench_barge_in_e2e_ab.py" in line:
            continue
        pids.append(parts[1])
    return pids


def _summary_stat(
    payload: dict[str, Any],
    metric: str,
    stat: str = "p95",
) -> float | None:
    value = (
        payload.get("summary", {})
        .get("metrics", {})
        .get(metric, {})
        .get(stat)
    )
    return float(value) if isinstance(value, (int, float)) else None


def _profile_brief(result: ProfileResult) -> dict[str, Any]:
    summary = result.metrics.get("summary", {})
    return {
        "profile": result.profile,
        "output_dir": str(result.output_dir),
        "overlay_path": str(result.overlay_path) if result.overlay_path else None,
        "worker_log_path": str(result.worker_log_path) if result.worker_log_path else None,
        "total": summary.get("total"),
        "passed": summary.get("passed"),
        "failed": summary.get("failed"),
        "flaky": summary.get("flaky"),
        "pass_rate": summary.get("pass_rate"),
        "key_latencies_p95": {
            metric: _summary_stat(result.metrics, metric, "p95")
            for metric in KEY_LATENCY_METRICS
        },
        "timeline": result.timeline,
        "failed_cases": _failed_case_ids(result.metrics),
    }


def _failed_case_ids(payload: dict[str, Any]) -> list[str]:
    failed = []
    for case_id, info in payload.get("summary", {}).get("per_case", {}).items():
        if isinstance(info, dict) and info.get("pass_rate") != 1.0:
            failed.append(str(case_id))
    return sorted(failed)


def _recommendation(briefs: list[dict[str, Any]]) -> dict[str, str]:
    by_profile = {item["profile"]: item for item in briefs}
    channel = by_profile.get("channel")
    native = by_profile.get("livekit_native_adaptive")
    if channel is None or native is None:
        profile = briefs[0]["profile"] if briefs else "unknown"
        return {
            "decision": "single_profile_baseline",
            "reason": f"只运行了 {profile}，这是基线采样，不构成 A/B 切换结论。",
        }

    channel_rate = float(channel.get("pass_rate") or 0.0)
    native_rate = float(native.get("pass_rate") or 0.0)
    if native_rate < channel_rate:
        return {
            "decision": "keep_channel_owner",
            "reason": "native adaptive 通过率低于 channel owner，不能切换。",
        }
    if channel_rate < 1.0:
        return {
            "decision": "fix_channel_baseline_first",
            "reason": "channel owner 基线未全绿，先修复 owner 链路再比较切换。",
        }

    channel_cancel = _latency_from_brief(channel, "timeline_interrupt_speech_to_resolved_ms")
    native_cancel = _latency_from_brief(native, "timeline_interrupt_speech_to_resolved_ms")
    if channel_cancel is None or native_cancel is None:
        return {
            "decision": "need_more_timeline_evidence",
            "reason": "缺少 VAD 起声到完成的 p95 延迟，不能做可靠切换判断。",
        }
    if native_cancel <= channel_cancel * 1.05:
        return {
            "decision": "native_candidate_needs_context_ledger_review",
            "reason": (
                "native adaptive 在通过率和 cancel 延迟上不差于 channel owner，"
                "但仍需人工确认 partial assistant context 和 playback offset 闭环。"
            ),
        }
    return {
        "decision": "keep_channel_owner",
        "reason": (
            "channel owner 在保持全绿的同时拥有更低的打断完成 p95，"
            "继续作为默认 owner。"
        ),
    }


def _latency_from_brief(brief: dict[str, Any], metric: str) -> float | None:
    value = brief.get("key_latencies_p95", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _write_ab_artifacts(
    *,
    run_root: Path,
    cases: list[str],
    repeat: int,
    managed_worker: bool,
    results: list[ProfileResult],
) -> dict[str, Any]:
    briefs = [_profile_brief(result) for result in results]
    payload = {
        "run_root": str(run_root),
        "cases": cases,
        "repeat": repeat,
        "managed_worker": managed_worker,
        "profiles": briefs,
        "recommendation": _recommendation(briefs),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "e2e_ab_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_root / "e2e_ab_report.md").write_text(
        _render_ab_report(payload),
        encoding="utf-8",
    )
    return payload


def _render_ab_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Barge-in E2E A/B 验证报告",
        "",
        f"- 运行目录：`{payload['run_root']}`",
        f"- 重复次数：`{payload['repeat']}`",
        f"- 临时托管 worker：`{payload['managed_worker']}`",
        "",
        "## 剧本",
        "",
    ]
    lines.extend(f"- `{case}`" for case in payload["cases"])
    lines.extend(
        [
            "",
            "## Owner 对比",
            "",
            "| Profile | 通过率 | 失败 | Flaky | 失败用例 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for profile in payload["profiles"]:
        lines.append(
            "| {profile_name} | {rate} | {failed} | {flaky} | {failed_cases} |".format(
                profile_name=profile["profile"],
                rate=_fmt_rate(profile.get("pass_rate")),
                failed=profile.get("failed"),
                flaky=profile.get("flaky"),
                failed_cases=", ".join(profile.get("failed_cases") or []) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## 关键延迟 p95",
            "",
            "| Metric | channel | livekit_native_adaptive |",
            "| --- | ---: | ---: |",
        ]
    )
    by_profile = {profile["profile"]: profile for profile in payload["profiles"]}
    for metric in KEY_LATENCY_METRICS:
        lines.append(
            "| `{metric}` | {channel} | {native} |".format(
                metric=metric,
                channel=_fmt_ms(
                    by_profile.get("channel", {})
                    .get("key_latencies_p95", {})
                    .get(metric)
                ),
                native=_fmt_ms(
                    by_profile.get("livekit_native_adaptive", {})
                    .get("key_latencies_p95", {})
                    .get(metric)
                ),
            )
        )
    recommendation = payload["recommendation"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 决策：`{recommendation['decision']}`",
            f"- 原因：{recommendation['reason']}",
            "",
            "## Artifact",
            "",
        ]
    )
    for profile in payload["profiles"]:
        lines.append(f"- `{profile['profile']}` report: `{profile['output_dir']}/report.md`")
        if profile.get("worker_log_path"):
            lines.append(f"- `{profile['profile']}` worker log: `{profile['worker_log_path']}`")
        if profile.get("overlay_path"):
            lines.append(f"- `{profile['profile']}` overlay: `{profile['overlay_path']}`")
    return "\n".join(lines)


def _fmt_rate(value: Any) -> str:
    return f"{float(value):.0%}" if isinstance(value, (int, float)) else "-"


def _fmt_ms(value: Any) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else "-"


def _tail_text(path: Path, *, limit: int = 20_000) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    return data[-limit:].decode("utf-8", errors="replace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=OWNER_PROFILES,
        default=list(OWNER_PROFILES),
        help="Owner profiles to run. Multiple profiles require --manage-worker.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Benchmark case YAML files. Defaults to enforced real-audio barge-in suites.",
    )
    parser.add_argument("--output-dir", default="benchmarks/runs")
    parser.add_argument(
        "--run-id",
        default=f"barge-in-e2e-ab-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat count per profile. Use >=20 for release-grade confidence.",
    )
    parser.add_argument(
        "--manage-worker",
        action="store_true",
        help=(
            "Spawn one temporary channel worker per profile with a generated "
            "overlay. Required for true unattended A/B."
        ),
    )
    parser.add_argument(
        "--allow-existing-workers",
        action="store_true",
        help=(
            "Allow managed A/B while other eidolon channel workers are running. "
            "This can make LiveKit dispatch nondeterministic."
        ),
    )
    parser.add_argument("--worker-base-port", type=int, default=18766)
    parser.add_argument("--worker-ready-timeout-sec", type=float, default=25.0)
    parser.add_argument("--worker-stop-timeout-sec", type=float, default=8.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--lenient-realcall", action="store_true")
    parser.add_argument("--livekit-room-timeout-sec", type=float, default=45.0)
    parser.add_argument("--livekit-room-settle-sec", type=float, default=2.0)
    parser.add_argument("--livekit-agent-ready-timeout-sec", type=float, default=12.0)
    parser.add_argument("--livekit-timeline-flush-grace-sec", type=float, default=5.0)
    parser.add_argument("--livekit-agent-name", default="eidolon")
    parser.add_argument(
        "--livekit-participant-identity",
        default=os.environ.get("EIDOLON_BENCH_LIVEKIT_PARTICIPANT_IDENTITY"),
    )
    parser.add_argument(
        "--livekit-participant-kind",
        default=os.environ.get("EIDOLON_BENCH_LIVEKIT_PARTICIPANT_KIND", "user"),
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    root = Path.cwd()
    suites = load_suites(args.cases)
    run_root = Path(args.output_dir) / args.run_id / "barge_in_e2e_ab"

    if args.manage_worker:
        existing = _existing_worker_pids()
        if existing and not args.allow_existing_workers:
            raise SystemExit(
                "existing eidolon channel worker process(es) detected: "
                f"{', '.join(existing)}. Stop them before managed A/B, or pass "
                "--allow-existing-workers if you intentionally accept "
                "nondeterministic LiveKit dispatch."
            )

    results: list[ProfileResult] = []
    for index, profile in enumerate(args.profiles):
        print(f"[barge-in-e2e-ab] running profile={profile}")
        result = await _run_profile(
            args,
            profile=profile,
            profile_index=index,
            root=root,
            suites=suites,
            run_root=run_root,
        )
        results.append(result)
        print(f"[barge-in-e2e-ab] wrote {result.output_dir / 'report.md'}")

    payload = _write_ab_artifacts(
        run_root=run_root,
        cases=args.cases,
        repeat=args.repeat,
        managed_worker=args.manage_worker,
        results=results,
    )
    print(f"[barge-in-e2e-ab] wrote {run_root / 'e2e_ab_report.md'}")
    print(
        "[barge-in-e2e-ab] recommendation="
        f"{payload['recommendation']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
