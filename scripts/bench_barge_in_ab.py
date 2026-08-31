#!/usr/bin/env python3
"""Automate barge-in owner A/B evaluation.

This script is intentionally honest about what can be proven offline:

* channel_owner is evaluated with the deterministic policy benchmark runner.
* livekit_native_adaptive is evaluated as a compatibility/preflight contract.
  A real native adaptive result still requires a running LiveKit room, a worker
  started with the native owner profile, and LiveKit inference credentials or a
  configured local inference endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from eidolon_sdk.biz.contracts import INTERACTION_MODE_HALF_DUPLEX

from eidolon.livekit.agent.runtime import apply_interaction_mode
from eidolon.livekit.agent.server import _use_ptt_pipeline
from eidolon.livekit.agent.full_duplex import StreamingPipeline
from benchmark.policy_runner import (
    run_policy_suite,
    write_policy_outputs,
)
from benchmark.report import write_repeated_reports
from benchmark.schema import load_suites
from eidolon.livekit.common.config import TurnPolicyConfig, load_effective_config
from eidolon.livekit.plugins.stt.bailian import BailianFunASRSTT


DEFAULT_CASES = (
    "benchmark/cases/full_duplex/barge_in_ab_matrix_enforced.yaml",
    "benchmark/cases/full_duplex/barge_in_probe_enforced.yaml",
    "benchmark/cases/full_duplex/v1_interrupt_tiers_enforced.yaml",
    "benchmark/cases/full_duplex/v1_realistic_interaction_flows_enforced.yaml",
    "benchmark/cases/full_duplex/explicit_control_enforced.yaml",
    "benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml",
    "benchmark/cases/shared/offline_policy_regression_enforced.yaml",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _channel_policy(base: TurnPolicyConfig) -> TurnPolicyConfig:
    return replace(
        base,
        interruption_owner="channel",
        attention=replace(base.attention, enforce=True),
    )


def _native_policy(base: TurnPolicyConfig) -> TurnPolicyConfig:
    return replace(base, interruption_owner="livekit_native_adaptive")


def _run_channel_policy(
    *,
    cases: list[str],
    output_dir: Path,
    run_id: str,
    repeat: int,
) -> dict[str, Any]:
    cfg = load_effective_config()
    suites = load_suites(cases)
    policy = _channel_policy(cfg.turn_policy)
    runs = []
    profile = f"{policy.profile}:owner=channel:attention_enforce=true"
    for index in range(repeat):
        run = run_policy_suite(
            suites,
            run_id=f"{run_id}-channel-owner",
            turn_policy=policy,
        )
        run.profile = profile
        repeat_dir = output_dir if repeat == 1 else output_dir / f"repeat-{index:02d}"
        write_policy_outputs(run, repeat_dir)
        runs.append(run)
    return write_repeated_reports(runs, output_dir)


def _native_contract() -> dict[str, Any]:
    cfg = load_effective_config()
    policy = _native_policy(cfg.turn_policy)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "ok": ok,
                "blocking": blocking,
                "detail": detail,
            }
        )

    stt = BailianFunASRSTT(api_url="ws://localhost:9", api_key="contract-test")
    add(
        "bailian_streaming_word_aligned_transcript",
        bool(
            stt.capabilities.streaming
            and stt.capabilities.interim_results
            and stt.capabilities.aligned_transcript == "word"
        ),
        "Bailian FunASR declares streaming interim STT with word-level TimedString output.",
    )

    pipeline = StreamingPipeline.__new__(StreamingPipeline)
    pipeline._turn_policy = policy
    pipeline._allow_interruptions = True
    pipeline._false_interruption_timeout = 6.0
    pipeline._avatar_enabled = False
    interruption = pipeline._build_turn_handling()["interruption"]
    add(
        "pipeline_passes_livekit_adaptive_mode",
        interruption.get("mode") == "adaptive"
        and interruption.get("resume_false_interruption") is True,
        f"turn_handling.interruption={interruption}",
    )

    half_policy, half_allow_interruptions = apply_interaction_mode(
        turn_policy=policy,
        allow_interruptions=True,
        interaction_mode=INTERACTION_MODE_HALF_DUPLEX,
    )
    add(
        "half_duplex_streaming_no_barge_in",
        _use_ptt_pipeline(INTERACTION_MODE_HALF_DUPLEX) is False
        and half_allow_interruptions is False
        and half_policy.attention.enabled is False,
        "half_duplex runs the streaming (EOT) pipeline with barge-in disabled",
    )

    detector_ready = _livekit_inference_ready_detail()
    add(
        "livekit_adaptive_detector_can_be_created_by_environment",
        detector_ready["ok"],
        detector_ready["detail"],
        blocking=False,
    )

    current_owner = getattr(cfg.turn_policy, "interruption_owner", "channel")
    add(
        "effective_config_owner_is_explicit",
        current_owner in {"channel", "livekit_native_adaptive"},
        f"settings owner={current_owner!r}",
    )

    blocking_ok = all(check["ok"] for check in checks if check["blocking"])
    real_room_ready = blocking_ok and detector_ready["ok"]
    return {
        "profile": "livekit_native_adaptive_contract",
        "git_sha": _git_sha(),
        "configured_owner": current_owner,
        "blocking_contract_ok": blocking_ok,
        "real_room_native_ready": real_room_ready,
        "checks": checks,
        "notes": [
            "Native adaptive cannot be scored by the pure policy runner because its owner is LiveKit AgentActivity plus AdaptiveInterruptionDetector.",
            "A trusted native result requires a real-room run with the worker started under turn_policy.interruption_owner=livekit_native_adaptive.",
        ],
    }


def _livekit_inference_ready_detail() -> dict[str, Any]:
    from livekit.agents.inference._utils import (  # type: ignore[attr-defined]
        DEFAULT_INFERENCE_URL,
        STAGING_INFERENCE_URL,
        get_default_inference_url,
    )

    base_url = os.getenv("LIVEKIT_REMOTE_EOT_URL", get_default_inference_url())
    is_default_inference = base_url in {DEFAULT_INFERENCE_URL, STAGING_INFERENCE_URL}
    if not is_default_inference:
        return {
            "ok": True,
            "detail": f"LIVEKIT_REMOTE_EOT_URL points to custom inference endpoint: {base_url}",
        }
    has_key = bool(
        os.getenv("LIVEKIT_INFERENCE_API_KEY") or os.getenv("LIVEKIT_API_KEY")
    )
    has_secret = bool(
        os.getenv("LIVEKIT_INFERENCE_API_SECRET") or os.getenv("LIVEKIT_API_SECRET")
    )
    ok = has_key and has_secret
    return {
        "ok": ok,
        "detail": (
            "Default LiveKit inference URL requires LIVEKIT_INFERENCE_API_KEY/"
            "LIVEKIT_INFERENCE_API_SECRET or LIVEKIT_API_KEY/LIVEKIT_API_SECRET; "
            f"key_present={has_key}, secret_present={has_secret}"
        ),
    }


def _policy_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    per_case = summary.get("per_case", {})
    failed = [
        case_id
        for case_id, info in per_case.items()
        if info.get("pass_rate", 0.0) < 1.0
    ]
    return {
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "flaky": summary.get("flaky", 0),
        "pass_rate": summary["pass_rate"],
        "failed_cases": failed,
        "metrics": summary.get("metrics", {}),
    }


def _build_ab_summary(
    *,
    run_id: str,
    channel_payload: dict[str, Any],
    native_contract: dict[str, Any],
) -> dict[str, Any]:
    channel = _policy_summary(channel_payload)
    native_blocking_ok = bool(native_contract["blocking_contract_ok"])
    native_real_ready = bool(native_contract["real_room_native_ready"])
    channel_all_green = channel["pass_rate"] == 1.0 and channel["flaky"] == 0

    if channel_all_green and not native_real_ready:
        recommendation = "channel_owner_for_production_now"
        confidence = "high_for_current_code_contract_medium_for_real_acoustics"
        reason = (
            "Channel owner passed deterministic barge-in policy matrix. "
            "LiveKit native adaptive contract is code-compatible, but real native A/B "
            "is blocked until adaptive inference credentials/endpoint and a native-profile worker are available."
        )
    elif channel_all_green and native_real_ready:
        recommendation = "run_real_room_native_ab_before_switching"
        confidence = "medium"
        reason = (
            "Both profiles are code-runnable. Native adaptive still needs real-room "
            "latency/false-interruption/context-ledger evidence before it can own production."
        )
    elif not native_blocking_ok:
        recommendation = "fix_native_contract_before_ab"
        confidence = "high"
        reason = "Native adaptive fails blocking local compatibility checks."
    else:
        recommendation = "fix_channel_policy_regressions_before_ab"
        confidence = "high"
        reason = "Channel owner deterministic policy matrix is not green."

    return {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "channel_owner_policy": channel,
        "livekit_native_adaptive_contract": native_contract,
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    channel = summary["channel_owner_policy"]
    native = summary["livekit_native_adaptive_contract"]
    lines = [
        f"# Barge-in Owner A/B 报告：{summary['run_id']}",
        "",
        f"- Git：`{summary['git_sha']}`",
        f"- 推荐：`{summary['recommendation']}`",
        f"- 可信度：`{summary['confidence']}`",
        f"- 结论依据：{summary['reason']}",
        "",
        "## Channel Owner",
        "",
        f"- 通过率：`{channel['passed']}/{channel['total']}`",
        f"- flaky 用例：`{channel['flaky']}`",
    ]
    if channel["failed_cases"]:
        lines.append(f"- 失败用例：`{', '.join(channel['failed_cases'])}`")
    lines.extend(
        [
            "",
            "## LiveKit Native Adaptive",
            "",
            f"- 当前配置 owner：`{native['configured_owner']}`",
            f"- 阻塞合同检查：`{native['blocking_contract_ok']}`",
            f"- 可直接跑真实 native room：`{native['real_room_native_ready']}`",
            "",
            "| 检查 | 结果 | 阻塞 | 说明 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in native["checks"]:
        lines.append(
            "| {name} | {ok} | {blocking} | {detail} |".format(
                name=check["name"],
                ok="pass" if check["ok"] else "fail",
                blocking="yes" if check["blocking"] else "no",
                detail=str(check["detail"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 离线 policy A/B 只能验证 Eidolon channel owner 的裁决质量；LiveKit native adaptive 的关键能力在 LiveKit AgentActivity 与远程/本地 adaptive detector 内，不能用 policy runner 伪造。",
            "- 所以本报告把 native 分成合同检查和真实房间检查：合同不通过时无需 dogfood；合同通过后，必须用 native-profile worker 跑 livekit_room/HIL 才能比较真实延迟、误打断、恢复和上下文 ledger。",
        ]
    )
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--output-dir", default="benchmark/runs")
    parser.add_argument("--run-id", default=f"barge-in-ab-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--require-native-runnable",
        action="store_true",
        help="Exit non-zero when native adaptive detector credentials/endpoint are not ready.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    root = Path(args.output_dir) / args.run_id / "barge_in_ab"
    channel_payload = _run_channel_policy(
        cases=args.cases,
        output_dir=root / "channel_owner_policy",
        run_id=args.run_id,
        repeat=args.repeat,
    )
    native_contract = _native_contract()
    native_path = root / "livekit_native_adaptive_contract.json"
    native_path.write_text(
        json.dumps(native_contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = _build_ab_summary(
        run_id=args.run_id,
        channel_payload=channel_payload,
        native_contract=native_contract,
    )
    (root / "ab_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = _render_markdown(summary)
    report_path = root / "ab_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report_path)
    if channel_payload["summary"]["failed"]:
        return 1
    if args.require_native_runnable and not native_contract["real_room_native_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
