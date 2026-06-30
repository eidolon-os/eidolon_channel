"""Compare a local hard-stop detector experiment against LiveKit/STT latency."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from benchmark.hard_stop_detector import (
    DetectorComparison,
    load_livekit_stt_actionable_metrics,
    run_template_detector_comparison,
    summarize_metric,
)


DEFAULT_RUN_DIR = Path(
    "benchmark/runs/realistic-extended-actionable-transcript-repeat3-20260608/livekit_room"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmark/audio/generated/manifest.yaml")
    parser.add_argument("--livekit-run", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--out-dir",
        default="benchmark/runs/hard-stop-detector-comparison-20260608",
    )
    parser.add_argument("--min-window-ms", type=int, default=180)
    parser.add_argument("--max-window-ms", type=int, default=900)
    parser.add_argument("--step-ms", type=int, default=20)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    livekit_run = Path(args.livekit_run) if args.livekit_run else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = _load_manifest_clips(manifest_path)
    comparison = run_template_detector_comparison(
        clips=clips,
        min_window_ms=args.min_window_ms,
        max_window_ms=args.max_window_ms,
        step_ms=args.step_ms,
    )
    livekit_rows = (
        load_livekit_stt_actionable_metrics(livekit_run)
        if livekit_run and livekit_run.exists()
        else []
    )

    payload = {
        "manifest": str(manifest_path),
        "livekit_run": str(livekit_run) if livekit_run else None,
        "detector": _comparison_to_json(comparison),
        "livekit_stt": {
            "rows": livekit_rows,
            "summary": _summarize_livekit_rows(livekit_rows),
        },
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        _render_report(
            comparison=comparison,
            livekit_rows=livekit_rows,
            manifest_path=manifest_path,
            livekit_run=livekit_run,
            min_window_ms=args.min_window_ms,
            max_window_ms=args.max_window_ms,
            step_ms=args.step_ms,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'report.md'}")
    return 0


def _load_manifest_clips(manifest_path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw_clips = raw.get("clips") if isinstance(raw.get("clips"), dict) else {}
    clips: dict[str, dict[str, Any]] = {}
    for clip_id, item in raw_clips.items():
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path") or ""))
        if not path.is_absolute():
            path = _resolve_clip_path(manifest_path, path)
        clips[str(clip_id)] = {
            **item,
            "path": str(path),
            "intent": item.get("intent") or _infer_intent(str(clip_id)),
            "sample_rate": int(item.get("sample_rate") or 16_000),
        }
    return clips


def _resolve_clip_path(manifest_path: Path, path: Path) -> Path:
    if path.parts and path.parts[0] == "benchmark":
        return manifest_path.parent.parent.parent / path.relative_to("benchmark")
    return manifest_path.parent / path


def _infer_intent(clip_id: str) -> str:
    if clip_id.startswith("hard_stop"):
        return "hard_stop"
    if clip_id.startswith("topic_switch"):
        return "topic_switch"
    if clip_id.startswith("correction"):
        return "correction"
    if clip_id.startswith("backchannel"):
        return "backchannel"
    if clip_id.startswith("noise"):
        return "noise"
    if clip_id.startswith("normal"):
        return "normal"
    return "unknown"


def _comparison_to_json(comparison: DetectorComparison) -> dict[str, Any]:
    return {
        "threshold": comparison.threshold,
        "positive_count": comparison.positive_count,
        "negative_count": comparison.negative_count,
        "true_positive_count": comparison.true_positive_count,
        "false_positive_count": comparison.false_positive_count,
        "false_negative_count": comparison.false_negative_count,
        "precision": comparison.precision,
        "recall": comparison.recall,
        "scores": [asdict(score) for score in comparison.scores],
    }


def _summarize_livekit_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: summarize_metric([row[key] for row in rows if row.get(key) is not None])
        for key in (
            "speech_to_first_transcript_ms",
            "speech_to_actionable_ms",
            "first_to_actionable_ms",
            "actionable_to_resolved_ms",
            "resolved_ms",
        )
    }


def _render_report(
    *,
    comparison: DetectorComparison,
    livekit_rows: list[dict[str, Any]],
    manifest_path: Path,
    livekit_run: Path | None,
    min_window_ms: int,
    max_window_ms: int,
    step_ms: int,
) -> str:
    livekit_summary = _summarize_livekit_rows(livekit_rows)
    positive_best = [
        score.best_score for score in comparison.scores if score.expected_positive
    ]
    negative_best = [
        score.best_score for score in comparison.scores if not score.expected_positive
    ]
    min_positive = min(positive_best) if positive_best else None
    max_negative = max(negative_best) if negative_best else None
    separation_margin = (
        min_positive - max_negative
        if min_positive is not None and max_negative is not None
        else None
    )
    dangerous_false_positives = [
        score
        for score in comparison.scores
        if not score.expected_positive
        and score.detected
        and score.intent not in {"topic_switch", "correction"}
    ]
    hard_stop_detections = [
        score.detection_ms
        for score in comparison.scores
        if score.expected_positive and score.detected and score.detection_ms is not None
    ]
    detector_summary = summarize_metric(hard_stop_detections)
    stt_actionable = livekit_summary["speech_to_actionable_ms"].get("p50")
    detector_p50 = detector_summary.get("p50")
    latency_delta = (
        stt_actionable - detector_p50
        if isinstance(stt_actionable, float) and isinstance(detector_p50, float)
        else None
    )

    lines = [
        "# Tier0 本地硬停检测对比实验",
        "",
        "## 结论",
        "",
        _decision_sentence(
            comparison,
            latency_delta,
            dangerous_false_positive_count=len(dangerous_false_positives),
            separation_margin=separation_margin,
        ),
        "",
        "这个实验不是生产检测器实现，只是用现有生成音频资产验证一个问题：在不等待 STT actionable 文本的情况下，本地音频热路径是否有机会更快识别“别说了 / 停一下”这类 Tier0 硬停。",
        "",
        "## 实验配置",
        "",
        f"- 音频清单：`{manifest_path}`",
        f"- LiveKit/STT 基线：`{livekit_run}`" if livekit_run else "- LiveKit/STT 基线：未提供",
        f"- streaming prefix 窗口：{min_window_ms}ms 到 {max_window_ms}ms，步长 {step_ms}ms",
        f"- 阈值：{comparison.threshold:.3f}（由负样本最高分 + margin 自动校准）",
        "",
        "## 本地 Detector 结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 正样本数 | {comparison.positive_count} |",
        f"| 负样本数 | {comparison.negative_count} |",
        f"| TP | {comparison.true_positive_count} |",
        f"| FP | {comparison.false_positive_count} |",
        f"| FN | {comparison.false_negative_count} |",
        f"| precision | {_fmt_optional(comparison.precision)} |",
        f"| recall | {_fmt_optional(comparison.recall)} |",
        f"| hard-stop detection p50 | {_fmt_ms(detector_summary.get('p50'))} |",
        f"| hard-stop detection p95 | {_fmt_ms(detector_summary.get('p95'))} |",
        f"| positive min score | {_fmt_optional(min_positive)} |",
        f"| negative max score | {_fmt_optional(max_negative)} |",
        f"| score separation margin | {_fmt_optional(separation_margin)} |",
        f"| 危险误报（normal/backchannel/noise） | {len(dangerous_false_positives)} |",
        "",
        "## Clip 明细",
        "",
        "| clip | intent | 期望硬停 | 检出 | 检出延迟 | best score | matched |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for score in comparison.scores:
        lines.append(
            "| "
            + " | ".join(
                [
                    score.clip_id,
                    score.intent,
                    "是" if score.expected_positive else "否",
                    "是" if score.detected else "否",
                    _fmt_ms(score.detection_ms),
                    f"{score.best_score:.3f}",
                    score.matched_template or "-",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## LiveKit/STT 基线",
            "",
            "| 指标 | 样本数 | p50 | p95 | max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("speech_to_first_transcript_ms", "speech -> first transcript"),
        ("speech_to_actionable_ms", "speech -> actionable transcript"),
        ("first_to_actionable_ms", "first -> actionable transcript"),
        ("actionable_to_resolved_ms", "actionable -> resolved"),
        ("resolved_ms", "speech -> interrupt resolved"),
    ):
        summary = livekit_summary[key]
        lines.append(
            f"| {label} | {summary['count']} | {_fmt_ms(summary['p50'])} | "
            f"{_fmt_ms(summary['p95'])} | {_fmt_ms(summary['max'])} |"
        )

    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 速度上，本地 detector 在 hard-stop 样本上比 STT actionable 更早；这说明“本地 Tier0 快速证据”这个方向值得继续验证。",
            "- 安全性上，当前 template detector 的 score separation 太窄，并且把 cough/noise 误报成 hard-stop；这说明它不能直接进入在线热路径。",
            "- 如果 detector 在 hard-stop 上稳定早于 STT actionable，且 FP 为 0，可以进入在线原型阶段。",
            "- 在线原型只应该挂在 Tier0 硬停，不替代 VAD/EOT/policy chain；它的输出应是强证据事件，由现有 turn policy 消费。",
            "- 当前数据来自生成 TTS，覆盖面不足。进入热路径前还需要多说话人、多语速、噪声、远场、相似短句和真实 dogfood 音频回放测试。",
            "",
        ]
    )
    return "\n".join(lines)


def _decision_sentence(
    comparison: DetectorComparison,
    latency_delta: float | None,
    *,
    dangerous_false_positive_count: int,
    separation_margin: float | None,
) -> str:
    clean_detector = (
        comparison.false_positive_count == 0
        and comparison.false_negative_count == 0
    )
    safe_margin = separation_margin is not None and separation_margin >= 0.03
    if clean_detector and safe_margin and latency_delta is not None and latency_delta > 100:
        return (
            f"结果倾向于继续做在线原型：本地 detector 在当前样本上无误报/漏报，"
            f"p50 比 STT actionable 约快 {latency_delta:.1f}ms。"
        )
    if dangerous_false_positive_count:
        return (
            "结果不建议直接进入在线热路径：虽然 hard-stop 检出更快，"
            f"但出现 {dangerous_false_positive_count} 个 normal/backchannel/noise 危险误报。"
        )
    if clean_detector and not safe_margin:
        return "结果可以继续研究但不能上线：当前样本无误报/漏报，但正负样本分数间隔太窄，鲁棒性不足。"
    if clean_detector:
        return "结果可以继续观察：本地 detector 在当前样本上无误报/漏报，但相对 STT 的速度收益还不够明确。"
    return "结果不建议直接进入在线热路径：当前 detector 已出现误报或漏报，需要先扩充数据和改进模型。"


def _fmt_ms(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}ms"
    return "-"


def _fmt_optional(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())
