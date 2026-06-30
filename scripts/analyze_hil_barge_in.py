#!/usr/bin/env python3
"""Analyze real-device full-duplex barge-in evidence from worker timeline JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.hil_barge_in import analyze_hil_barge_in


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeline",
        type=Path,
        default=Path("benchmark/runs/channel-worker-turn-timeline.jsonl"),
        help="Timeline JSONL file or directory to inspect.",
    )
    parser.add_argument(
        "--room-contains",
        default="",
        help="Optional substring filter for attrs.room_name.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="Only inspect the latest N matching records.",
    )
    parser.add_argument("--require-cancel", action="store_true")
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--max-suspend-ms", type=float, default=120.0)
    parser.add_argument("--max-cancel-ms", type=float, default=500.0)
    parser.add_argument("--max-resume-ms", type=float, default=900.0)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    report = analyze_hil_barge_in(
        args.timeline,
        room_contains=args.room_contains,
        latest=args.latest,
        require_cancel=args.require_cancel,
        require_resume=args.require_resume,
        max_speech_start_to_suspend_ms=args.max_suspend_ms,
        max_speech_start_to_cancel_ms=args.max_cancel_ms,
        max_speech_start_to_resume_ms=args.max_resume_ms,
    )
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if report.passed else "FAIL"
        print(f"HIL barge-in evidence: {status}")
        print(f"records: {report.records}")
        if report.room_names:
            print("rooms: " + ", ".join(report.room_names))
        if report.findings:
            print("findings:")
            for finding in report.findings:
                print(f"  - {finding}")
        print("evidence:")
        for key, value in payload["evidence"].items():
            print(f"  {key}: {value}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
