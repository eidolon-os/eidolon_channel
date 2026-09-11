#!/usr/bin/env python3
"""Render one voice session's trace as a waterfall: where its time went.

Reads a session trace either from a file on this Host or through the Channel
Provider's read surface, and prints the session envelope followed by one
waterfall per turn.

The stage vocabulary is not invented here. Turn segments come from
``PROVIDER_LATENCY_SEGMENTS`` — the same 30 definitions the benchmark
dashboard uses — so a number in this output and a number on that dashboard mean
the same thing by construction rather than by convention.

    # A file the worker wrote
    ./.venv/bin/python scripts/report_session_trace.py \\
        ~/eidolon/data/channel/traces/2026-09-11/owner-1__companion-1__sess.ndjson

    # Or through the Provider, which is the interface every other client uses
    ./.venv/bin/python scripts/report_session_trace.py --session sess-abc
    ./.venv/bin/python scripts/report_session_trace.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from eidolon.livekit.agent.observability import PROVIDER_LATENCY_SEGMENTS
from eidolon.livekit.agent.observability.session_trace import (
    RECORD_EVENT,
    RECORD_SESSION_CLOSE,
    RECORD_SESSION_MARK,
    RECORD_SESSION_OPEN,
    RECORD_TURN_FINAL,
    SESSION_MARKS,
)

_DEFAULT_PROVIDER = "http://127.0.0.1:8767"
_BAR_WIDTH = 44


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def _from_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                # The writer appends while this reads; a torn last line is
                # expected and is not a reason to refuse the rest.
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _provider_get(base: str, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    token = os.environ.get("EIDOLON_CHANNEL_PROVIDER_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "EIDOLON_CHANNEL_PROVIDER_TOKEN is not set — the Provider's read "
            "surface is authenticated, same as its write surface."
        )
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - loopback
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"provider answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"provider unreachable at {base}: {exc.reason}") from exc


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_session(records: list[dict[str, Any]]) -> list[str]:
    opened = _first(records, RECORD_SESSION_OPEN)
    closed = _first(records, RECORD_SESSION_CLOSE)
    lines: list[str] = []
    if opened is None:
        return ["(no session_open record — is this a session trace?)"]

    anchor = opened.get("clock_anchor") or {}
    lines.append("=" * 78)
    lines.append(f"session   {opened.get('session_id')}")
    lines.append(
        f"owner     {opened.get('owner_id')}    companion {opened.get('companion_id')}"
    )
    lines.append(
        f"room      {opened.get('room_name')}    mode      {opened.get('interaction_mode')}"
    )
    lines.append(f"started   {anchor.get('iso')}")
    if closed is not None:
        lines.append(
            f"ended     reason={closed.get('reason')}  "
            f"duration={_ms(closed.get('duration_ms'))}  "
            f"records={closed.get('written_record_count')} "
            f"dropped={closed.get('dropped_record_count')}"
            + ("  TRUNCATED" if closed.get("truncated") else "")
        )
    else:
        lines.append("ended     (no close record — the session was still open, or the worker died)")
    lines.append("=" * 78)

    # Session setup, in the order a healthy session reaches it.
    marks = {
        str(record.get("mark")): record.get("since_open_ms")
        for record in records
        if record.get("record_kind") == RECORD_SESSION_MARK
    }
    lines.append("")
    lines.append("session setup")
    longest = max((float(v) for v in marks.values() if isinstance(v, (int, float))), default=0.0)
    previous = 0.0
    for name in SESSION_MARKS:
        at = marks.get(name)
        if not isinstance(at, (int, float)):
            lines.append(f"  {name:30s} {'—':>9}")
            continue
        step = float(at) - previous
        previous = float(at)
        lines.append(
            f"  {name:30s} {_ms(at):>9}  (+{_ms(step):>8})  {_bar(step, longest or 1.0)}"
        )
    missing = (closed or {}).get("missing_session_marks")
    if missing:
        lines.append(f"  never reached: {', '.join(missing)}")
    return lines


def _render_turns(records: list[dict[str, Any]]) -> list[str]:
    turns = [r for r in records if r.get("record_kind") == RECORD_TURN_FINAL]
    events_by_turn: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_kind") != RECORD_EVENT:
            continue
        turn_id = str(record.get("channel_turn_id") or "")
        if turn_id:
            events_by_turn.setdefault(turn_id, []).append(record)

    lines: list[str] = []
    if not turns:
        lines.append("")
        lines.append("no settled turns in this session")
        # Progress rows exist for turns a later flush would have settled; saying
        # so beats an empty section that reads as "nothing happened".
        pending = sum(1 for r in records if r.get("record_kind") == "turn_progress")
        if pending:
            lines.append(f"  ({pending} turn_progress row(s) — none reached a terminal)")
        return lines

    for index, turn in enumerate(turns, start=1):
        timestamps = turn.get("timestamps") or {}
        durations = _segment_durations(timestamps)
        longest = max((v for _, v in durations), default=1.0) or 1.0
        turn_id = str(turn.get("turn_id") or "")
        lines.append("")
        lines.append("-" * 78)
        lines.append(
            f"turn {index}  {turn_id}  reason={turn.get('record_reason')}  "
            f"marks={len(timestamps)}"
        )
        terminal = next(
            (
                event
                for event in reversed(events_by_turn.get(turn_id, []))
                if str(event.get("event_type", "")).startswith("channel.turn.")
                and event.get("status")
            ),
            None,
        )
        if terminal is not None:
            lines.append(
                f"          status={terminal.get('status')} "
                f"phase={terminal.get('phase')} "
                f"terminal_reason={terminal.get('terminal_reason')}"
            )
        lines.append("-" * 78)
        if not durations:
            lines.append("  (no segment had both of its marks — the turn ended early)")
            continue
        for label, value in durations:
            lines.append(f"  {label:52s} {_ms(value):>9}  {_bar(value, longest)}")
    return lines


def _segment_durations(timestamps: dict[str, Any]) -> list[tuple[str, float]]:
    """Only the segments this turn actually reached, in declaration order.

    A segment missing either mark is left out rather than shown as zero: a
    phase a turn never reached is not a phase that took no time.
    """

    out: list[tuple[str, float]] = []
    for segment in PROVIDER_LATENCY_SEGMENTS:
        start = timestamps.get(segment.start)
        end = timestamps.get(segment.end)
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end < start:
            continue
        out.append((f"[{segment.stage}] {segment.name}", (float(end) - float(start)) * 1000))
    return out


def _first(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("record_kind") == kind), None)


def _bar(value: float, longest: float) -> str:
    filled = int(round(_BAR_WIDTH * min(1.0, max(0.0, value / longest)))) if longest else 0
    return "█" * filled


def _ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.1f}ms"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", nargs="?", help="path to a session trace .ndjson")
    parser.add_argument("--session", help="session id to fetch through the Provider")
    parser.add_argument("--list", action="store_true", help="list recorded sessions")
    parser.add_argument("--owner-id", default="", help="filter the listing by Owner")
    parser.add_argument("--limit", type=int, default=20, help="listing size (default 20)")
    parser.add_argument(
        "--provider",
        default=os.environ.get("EIDOLON_CHANNEL_PROVIDER_URL", _DEFAULT_PROVIDER),
        help=f"Provider base URL (default {_DEFAULT_PROVIDER})",
    )
    parser.add_argument("--json", action="store_true", help="emit the records instead of a report")
    args = parser.parse_args()

    if args.list:
        params = {"limit": str(args.limit)}
        if args.owner_id:
            params["owner_id"] = args.owner_id
        body = _provider_get(args.provider, "/v1/session-traces", params)
        if args.json:
            print(json.dumps(body, ensure_ascii=False, indent=2))
            return 0
        sessions = body.get("sessions") or []
        if not body.get("recording"):
            print("the Provider reports no trace directory — is session_trace_path set?")
        print(f"{'started':33s} {'session':26s} {'status':8s} {'duration':>10s}  reason")
        for summary in sessions:
            print(
                f"{str(summary.get('started_at') or '—'):33s} "
                f"{str(summary.get('session_id') or '—'):26s} "
                f"{str(summary.get('status') or '—'):8s} "
                f"{_ms(summary.get('duration_ms')):>10s}  {summary.get('reason') or '—'}"
            )
        return 0

    if args.trace:
        path = Path(args.trace).expanduser()
        if not path.is_file():
            raise SystemExit(f"no such trace file: {path}")
        records = _from_file(path)
    elif args.session:
        body = _provider_get(args.provider, f"/v1/session-traces/{args.session}")
        records = [r for r in (body.get("records") or []) if isinstance(r, dict)]
    else:
        parser.error("pass a trace file, --session <id>, or --list")
        return 2

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0

    lines = _render_session(records) + _render_turns(records)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
