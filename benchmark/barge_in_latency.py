#!/usr/bin/env python3
"""Decompose barge-in latency from the turn-timeline JSONL dump.

Measurement tool for the near-term slice (2026-07): the web dogfood showed a
substantive Chinese barge-in ("那你现在能帮我做什么。") with
``speech-start-to-cancel`` ≈ 1984ms. Before choosing a fix we need to know what
that ~2s actually *is*. These server marks describe decisions and output
admission, not when a receiver or physical speaker becomes quiet. Already
queued audio can continue playing after ``interrupt_started_at``.

Everything needed is already recorded per turn in the timeline snapshot written
to ``observability.timeline_debug_path`` (default
``~/eidolon/logs/channel/turn-timeline.jsonl``). This reader computes the
decomposition straight from the raw ``timestamps`` marks, so it also works on
snapshots dumped before the convenience durations were added.

Usage
-----
    python -m benchmark.barge_in_latency                 # default path, cancel turns
    python -m benchmark.barge_in_latency --last 5        # last 5 matching turns
    python -m benchmark.barge_in_latency --all           # every turn (not just cancels)
    python -m benchmark.barge_in_latency PATH.jsonl

The key rows:
  speech_start -> duck_requested server starts ducking (not audible silence)
  speech_start -> cancel          the ~1984ms headline
  cancel       -> commit          gap from cancel to committing the user turn
  commit       -> first_delta     brain latency
  first_delta  -> first_audio     TTS latency
  speech_start -> first_audio     server first-audio mark (not client playout)

``speech_started_at`` is the server speech callback, not microphone onset.
Receiver attenuation and device playout need separate audio measurements;
this report cannot establish an audible-overlap latency bound.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any

DEFAULT_PATH = "~/eidolon/logs/channel/turn-timeline.jsonl"

# (label, from_mark, to_mark). First present brain-delta mark wins via _pick.
_STEPS: tuple[tuple[str, str, str], ...] = (
    ("speech_start -> duck_requested", "speech_started_at", "interrupt_started_at"),
    ("speech_start -> cancel", "speech_started_at", "interrupt_cancel_resolved_at"),
    ("speech_start -> speech_stop", "speech_started_at", "speech_stopped_at"),
    ("speech_stop  -> commit", "speech_stopped_at", "turn_committed_at"),
    ("cancel       -> commit", "interrupt_cancel_resolved_at", "turn_committed_at"),
    ("commit       -> first_delta", "turn_committed_at", "@brain_first_delta"),
    ("first_delta  -> first_audio", "@brain_first_delta", "tts_first_audio_at"),
    ("commit       -> first_audio", "turn_committed_at", "tts_first_audio_at"),
    ("speech_start -> first_audio", "speech_started_at", "tts_first_audio_at"),
)

# Rolled-back (false-interrupt) barge-ins resolve here instead of cancel.
_ROLLBACK_MARK = "interrupt_rollback_resolved_at"


def receiver_attenuation_ms(
    frames: list[tuple[float, float]], *, onset_ms: float,
) -> float | None:
    """Observe a >=12 dB RMS drop lasting 100 ms in received 20 ms frames.

    Reference the 120 ms preceding microphone publication onset. Missing,
    quiet, or stale baseline audio and packet gaps are not evidence of ducking.
    This is an acoustic observation, not causal proof: natural TTS pauses can
    also qualify. Use continuous test audio to validate output-control latency;
    never substitute this receiver metric for physical speaker measurements.
    """
    before = [(at, rms) for at, rms in frames if onset_ms - 120 <= at < onset_ms]
    if len(before) < 4 or onset_ms - before[-1][0] > 40:
        return None
    baseline = statistics.median(rms for _, rms in before)
    if baseline < 120:
        return None
    threshold = baseline * 10 ** (-12 / 20)
    quiet_at = previous = None
    for at, rms in frames:
        if at < onset_ms:
            continue
        if previous is not None and at - previous > 60:
            quiet_at = None
        previous = at
        if rms > threshold:
            quiet_at = None
        elif quiet_at is None:
            quiet_at = at
        elif at - quiet_at >= 100:
            return quiet_at - onset_ms
    return None


def _pick(ts: dict[str, float], name: str) -> float | None:
    """Resolve a mark name, expanding @brain_first_delta to the first present."""
    if name == "@brain_first_delta":
        for candidate in ("brain_first_delta_at", "llm_first_delta_at"):
            if candidate in ts:
                return ts[candidate]
        return None
    return ts.get(name)


def _ms(ts: dict[str, float], a: str, b: str) -> float | None:
    ta, tb = _pick(ts, a), _pick(ts, b)
    if ta is None or tb is None:
        return None
    delta = (tb - ta) * 1000.0
    return delta if delta >= 0 else None


def _is_barge_in(turn: dict[str, Any]) -> bool:
    ts = turn.get("timestamps", {})
    return "interrupt_cancel_resolved_at" in ts or "interrupt_started_at" in ts


def _transcript(turn: dict[str, Any]) -> str:
    attrs = turn.get("attrs", {})
    for key in ("committed_user_text", "user_transcript", "final_transcript"):
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    dec = attrs.get("decision") or attrs.get("last_decision") or {}
    if isinstance(dec, dict):
        prev = dec.get("transcript_preview")
        if isinstance(prev, str) and prev.strip():
            return prev.strip()
    return ""


def _fmt(v: float | None) -> str:
    return f"{v:8.1f}ms" if v is not None else "       —"


def _print_turn(turn: dict[str, Any]) -> None:
    ts = turn.get("timestamps", {})
    attrs = turn.get("attrs", {})
    resolved = "cancel" if "interrupt_cancel_resolved_at" in ts else (
        "rollback" if _ROLLBACK_MARK in ts else "?"
    )
    reason = attrs.get("cancel_reason") or attrs.get("interrupt_resolution_reason") or ""
    text = _transcript(turn)
    print(f"turn {turn.get('turn_id', '?')}  resolved={resolved}"
          f"{('  reason=' + str(reason)) if reason else ''}")
    if text:
        print(f"  transcript: {text[:60]!r}")
    for label, a, b in _STEPS:
        print(f"  {label:32s} {_fmt(_ms(ts, a, b))}")
    # rolled-back barge-ins: show the resume latency instead of cancel
    if resolved == "rollback":
        print(f"  {'speech_start -> rollback':32s} "
              f"{_fmt(_ms(ts, 'speech_started_at', _ROLLBACK_MARK))}")
    print()


def _print_aggregate(turns: list[dict[str, Any]]) -> None:
    if len(turns) < 2:
        return
    print(f"=== aggregate over {len(turns)} turns (p50 / p95) ===")
    for label, a, b in _STEPS:
        vals = sorted(v for v in (_ms(t.get("timestamps", {}), a, b) for t in turns)
                      if v is not None)
        if not vals:
            print(f"  {label:32s}        — (n=0)")
            continue
        p50 = statistics.median(vals)
        p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
        print(f"  {label:32s} p50={p50:8.1f}ms  p95={p95:8.1f}ms  (n={len(vals)})")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=DEFAULT_PATH,
                    help=f"timeline JSONL (default: {DEFAULT_PATH})")
    ap.add_argument("--last", type=int, default=0,
                    help="only the last N matching turns (0 = all)")
    ap.add_argument("--all", action="store_true",
                    help="include non-barge-in turns too")
    args = ap.parse_args()

    path = os.path.expanduser(args.path)
    if not os.path.exists(path):
        raise SystemExit(
            f"timeline dump not found: {path}\n"
            "Enable it via observability.timeline_debug_path in config/settings.yaml, "
            "then run a web-client full-duplex dogfood barge-in."
        )

    turns: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.all or _is_barge_in(turn):
                turns.append(turn)

    if not turns:
        raise SystemExit("no matching turns in dump (barge-in turns have "
                         "interrupt_started_at / interrupt_cancel_resolved_at).")

    if args.last > 0:
        turns = turns[-args.last:]

    print(f"# {path}")
    print(f"# {len(turns)} turn(s)\n")
    for turn in turns:
        _print_turn(turn)
    _print_aggregate(turns)


if __name__ == "__main__":
    main()
