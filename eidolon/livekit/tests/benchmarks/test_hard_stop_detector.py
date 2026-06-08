"""Tests for the offline hard-stop detector comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eidolon.livekit.benchmarks.audio_assets import write_wav
from eidolon.livekit.benchmarks.hard_stop_detector import (
    load_livekit_stt_actionable_metrics,
    run_template_detector_comparison,
    summarize_metric,
)


def test_template_detector_calibrates_threshold_and_counts_errors(tmp_path: Path) -> None:
    sample_rate = 16_000
    hard_template = _tone_pcm(freq=440, sample_rate=sample_rate)
    negative = _tone_pcm(freq=880, sample_rate=sample_rate)
    write_wav(tmp_path / "hard_stop_dont.wav", hard_template, sample_rate=sample_rate)
    write_wav(tmp_path / "normal.wav", negative, sample_rate=sample_rate)

    comparison = run_template_detector_comparison(
        clips={
            "hard_stop_dont": {
                "path": str(tmp_path / "hard_stop_dont.wav"),
                "intent": "hard_stop",
                "sample_rate": sample_rate,
            },
            "normal": {
                "path": str(tmp_path / "normal.wav"),
                "intent": "normal",
                "sample_rate": sample_rate,
            },
        },
        template_ids=("hard_stop_dont",),
        min_window_ms=180,
        max_window_ms=400,
        step_ms=20,
    )

    assert comparison.positive_count == 1
    assert comparison.negative_count == 1
    assert comparison.true_positive_count == 1
    assert comparison.false_positive_count == 0
    assert comparison.false_negative_count == 0
    assert comparison.precision == 1.0
    assert comparison.recall == 1.0
    assert all(
        score.detection_ms is not None
        for score in comparison.scores
        if score.expected_positive
    )


def test_load_livekit_stt_actionable_metrics_reads_interrupt_rows(tmp_path: Path) -> None:
    results = tmp_path / "repeat-00" / "livekit_room_results.jsonl"
    results.parent.mkdir(parents=True)
    rows = [
        {
            "case_id": "flow_normal_question_then_hard_stop_001",
            "passed": True,
            "metrics": {
                "timeline_interrupt_speech_to_first_transcript_ms": 210.0,
                "timeline_stt_speech_to_actionable_transcript_ms": 540.0,
                "timeline_stt_first_transcript_to_actionable_transcript_ms": 330.0,
                "timeline_interrupt_actionable_transcript_to_resolved_ms": 1.0,
                "timeline_vad_start_to_interrupt_resolved": 541.0,
            },
        },
        {
            "case_id": "flow_normal_question_then_agent_reply_001",
            "passed": True,
            "metrics": {},
        },
    ]
    results.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    loaded = load_livekit_stt_actionable_metrics(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["repeat"] == "repeat-00"
    assert loaded[0]["case_id"] == "flow_normal_question_then_hard_stop_001"
    assert loaded[0]["speech_to_actionable_ms"] == 540.0
    assert summarize_metric([row["speech_to_actionable_ms"] for row in loaded]) == {
        "count": 1,
        "p50": 540.0,
        "p95": 540.0,
        "max": 540.0,
    }


def _tone_pcm(*, freq: float, sample_rate: int, duration_s: float = 0.8) -> bytes:
    t = np.arange(round(sample_rate * duration_s), dtype=np.float32) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    return np.clip(audio * 32767, -32768, 32767).astype("<i2").tobytes()
