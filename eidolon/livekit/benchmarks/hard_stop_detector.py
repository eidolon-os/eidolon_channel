"""Offline hard-stop detector experiments for benchmark audio assets."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClipScore:
    clip_id: str
    intent: str
    expected_positive: bool
    detected: bool
    detection_ms: int | None
    best_score: float
    matched_template: str


@dataclass(frozen=True)
class DetectorComparison:
    threshold: float
    positive_count: int
    negative_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    scores: list[ClipScore]

    @property
    def precision(self) -> float | None:
        predicted = self.true_positive_count + self.false_positive_count
        return self.true_positive_count / predicted if predicted else None

    @property
    def recall(self) -> float | None:
        return (
            self.true_positive_count / self.positive_count
            if self.positive_count
            else None
        )


def run_template_detector_comparison(
    *,
    clips: dict[str, dict[str, Any]],
    template_ids: tuple[str, ...] = ("hard_stop_dont", "hard_stop_stop"),
    positive_intent: str = "hard_stop",
    min_window_ms: int = 180,
    max_window_ms: int = 900,
    step_ms: int = 20,
    threshold_margin: float = 0.03,
) -> DetectorComparison:
    """Run a conservative template-similarity detector over benchmark WAVs.

    This is intentionally an offline experiment, not a production detector. It
    estimates whether a tiny local hard-stop path could beat STT actionability
    on the existing benchmark audio, while checking false positives on normal,
    topic-switch, correction, backchannel, and noise clips.
    """

    loaded = {
        clip_id: _load_wav_float(Path(item["path"]))
        for clip_id, item in clips.items()
        if Path(item["path"]).exists()
    }
    templates = {
        clip_id: loaded[clip_id]
        for clip_id in template_ids
        if clip_id in loaded
    }
    if not templates:
        raise ValueError("no hard-stop templates were found")

    raw_scores: list[tuple[str, dict[str, Any], float, str]] = []
    for clip_id, item in clips.items():
        audio = loaded.get(clip_id)
        if audio is None:
            continue
        _, best_score, matched = _score_streaming_prefix(
            audio,
            templates=templates,
            sample_rate=int(item.get("sample_rate") or 16_000),
            threshold=None,
            min_window_ms=min_window_ms,
            max_window_ms=max_window_ms,
            step_ms=step_ms,
        )
        raw_scores.append((clip_id, item, best_score, matched))

    negative_best = [
        score
        for _, item, score, _ in raw_scores
        if str(item.get("intent") or "") != positive_intent
    ]
    threshold = (max(negative_best) + threshold_margin) if negative_best else 0.75
    threshold = min(max(threshold, 0.50), 0.995)

    scores: list[ClipScore] = []
    for clip_id, item, _, _ in raw_scores:
        audio = loaded[clip_id]
        detection_ms, best_score, matched = _score_streaming_prefix(
            audio,
            templates=templates,
            sample_rate=int(item.get("sample_rate") or 16_000),
            threshold=threshold,
            min_window_ms=min_window_ms,
            max_window_ms=max_window_ms,
            step_ms=step_ms,
        )
        expected_positive = str(item.get("intent") or "") == positive_intent
        scores.append(
            ClipScore(
                clip_id=clip_id,
                intent=str(item.get("intent") or ""),
                expected_positive=expected_positive,
                detected=detection_ms is not None,
                detection_ms=detection_ms,
                best_score=best_score,
                matched_template=matched,
            )
        )

    positive_count = sum(1 for score in scores if score.expected_positive)
    negative_count = len(scores) - positive_count
    true_positive_count = sum(
        1 for score in scores if score.expected_positive and score.detected
    )
    false_positive_count = sum(
        1 for score in scores if not score.expected_positive and score.detected
    )
    false_negative_count = sum(
        1 for score in scores if score.expected_positive and not score.detected
    )
    return DetectorComparison(
        threshold=threshold,
        positive_count=positive_count,
        negative_count=negative_count,
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        scores=scores,
    )


def load_livekit_stt_actionable_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Load STT actionability rows from a livekit_room repeat run directory."""

    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("livekit_room_results.jsonl")):
        repeat = path.parent.name if path.parent.name.startswith("repeat-") else ""
        with path.open(encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                item = json.loads(raw)
                case_id = str(item.get("case_id") or "")
                metrics = (
                    item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
                )
                if not case_id or not _is_interrupt_case(case_id):
                    continue
                rows.append(
                    {
                        "repeat": repeat,
                        "case_id": case_id,
                        "passed": bool(item.get("passed")),
                        "errors": item.get("errors") or [],
                        "speech_to_first_transcript_ms": metrics.get(
                            "timeline_interrupt_speech_to_first_transcript_ms"
                        ),
                        "speech_to_actionable_ms": metrics.get(
                            "timeline_stt_speech_to_actionable_transcript_ms"
                        ),
                        "first_to_actionable_ms": metrics.get(
                            "timeline_stt_first_transcript_to_actionable_transcript_ms"
                        ),
                        "actionable_to_resolved_ms": metrics.get(
                            "timeline_interrupt_actionable_transcript_to_resolved_ms"
                        ),
                        "resolved_ms": metrics.get(
                            "timeline_vad_start_to_interrupt_resolved"
                        ),
                    }
                )
    return rows


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not clean:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "max": max(clean),
    }


def _score_streaming_prefix(
    audio: np.ndarray,
    *,
    templates: dict[str, np.ndarray],
    sample_rate: int,
    threshold: float | None,
    min_window_ms: int,
    max_window_ms: int,
    step_ms: int,
) -> tuple[int | None, float, str]:
    best_score = -1.0
    best_template = ""
    detection_ms: int | None = None
    max_ms = min(max_window_ms, math.ceil(len(audio) / sample_rate * 1000))
    for window_ms in range(min_window_ms, max_ms + 1, step_ms):
        samples = max(1, round(sample_rate * window_ms / 1000))
        clip_prefix = audio[:samples]
        for template_id, template in templates.items():
            template_prefix = template[: min(samples, len(template))]
            score = _prefix_similarity(clip_prefix, template_prefix, sample_rate)
            if score > best_score:
                best_score = score
                best_template = template_id
            if threshold is not None and score >= threshold:
                return window_ms, best_score, template_id
    return detection_ms, best_score, best_template


def _prefix_similarity(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    a_feat = _features(a, sample_rate)
    b_feat = _features(b, sample_rate)
    count = min(len(a_feat), len(b_feat))
    if count == 0:
        return 0.0
    a_vec = a_feat[:count].reshape(-1)
    b_vec = b_feat[:count].reshape(-1)
    a_vec = a_vec - float(np.mean(a_vec))
    b_vec = b_vec - float(np.mean(b_vec))
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def _features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, round(sample_rate * 0.025))
    hop = max(1, round(sample_rate * 0.010))
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    window = np.hanning(frame).astype(np.float32)
    rows: list[np.ndarray] = []
    for start in range(0, max(1, len(audio) - frame + 1), hop):
        chunk = audio[start : start + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        framed = chunk * window
        spectrum = np.abs(np.fft.rfft(framed)) ** 2
        freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
        bands = []
        for lo, hi in (
            (0, 300),
            (300, 700),
            (700, 1200),
            (1200, 2000),
            (2000, 3500),
            (3500, 5500),
            (5500, 8000),
        ):
            mask = (freqs >= lo) & (freqs < hi)
            value = np.mean(spectrum[mask]) if np.any(mask) else 0.0
            bands.append(float(np.log1p(value)))
        energy = float(np.log1p(np.mean(framed * framed)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(framed).astype(np.int8)))))
        rows.append(np.array([energy, zcr, *bands], dtype=np.float32))
    feats = np.vstack(rows)
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-6
    return (feats - mean) / std


def _load_wav_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV: {path}")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def _is_interrupt_case(case_id: str) -> bool:
    return any(token in case_id for token in ("hard_stop", "topic_switch", "correction"))


def _percentile(values: list[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac
