#!/usr/bin/env python3
"""Validate the bundled 3D-Speaker CAM++ model.

The default mode checks the local model bundle only: manifest shape, artifact
existence, file sizes, SHA256 hashes, and basic ModelScope config links.

Pass ``--run-pipeline`` with three 16 kHz mono WAV files to run the official
ModelScope speaker-verification pipeline:

    speaker1_a vs speaker1_b => expected same speaker
    speaker1_a vs speaker2_a => expected different speaker

This script intentionally keeps ModelScope/Torch imports inside the pipeline
path so bundle validation remains lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import yaml

from eidolon.livekit.plugins.speaker_verification import default_campplus_model_dir


DEFAULT_MODEL_DIR = default_campplus_model_dir()


class ValidationError(RuntimeError):
    """Raised when the local model bundle is malformed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return data


def _validate_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise ValidationError(
            f"{path} must be 16 kHz mono 16-bit PCM WAV; got "
            f"channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}"
        )
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_ms": int(frames * 1000 / sample_rate),
    }


def validate_bundle(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "manifest.json"
    config_path = model_dir / "configuration.json"
    yaml_path = model_dir / "config.yaml"

    manifest = _load_json(manifest_path)
    configuration = _load_json(config_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    if manifest.get("provider") != "3d_speaker":
        raise ValidationError("manifest provider must be 3d_speaker")
    if manifest.get("sample_rate") != 16000:
        raise ValidationError("manifest sample_rate must be 16000")
    if configuration.get("model_file") != "campplus_cn_common.bin":
        raise ValidationError("configuration.json must point to campplus_cn_common.bin")
    if configuration.get("model_config") != "config.yaml":
        raise ValidationError("configuration.json must point to config.yaml")
    if model_config.get("model") != "CAMPPlus":
        raise ValidationError("config.yaml model must be CAMPPlus")
    if model_config.get("frontend_conf", {}).get("fs") != 16000:
        raise ValidationError("config.yaml frontend_conf.fs must be 16000")

    artifacts = []
    for artifact in manifest.get("artifacts", []):
        rel_path = artifact.get("path")
        if not rel_path:
            raise ValidationError("manifest artifact is missing path")
        artifact_path = model_dir / rel_path
        if not artifact_path.is_file():
            raise ValidationError(f"missing artifact: {artifact_path}")
        size_bytes = artifact_path.stat().st_size
        sha256 = _sha256(artifact_path)
        if artifact.get("size_bytes") != size_bytes:
            raise ValidationError(
                f"{rel_path} size mismatch: manifest={artifact.get('size_bytes')} actual={size_bytes}"
            )
        if artifact.get("sha256") != sha256:
            raise ValidationError(
                f"{rel_path} sha256 mismatch: manifest={artifact.get('sha256')} actual={sha256}"
            )
        artifacts.append(
            {
                "path": rel_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "kind": artifact.get("kind"),
            }
        )

    return {
        "model_dir": str(model_dir),
        "model_id": manifest.get("model_id"),
        "model_alias": manifest.get("model_alias"),
        "configuration": {
            "framework": configuration.get("framework"),
            "task": configuration.get("task"),
            "threshold": configuration.get("model", {}).get("yesOrno_thr"),
        },
        "model_config": {
            "model": model_config.get("model"),
            "embedding_size": model_config.get("model_conf", {}).get("embedding_size"),
            "sample_rate": model_config.get("frontend_conf", {}).get("fs"),
        },
        "artifacts": artifacts,
    }


def _prediction_value(result: Any) -> bool | None:
    if isinstance(result, dict):
        for key in ("text", "output", "prediction", "pred", "label"):
            value = result.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"yes", "true", "same", "1"}:
                    return True
                if normalized in {"no", "false", "different", "0"}:
                    return False
        outputs = result.get("outputs")
        if isinstance(outputs, dict):
            return _prediction_value(outputs)
    return None


def run_pipeline(model_dir: Path, wav_a: Path, wav_b: Path, wav_c: Path) -> dict[str, Any]:
    for wav_path in (wav_a, wav_b, wav_c):
        if not wav_path.is_file():
            raise ValidationError(f"missing wav: {wav_path}")
        _validate_wav(wav_path)

    started = time.perf_counter()
    from modelscope.pipelines import pipeline

    sv_pipeline = pipeline(task="speaker-verification", model=str(model_dir))
    load_ms = int((time.perf_counter() - started) * 1000)

    same_started = time.perf_counter()
    same_result = sv_pipeline([str(wav_a), str(wav_b)])
    same_ms = int((time.perf_counter() - same_started) * 1000)

    diff_started = time.perf_counter()
    diff_result = sv_pipeline([str(wav_a), str(wav_c)])
    diff_ms = int((time.perf_counter() - diff_started) * 1000)

    same_prediction = _prediction_value(same_result)
    diff_prediction = _prediction_value(diff_result)
    if same_prediction is not True:
        raise ValidationError(f"same-speaker check failed: {same_result!r}")
    if diff_prediction is not False:
        raise ValidationError(f"different-speaker check failed: {diff_result!r}")

    return {
        "load_ms": load_ms,
        "same_speaker": {
            "wav_a": str(wav_a),
            "wav_b": str(wav_b),
            "latency_ms": same_ms,
            "result": same_result,
        },
        "different_speaker": {
            "wav_a": str(wav_a),
            "wav_b": str(wav_c),
            "latency_ms": diff_ms,
            "result": diff_result,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--wav-a", type=Path, help="Speaker 1 enrollment/reference WAV")
    parser.add_argument("--wav-b", type=Path, help="Speaker 1 comparison WAV")
    parser.add_argument("--wav-c", type=Path, help="Different speaker comparison WAV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    try:
        report: dict[str, Any] = {"bundle": validate_bundle(model_dir)}
        if args.run_pipeline:
            missing = [
                name
                for name in ("wav_a", "wav_b", "wav_c")
                if getattr(args, name.replace("-", "_"), None) is None
            ]
            if missing:
                raise ValidationError(f"--run-pipeline requires: {', '.join('--' + m for m in missing)}")
            report["pipeline"] = run_pipeline(
                model_dir=model_dir,
                wav_a=args.wav_a.resolve(),
                wav_b=args.wav_b.resolve(),
                wav_c=args.wav_c.resolve(),
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
