"""Offline reference extraction from explicit, consented evaluation WAV files.

No room connection, template writes, VAD splicing, or production policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import wave


PREPROCESSING = "reference-contiguous-pcm16-v1"


def extract_dataset(manifest, *, root, extractor, model):
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported capture manifest")
    root = root.resolve()
    rows = []
    for capture in manifest["samples"]:
        path = (root / capture["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("audio path escapes dataset root")
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("evaluation input must be mono PCM16 at 16 kHz")
            frames = audio.getnframes()
            if not 0 < frames <= 16000 * 30:
                raise ValueError("evaluation input must be nonempty and at most 30 seconds")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("truncated WAV")
        row = {k: capture[k] for k in ("id", "speaker", "device", "capture_session", "partition")}
        # Hash canonical PCM, so different WAV headers cannot hide duplicates.
        row.update(audio_sha256=hashlib.sha256(pcm).hexdigest(), duration_ms=frames * 1000 / 16000)
        import numpy as np

        values = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
        row["quality"] = {
            "rms": float(np.sqrt(np.mean(values * values))),
            "peak": float(np.max(np.abs(values))),
            "dc_offset": float(np.mean(values)),
            "clipping_fraction": float(np.mean(np.abs(values) >= 0.995)),
            "speech_ms": None,  # This reference path does not estimate speech duration.
        }
        started = time.perf_counter()
        if not np.any(values):
            row.update(status="insufficient", reason="digital_silence")
        elif frames < 24000:
            row.update(status="insufficient", reason="audio_too_short")
        else:
            try:
                vector = [float(x) for x in extractor.extract_embedding(pcm, 16000)]
                if not vector or not all(math.isfinite(x) for x in vector) or not math.hypot(*vector):
                    raise ValueError("invalid embedding")
                row.update(status="ok", vector=vector)
            except Exception as exc:
                row.update(status="unavailable", reason=type(exc).__name__)
        row["extraction_ms"] = round((time.perf_counter() - started) * 1000, 3)
        rows.append(row)
    return dict(schema_version=1, model=model, samples=rows, trials=manifest["trials"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from eidolon.livekit.plugins.speaker_verification.providers.modelscope_campplus import (
        ModelScopeCampPlusSpeakerVerificationProvider,
    )
    metadata = json.loads((args.model_dir / "manifest.json").read_text())
    for artifact in metadata["artifacts"]:
        path = (args.model_dir / artifact["path"]).resolve()
        if not path.is_relative_to(args.model_dir.resolve()):
            raise ValueError("artifact path escapes model directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("model artifact checksum mismatch")
    weights = args.model_dir / "campplus_cn_common.bin"
    model = dict(id=metadata["model_id"], weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                 preprocessing=PREPROCESSING)
    engine = ModelScopeCampPlusSpeakerVerificationProvider(model_dir=args.model_dir)
    result = extract_dataset(json.loads(args.manifest.read_text()),
                             root=args.manifest.resolve().parent, extractor=engine, model=model)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    # Embeddings stay in the explicitly requested private file, not console logs.
    print(f"Extracted {len(result['samples'])} records to {args.output}. No policy changed.")


if __name__ == "__main__":
    main()
