"""Offline reference extraction from explicit, consented evaluation WAV files.

No room connection, template writes, VAD splicing, or production policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import wave


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
            "speech_ms": None,  # Filled only after the shared quality path succeeds.
        }
        started = time.perf_counter()
        if not np.any(values):
            row.update(status="insufficient", reason="digital_silence")
        elif frames < 24000:
            row.update(status="insufficient", reason="audio_too_short")
        else:
            try:
                sample = extractor.extract(pcm)
                row["quality"]["speech_ms"] = sample.speech_ms
                row.update(status="ok", vector=sample.vector)
            except Exception as exc:
                from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
                    InvalidSpeakerAudio,
                )
                if isinstance(exc, InvalidSpeakerAudio):
                    row.update(status="insufficient", reason=str(exc))
                else:
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
    from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
        ReferenceSpeakerExtractor,
    )
    engine = ReferenceSpeakerExtractor(args.model_dir)
    engine.warm_up()
    model_id, digest, preprocessing = engine.model_identity
    model = dict(id=model_id, weights_sha256=digest, preprocessing=preprocessing)
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
