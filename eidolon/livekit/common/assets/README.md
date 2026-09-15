# Welcome audio

`soft-ready.wav` is the original, procedurally synthesized cue selected for this
project on 2026-09-15. It contains no recording or third-party sample.

- Mono, signed 16-bit PCM WAV, 24,000 Hz; 440 ms; 21,164 bytes.
- Two overlapping sine-based tones: E5 (659.255 Hz) and A5 (880 Hz).
- Soft attack/release, exponential decay, a quiet second harmonic; peak −12 dBFS.
- Distributed under the repository license and included as Python package data.

Use `behavior.welcome_message: {audio: builtin:soft-ready}` to select it.
Runtime playback resamples and caches PCM at the configured output rate.
