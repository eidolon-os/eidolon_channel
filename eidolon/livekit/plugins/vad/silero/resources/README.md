# Silero VAD (16 kHz, opset 15)

| File | Upstream path | SHA-256 |
|---|---|---|
| `silero_vad_16k_op15.onnx` | `src/silero_vad/data/silero_vad_16k_op15.onnx` | `7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49` |
| `LICENSE` | `LICENSE` (MIT, Silero Team) | `2e63e9a38b6e8fc0c7bc37ce174caca1862870856c6daf5697cfb785e925520b` |

Source: https://github.com/snakers4/silero-vad at revision `7e30209a3e901f9842f81b225f3e93d8199902b1`,
the exact model and revision the official Reachy Mini Conversation App (1.1.1,
`audio/local_vad.py`) downloads and verifies. Keeping the same bytes lets Host
VAD decisions be compared one-to-one with the native app.

The file is stored in plain git (1.3 MB), not LFS: the loader verifies the
SHA-256 above, and a release snapshot must never load an LFS pointer.
