# 3D-Speaker CAM++ Model Validation

Date: 2026-06-10

Model bundle:

```text
eidolon/livekit/plugins/speaker_verification/resources/3dspeaker/campplus_zh_16k_common/
```

Validated artifacts:

- `campplus_cn_common.bin`
- `config.yaml`
- `configuration.json`

## Commands

Bundle validation using the channel virtualenv:

```bash
./.venv/bin/python scripts/validate_3dspeaker_model.py
```

Pipeline validation was run in a temporary venv with ModelScope audio dependencies:

```bash
/private/tmp/eidolon-3dspeaker-validate-venv/bin/python \
  scripts/validate_3dspeaker_model.py \
  --run-pipeline \
  --wav-a /private/tmp/eidolon-3dspeaker-campplus-inspect/examples/speaker1_a_cn_16k.wav \
  --wav-b /private/tmp/eidolon-3dspeaker-campplus-inspect/examples/speaker1_b_cn_16k.wav \
  --wav-c /private/tmp/eidolon-3dspeaker-campplus-inspect/examples/speaker2_a_cn_16k.wav
```

## Result

The local bundle loaded successfully through the official ModelScope
speaker-verification pipeline.

```json
{
  "same_speaker": {
    "score": 0.6936,
    "text": "yes",
    "latency_ms": 84
  },
  "different_speaker": {
    "score": -0.08418,
    "text": "no",
    "latency_ms": 70
  },
  "load_ms": 51220,
  "device": "cpu"
}
```

Interpretation:

- `speaker1_a_cn_16k.wav` vs `speaker1_b_cn_16k.wav` correctly returned same speaker.
- `speaker1_a_cn_16k.wav` vs `speaker2_a_cn_16k.wav` correctly returned different speaker.
- The official threshold in `configuration.json` is `0.31`.
- First load on CPU is slow because ModelScope/Torch imports and initializes the pipeline.
- Warm verification latency on the official examples was under 100 ms on this machine.

## Runtime Note

The channel project now declares `modelscope[audio]` as a runtime dependency so
the PyTorch baseline provider can be wired directly. The realtime path should
still target an ONNX provider after embedding and threshold parity checks; once
that lands, channel can drop the PyTorch/ModelScope dependency while admin may
keep it for enrollment.
