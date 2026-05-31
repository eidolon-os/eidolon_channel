# Voice Benchmark Report: component-smoke-2

- runner: `component`
- profile: `real_components:vad=firered_pvad,stt=bailian,tts=bailian,eot=balanced_semantic`
- git_sha: `43f0302`
- pass_rate: `20/20`

## Metric Summary

| metric | p50 | p95 | max | count |
| --- | ---: | ---: | ---: | ---: |
| elapsed_ms | 344.5 | 4235.9 | 4253.0 | 20 |
| eot_elapsed_ms | 3.0 | 24.0 | 29.0 | 5 |
| eot_expected_cancel | 1.0 | 1.0 | 1.0 | 5 |
| eot_interrupt_count | 1.0 | 1.0 | 1.0 | 5 |
| eot_max_score | 0.0 | 0.7 | 0.9 | 5 |
| eot_step_count | 1.0 | 1.8 | 2.0 | 5 |
| stt_elapsed_ms | 3702.0 | 4249.4 | 4253.0 | 5 |
| stt_exact_match | 1.0 | 1.0 | 1.0 | 5 |
| stt_expected_chars | 8.0 | 11.6 | 12.0 | 5 |
| stt_nonempty | 1.0 | 1.0 | 1.0 | 5 |
| stt_output_chars | 8.0 | 11.8 | 12.0 | 5 |
| tts_audio_bytes | 74880.0 | 98432.0 | 99200.0 | 5 |
| tts_elapsed_ms | 1236.0 | 1713.2 | 1754.0 | 5 |
| tts_first_sample_rate | 16000.0 | 16000.0 | 16000.0 | 5 |
| tts_frame_count | 17.0 | 20.0 | 20.0 | 5 |
| tts_nonempty_audio | 1.0 | 1.0 | 1.0 | 5 |
| vad_detected_speech | 1.0 | 1.0 | 1.0 | 5 |
| vad_elapsed_ms | 127.0 | 174.6 | 180.0 | 5 |
| vad_end_count | 0.0 | 1.0 | 1.0 | 5 |
| vad_event_count | 235.0 | 308.8 | 311.0 | 5 |
| vad_inference_count | 234.0 | 307.6 | 310.0 | 5 |
| vad_max_probability | 1.0 | 1.0 | 1.0 | 5 |
| vad_start_count | 1.0 | 1.0 | 1.0 | 5 |

## Cases

### normal_single_turn_001:vad - PASS

```json
{
  "vad_elapsed_ms": 153,
  "vad_event_count": 300,
  "vad_start_count": 1,
  "vad_end_count": 1,
  "vad_inference_count": 298,
  "vad_max_probability": 0.9973984451194036,
  "vad_detected_speech": true,
  "elapsed_ms": 153
}
```

### normal_single_turn_001:stt - PASS

```json
{
  "stt_elapsed_ms": 4235,
  "stt_output_chars": 12,
  "stt_expected_chars": 12,
  "stt_exact_match": true,
  "stt_nonempty": true,
  "elapsed_ms": 4235
}
```

### normal_single_turn_001:tts - PASS

```json
{
  "tts_elapsed_ms": 1754,
  "tts_frame_count": 20,
  "tts_audio_bytes": 95360,
  "tts_first_sample_rate": 16000,
  "tts_nonempty_audio": true,
  "elapsed_ms": 1754
}
```

### hard_interrupt_001:vad - PASS

```json
{
  "vad_elapsed_ms": 109,
  "vad_event_count": 180,
  "vad_start_count": 1,
  "vad_end_count": 1,
  "vad_inference_count": 178,
  "vad_max_probability": 0.9986303747690342,
  "vad_detected_speech": true,
  "elapsed_ms": 109
}
```

### hard_interrupt_001:stt - PASS

```json
{
  "stt_elapsed_ms": 3384,
  "stt_output_chars": 3,
  "stt_expected_chars": 3,
  "stt_exact_match": true,
  "stt_nonempty": true,
  "elapsed_ms": 3384
}
```

### hard_interrupt_001:tts - PASS

```json
{
  "tts_elapsed_ms": 708,
  "tts_frame_count": 14,
  "tts_audio_bytes": 56960,
  "tts_first_sample_rate": 16000,
  "tts_nonempty_audio": true,
  "elapsed_ms": 708
}
```

### topic_switch_001:vad - PASS

```json
{
  "vad_elapsed_ms": 180,
  "vad_event_count": 311,
  "vad_start_count": 1,
  "vad_end_count": 0,
  "vad_inference_count": 310,
  "vad_max_probability": 0.9965800733259145,
  "vad_detected_speech": true,
  "elapsed_ms": 180
}
```

### topic_switch_001:stt - PASS

```json
{
  "stt_elapsed_ms": 4253,
  "stt_output_chars": 11,
  "stt_expected_chars": 10,
  "stt_exact_match": false,
  "stt_nonempty": true,
  "elapsed_ms": 4253
}
```

### topic_switch_001:tts - PASS

```json
{
  "tts_elapsed_ms": 1550,
  "tts_frame_count": 20,
  "tts_audio_bytes": 99200,
  "tts_first_sample_rate": 16000,
  "tts_nonempty_audio": true,
  "elapsed_ms": 1550
}
```

### correction_001:vad - PASS

```json
{
  "vad_elapsed_ms": 127,
  "vad_event_count": 235,
  "vad_start_count": 1,
  "vad_end_count": 0,
  "vad_inference_count": 234,
  "vad_max_probability": 0.9977082932056508,
  "vad_detected_speech": true,
  "elapsed_ms": 127
}
```

### correction_001:stt - PASS

```json
{
  "stt_elapsed_ms": 3702,
  "stt_output_chars": 8,
  "stt_expected_chars": 8,
  "stt_exact_match": true,
  "stt_nonempty": true,
  "elapsed_ms": 3702
}
```

### correction_001:tts - PASS

```json
{
  "tts_elapsed_ms": 1236,
  "tts_frame_count": 17,
  "tts_audio_bytes": 74880,
  "tts_first_sample_rate": 16000,
  "tts_nonempty_audio": true,
  "elapsed_ms": 1236
}
```

### backchannel_001:vad - PASS

```json
{
  "vad_elapsed_ms": 60,
  "vad_event_count": 97,
  "vad_start_count": 1,
  "vad_end_count": 0,
  "vad_inference_count": 96,
  "vad_max_probability": 0.7866813598693231,
  "vad_detected_speech": true,
  "elapsed_ms": 60
}
```

### backchannel_001:stt - PASS

```json
{
  "stt_elapsed_ms": 3481,
  "stt_output_chars": 1,
  "stt_expected_chars": 1,
  "stt_exact_match": true,
  "stt_nonempty": true,
  "elapsed_ms": 3481
}
```

### backchannel_001:tts - PASS

```json
{
  "tts_elapsed_ms": 509,
  "tts_frame_count": 10,
  "tts_audio_bytes": 31040,
  "tts_first_sample_rate": 16000,
  "tts_nonempty_audio": true,
  "elapsed_ms": 509
}
```

### normal_single_turn_001:eot - PASS

```json
{
  "eot_elapsed_ms": 29,
  "eot_step_count": 1,
  "eot_max_score": 0.9197000917949641,
  "eot_interrupt_count": 1,
  "eot_expected_cancel": false,
  "elapsed_ms": 29
}
```

### hard_interrupt_001:eot - PASS

```json
{
  "eot_elapsed_ms": 4,
  "eot_step_count": 2,
  "eot_max_score": 0.0,
  "eot_interrupt_count": 1,
  "eot_expected_cancel": true,
  "elapsed_ms": 4
}
```

### topic_switch_001:eot - PASS

```json
{
  "eot_elapsed_ms": 3,
  "eot_step_count": 1,
  "eot_max_score": 0.0,
  "eot_interrupt_count": 0,
  "eot_expected_cancel": true,
  "elapsed_ms": 3
}
```

### correction_001:eot - PASS

```json
{
  "eot_elapsed_ms": 0,
  "eot_step_count": 1,
  "eot_max_score": 0.0,
  "eot_interrupt_count": 1,
  "eot_expected_cancel": true,
  "elapsed_ms": 0
}
```

### backchannel_001:eot - PASS

```json
{
  "eot_elapsed_ms": 0,
  "eot_step_count": 1,
  "eot_max_score": 0.0,
  "eot_interrupt_count": 0,
  "eot_expected_cancel": false,
  "elapsed_ms": 0
}
```
