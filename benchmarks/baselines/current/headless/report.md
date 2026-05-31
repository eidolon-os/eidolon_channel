# Voice Benchmark Report: baseline-smoke-2

- runner: `headless`
- profile: `headless_audio_replay`
- git_sha: `43f0302`
- pass_rate: `5/5`

## Metric Summary

| metric | p50 | p95 | max | count |
| --- | ---: | ---: | ---: | ---: |
| agent_message_count | 0.0 | 1.0 | 1.0 | 5 |
| audio_bytes | 0.0 | 65344.0 | 72000.0 | 5 |
| cleared_segments | 0.0 | 0.0 | 0.0 | 5 |
| elapsed_ms | 1486.0 | 2467.0 | 2560.0 | 5 |
| llm_call_count | 1.0 | 1.8 | 2.0 | 5 |
| user_final_count | 1.0 | 1.8 | 2.0 | 5 |

## Cases

### normal_single_turn_001 - PASS

```json
{
  "elapsed_ms": 2095,
  "user_final_count": 1,
  "agent_message_count": 1,
  "audio_bytes": 38720,
  "cleared_segments": 0,
  "llm_call_count": 1
}
```

### hard_interrupt_001 - PASS

```json
{
  "elapsed_ms": 2560,
  "user_final_count": 2,
  "agent_message_count": 1,
  "audio_bytes": 72000,
  "cleared_segments": 0,
  "llm_call_count": 2
}
```

### topic_switch_001 - PASS

```json
{
  "elapsed_ms": 1486,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": 1
}
```

### correction_001 - PASS

```json
{
  "elapsed_ms": 1486,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": 1
}
```

### backchannel_001 - PASS

```json
{
  "elapsed_ms": 969,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": 1
}
```
