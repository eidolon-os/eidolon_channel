# Voice Benchmark Report: baseline-direct-verify

- runner: `headless`
- profile: `headless_audio_replay:direct`
- git_sha: `43f0302`
- pass_rate: `5/5`

## Metric Summary

| metric | p50 | p95 | max | count |
| --- | ---: | ---: | ---: | ---: |
| agent_message_count | 0.0 | 1.0 | 1.0 | 5 |
| audio_bytes | 0.0 | 304448.0 | 315200.0 | 5 |
| cleared_segments | 0.0 | 0.0 | 0.0 | 5 |
| elapsed_ms | 1494.0 | 5570.0 | 5751.0 | 5 |
| user_final_count | 1.0 | 1.8 | 2.0 | 5 |

## Cases

### normal_single_turn_001 - PASS

```json
{
  "elapsed_ms": 5751,
  "user_final_count": 1,
  "agent_message_count": 1,
  "audio_bytes": 261440,
  "cleared_segments": 0,
  "llm_call_count": null
}
```

### hard_interrupt_001 - PASS

```json
{
  "elapsed_ms": 4846,
  "user_final_count": 2,
  "agent_message_count": 1,
  "audio_bytes": 315200,
  "cleared_segments": 0,
  "llm_call_count": null
}
```

### topic_switch_001 - PASS

```json
{
  "elapsed_ms": 1494,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": null
}
```

### correction_001 - PASS

```json
{
  "elapsed_ms": 1488,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": null
}
```

### backchannel_001 - PASS

```json
{
  "elapsed_ms": 971,
  "user_final_count": 1,
  "agent_message_count": 0,
  "audio_bytes": 0,
  "cleared_segments": 0,
  "llm_call_count": null
}
```
