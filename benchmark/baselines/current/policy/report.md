# Voice Benchmark Report: baseline-smoke-2

- runner: `policy`
- profile: `balanced_semantic`
- git_sha: `43f0302`
- pass_rate: `5/5`

## Metric Summary

| metric | p50 | p95 | max | count |
| --- | ---: | ---: | ---: | ---: |
| correction_hint | 0.0 | 0.8 | 1.0 | 5 |
| elapsed_ms | 0.0 | 0.0 | 0.0 | 5 |
| interrupt_decision_ms | 80.0 | 80.0 | 80.0 | 4 |
| topic_switch_hint | 0.0 | 0.8 | 1.0 | 5 |

## Cases

### normal_single_turn_001 - PASS

```json
{
  "elapsed_ms": 0,
  "interrupt_decision_ms": null,
  "expected_action": "none",
  "actual_action": "none",
  "expected_intent": "uncertain",
  "actual_intent": "uncertain",
  "topic_switch_hint": false,
  "correction_hint": false
}
```

### hard_interrupt_001 - PASS

```json
{
  "elapsed_ms": 0,
  "interrupt_decision_ms": 80,
  "expected_action": "cancel",
  "actual_action": "cancel",
  "expected_intent": "hard_stop",
  "actual_intent": "hard_stop",
  "topic_switch_hint": false,
  "correction_hint": false
}
```

### topic_switch_001 - PASS

```json
{
  "elapsed_ms": 0,
  "interrupt_decision_ms": 80,
  "expected_action": "cancel",
  "actual_action": "cancel",
  "expected_intent": "topic_switch",
  "actual_intent": "topic_switch",
  "topic_switch_hint": true,
  "correction_hint": false
}
```

### correction_001 - PASS

```json
{
  "elapsed_ms": 0,
  "interrupt_decision_ms": 80,
  "expected_action": "cancel",
  "actual_action": "cancel",
  "expected_intent": "correction",
  "actual_intent": "correction",
  "topic_switch_hint": false,
  "correction_hint": true
}
```

### backchannel_001 - PASS

```json
{
  "elapsed_ms": 0,
  "interrupt_decision_ms": 80,
  "expected_action": "rollback",
  "actual_action": "rollback",
  "expected_intent": "backchannel",
  "actual_intent": "backchannel",
  "topic_switch_hint": false,
  "correction_hint": false
}
```
