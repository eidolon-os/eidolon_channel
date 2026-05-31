# Eidolon Realtime Voice Benchmark

This directory contains the first reproducible baseline for realtime voice
turn-taking work.

## Generate Audio

```bash
./.venv/bin/python scripts/generate_voice_benchmark_audio.py
```

The script uses the configured TTS provider and writes WAV clips to
`benchmarks/audio/generated`.

## Run Benchmark

```bash
./.venv/bin/python scripts/bench_voice.py --runner all --run-id candidate
```

Run only the real provider component layer:

```bash
./.venv/bin/python scripts/bench_voice.py \
  --runner component \
  --run-id candidate-components
```

Run the real LiveKit room boundary. Start the LiveKit server and `eidolon`
agent worker first:

```bash
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --run-id candidate-room
```

Use the configured direct LLM for the headless layer:

```bash
./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --llm-mode direct \
  --run-id candidate-direct
```

## Repeat For Stable Percentiles

A single pass produces one sample per case, so its p50/p95 are not meaningful.
Use `--repeat N` to run the whole suite N times and aggregate per-case
percentiles and jitter from a real distribution:

```bash
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --repeat 5 \
  --run-id candidate-room
```

For a single run, raw per-runner JSONL stays at the top level as before. For
`--repeat N > 1`, each pass writes raw output under `repeat-NN/`, while the
merged `metrics.json` / `report.*` at the runner root carry:

- pooled `summary.metrics` (every case x repeat sample) for SLO/regression
  gates, so percentiles are no longer single-value noise;
- `summary.per_case` with each case's p50/p95/`stdev` (jitter) and pass rate;
- `summary.flaky`, the count of cases that passed in some repeats but not all.

Run at least 5 repeats (ideally 10-20) before treating real-provider SLO gates
as hard pass/fail rather than advisory.

## Real-Call Verification

`component` and `livekit_room` must provably hit real providers, not mocks or a
dead worker. Before those runners execute, a pre-flight smoke check exercises the
configured stack (`SharedStageFactory.from_config`) and writes `preflight.json`;
the run aborts if any provider is unreachable.

Each case is then verified for real-call evidence (provider != mock, minimum
audio bytes, and for the room: brain-gRPC `request_id` + `brain_request_sent` /
`brain_first_delta` marks and an STT stream for the configured provider). A case
that cannot be proven real fails (strict, the default) and the dashboard renders
a `bad` "real call NOT verified" finding.

```bash
# Strict (default): real-call failures fail the case.
./.venv/bin/python scripts/bench_voice.py --runner livekit_room --repeat 20

# Bypass the pre-flight smoke gate (e.g. offline iteration on the harness):
./.venv/bin/python scripts/bench_voice.py --runner component --skip-preflight

# Record real-call failures as advisory warnings instead of failing cases:
./.venv/bin/python scripts/bench_voice.py --runner component --lenient-realcall
```

Outputs are written to:

```text
benchmarks/runs/<run-id>/policy
benchmarks/runs/<run-id>/headless
benchmarks/runs/<run-id>/component
```

Each runner writes JSONL, `metrics.json`, `report.md`, and `report.html`.

## Compare Against Baseline

```bash
./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmarks/baselines/current/policy \
  --candidate benchmarks/runs/candidate/policy

./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmarks/baselines/current/headless \
  --candidate benchmarks/runs/candidate/headless

./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmarks/baselines/current/component \
  --candidate benchmarks/runs/candidate-components/component \
  --max-p95-regression-pct 30
```

Use the default 10% P95 regression threshold for deterministic runners.
Use a wider threshold for `component` because it includes real network provider
latency; failed cases and empty outputs still fail regardless of this threshold.

## Visual Dashboard

```bash
./.venv/bin/python scripts/report_voice_bench_dashboard.py \
  --full-run benchmarks/runs/candidate \
  --direct-run benchmarks/runs/candidate-direct/headless \
  --livekit-room-run benchmarks/runs/candidate-room/livekit_room \
  --output benchmarks/runs/candidate/dashboard.html
```

The dashboard contains three layers of judgment:

- regression gates: compare candidate P95 against the committed baseline.
- SLO gates: compare important experience metrics against fixed thresholds.
- turn timeline: inspect per-turn node latency and turn-policy decisions.

SLO gates are tiered (`target` = industry top-tier, `acceptable` = upper limit)
and each carries a `min_samples` guard: timeline percentile gates stay
`advisory` (reported, never block) until a run has >=20 samples of that metric,
so a low-repeat run cannot claim or fail top-tier on noise.

### Enforcement ratchet

Only gates the system already meets are `required`; a required gate that
hard-fails makes `--enforce-slo` exit non-zero (CI / release gate). Advisory and
unmet gates never block.

```bash
./.venv/bin/python scripts/bench_voice.py --runner livekit_room --repeat 20 --enforce-slo
```

| gate | tier | threshold | enforced |
| --- | --- | ---: | :--: |
| `room_user_done_to_next_audio` | acceptable | 1400 ms p95 | ✅ required |
| `room_publish_to_first_audio` | acceptable | 2800 ms p95 | ✅ required |
| `commit_to_first_audio` (E2E) | target/acceptable | 800–1400 ms | Phase-2 (advisory) |
| `stt_final_after_commit` | target/acceptable | 350–500 ms | Phase-2 (advisory) |
| `tts_ttfb` | target/acceptable | 100–250 ms | Phase-2 (advisory) |
| `brain_first_delta` | target/acceptable | 375–750 ms p95 | advisory (upstream p95 variance) |
| `room_interrupt_resolved` | target/acceptable | 500–650 ms | advisory (semantic variance) |

The Phase-2 gates (STT finalization, TTS TTFB, E2E) are honest goals the
preemptive-generation + TTS work will close; they get promoted to `required`
once the system meets them with margin.

For real room timeline collection, set the channel worker config:

```yaml
observability:
  timeline_debug_path: "benchmarks/runs/channel-worker-turn-timeline.jsonl"
```

The `livekit_room` runner snapshots only the new lines written during that
benchmark run into:

```text
benchmarks/runs/<run-id>/livekit_room/turn_timeline.jsonl
```

## Runner Boundaries

- `policy`: pure `TurnPolicyRuntime` regression tests. Fast and deterministic.
- `headless`: real WAV input replay through the in-memory LiveKit AgentSession
  harness with scripted STT/VAD and mock TTS/LLM. Stable enough for PR checks.
- `component`: real FireRed VAD, local EOT model, configured STT provider, and
  configured TTS provider. This catches provider health and integration issues,
  but it is slower and can be affected by network/provider variance.
- `livekit_room`: real LiveKit room boundary. It expects the LiveKit server and
  `eidolon` agent worker to already be running, then joins as a participant,
  publishes benchmark audio, subscribes to agent audio, and records room-level
  latency.

The current baseline is intentionally small and should grow with real failure
cases from production sessions.
