# Eidolon Realtime Voice Benchmark

This directory contains the first reproducible baseline for realtime voice
turn-taking work.

## Generate Audio

```bash
./.venv/bin/python scripts/generate_voice_benchmark_audio.py
```

The script uses the configured TTS provider and writes WAV clips to
`benchmark/audio/generated`.

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
benchmark/runs/<run-id>/policy
benchmark/runs/<run-id>/headless
benchmark/runs/<run-id>/component
```

Each runner writes JSONL, `metrics.json`, `report.md`, and `report.html`.

## Compare Against Baseline

```bash
./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmark/baselines/current/policy \
  --candidate benchmark/runs/candidate/policy

./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmark/baselines/current/headless \
  --candidate benchmark/runs/candidate/headless

./.venv/bin/python scripts/compare_voice_bench.py \
  --baseline benchmark/baselines/current/component \
  --candidate benchmark/runs/candidate-components/component \
  --max-p95-regression-pct 30
```

Use the default 10% P95 regression threshold for deterministic runners.
Use a wider threshold for `component` because it includes real network provider
latency; failed cases and empty outputs still fail regardless of this threshold.

## Visual Dashboard

```bash
./.venv/bin/python scripts/report_voice_bench_dashboard.py \
  --full-run benchmark/runs/candidate \
  --direct-run benchmark/runs/candidate-direct/headless \
  --livekit-room-run benchmark/runs/candidate-room/livekit_room \
  --output benchmark/runs/candidate/dashboard.html
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
| `tts_ttfb` (`first_text_sent -> provider_first_audio`) | target/acceptable | 100–250 ms | Phase-2 (advisory) |
| `brain_first_delta` | target/acceptable | 375–750 ms p95 | advisory (upstream p95 variance) |
| `room_interrupt_resolved` | target/acceptable | 500–800 ms | advisory (semantic variance) |

The Phase-2 gates (STT finalization, TTS TTFB, E2E) are honest goals the
preemptive-generation + TTS work will close; they get promoted to `required`
once the system meets them with margin.

TTS note: `tts_request_started_at` marks when the LiveKit TTS stream opens. It
can happen before the brain has emitted any text, so it is useful as a composite
diagnostic but should not be used as provider TTFB. The provider-facing TTFB SLO
uses `tts_first_text_sent_to_provider_first_audio_ms`.

Interrupt note: Tier 0 hard stops still use VAD/speech-start ->
`interrupt_resolved_at` as the primary latency target. Tier 1 semantic
redirects/corrections are split into yield-vs-collect diagnostics:

- `timeline_yield_old_output_ms`: the first confirmed cancel path for the old
  agent output. `max_interrupt_decision_ms` checks this value for topic-switch
  and correction cases.
- `timeline_yield_old_output_playback_stop_ms`: the first client
  `playback.stop` path for the old agent output.
- `timeline_interrupt_speech_to_playback_stop_ms` and
  `timeline_interrupt_started_to_playback_stop_ms`: maximum-path diagnostics
  for when the client `playback.stop` command was queued.
- VAD/speech-start -> `interrupt_resolved_at`: the slowest per-case collection
  path, including continued user speech after the first cancel. This remains
  visible as a diagnostic, but is not the semantic redirect yield gate.
- `interrupt_started_at` -> `interrupt_resolved_at`: channel execution latency
  after attention admission has enough direct semantic evidence to duck/decide.
- `timeline_cancel_then_collect`: marks cases where the old output was already
  cancelled and the user continued speaking a new topic/correction turn.
- `timeline_collect_new_topic_turn_cancel_ms` and
  `timeline_collect_new_topic_turn_playback_stop_ms`: the later collect path,
  kept separate from the old-output yield gate.

The core suite uses a 650 ms acceptable hard-stop total gate to avoid flaking on
one-off STT first-token confusions, while the SLO dashboard still tracks the
500 ms target. Topic switch and correction assert a 250 ms admitted-to-resolved
target. The longer speech-start total remains visible in the timeline report as
an STT/evidence availability diagnostic.

For real room timeline collection, set the channel worker config:

```yaml
observability:
  timeline_debug_path: "benchmark/runs/channel-worker-turn-timeline.jsonl"
```

The `livekit_room` runner snapshots only the new lines written during that
benchmark run into:

```text
benchmark/runs/<run-id>/livekit_room/turn_timeline.jsonl
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

## Half / Full Duplex Suite Layout

Benchmark cases are mode-scoped under:

- `benchmark/cases/full_duplex/`: open-mic natural conversation, barge-in,
  backchannel, false-start, ambient/echo guard, and explicit full-duplex client
  controls.
- `benchmark/cases/half_duplex/`: push-to-talk (PTT) segment owner cases. These
  must run with `--livekit-interaction-mode ptt` — only `ptt` routes to
  `HalfDuplexPttPipeline`. (The directory keeps the legacy `half_duplex` name,
  matching the `HalfDuplexPttPipeline` class / `half_duplex/` package; the
  interaction mode is `ptt` after the 3-mode split in commit 6c2316e.)
- `benchmark/cases/shared/`: deterministic policy or reusable cases that are
  not tied to one room interaction mode.
- `benchmark/cases/legacy/`: historical compatibility suites. Default E2E
  gates do not run these.

`scripts/bench_barge_in_e2e_ab.py` now runs mode-specific suite sets. Its
default is `--suite-set full_duplex_gate`, which currently expands to:

```text
benchmark/cases/full_duplex/gate_enforced.yaml
benchmark/cases/full_duplex/explicit_control_enforced.yaml
```

Useful E2E invocations:

```bash
./.venv/bin/python scripts/bench_barge_in_e2e_ab.py \
  --suite-set full_duplex_gate \
  --livekit-interaction-mode full_duplex

./.venv/bin/python scripts/bench_barge_in_e2e_ab.py \
  --suite-set half_duplex_ptt_phase_a \
  --livekit-interaction-mode ptt
```

Room cases in these gate suites must declare `agent_audio_response` explicitly:
`none`, `first`, or `after_user_done`. The E2E A/B script rejects implicit
`auto` room expectations so a no-decision/no-audio case cannot look green by
accident.

Reports now expose two layers of failure attribution:

- `functional_outcome_passed`: terminal action, decision, context ledger, and
  required control packets are correct.
- `experience_slo_passed`: the functional behavior happened inside the
  configured latency bound.

This keeps "correct cancel but too slow" visible as an experience failure
instead of mixing it with owner logic failures.

## Human + Device Dogfood Suites

Dogfood cases model the whole product envelope: a human speaks while a device is
playing agent audio, the device publishes `client.audio_state`, and the mic
input can include deterministic echo/noise. The first suite is explicit-only:

```bash
./.venv/bin/python scripts/bench_voice.py \
  --cases benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml \
  --runner headless \
  --run-id dogfood-headless

./.venv/bin/python scripts/bench_voice.py \
  --cases benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml \
  --runner livekit_room \
  --run-id dogfood-room
```

For the real-room dogfood gate used before Box-3 hardware sessions, run the
suite through the E2E wrapper so the worker, timeline, real-call verification,
and participant metadata are captured together:

```bash
./.venv/bin/python scripts/bench_barge_in_e2e_ab.py \
  --profiles channel \
  --cases benchmark/cases/full_duplex/dogfood_box3_audio_first_enforced.yaml \
  --repeat 1 \
  --manage-worker \
  --run-id dogfood-box3-audio-first \
  --livekit-participant-identity "$EIDOLON_BOX3_DEVICE_ID" \
  --livekit-participant-kind device \
  --livekit-interaction-mode full_duplex
```

`EIDOLON_BOX3_DEVICE_ID` must be a claimed, companion-bound device that passes
admin `/api/resolve/device/{id}`. The runner checks this before opening a room;
a synthetic name such as `bench-device` does not exercise the runtime
identity/token boundary used by hardware dogfood.

Do not treat the generic full-duplex gate as a substitute for this suite. A
controlled Box-3 full-duplex dogfood session requires both the owner-follow-up
and backchannel dogfood cases to pass with real-call evidence. Policy/headless
results are useful regressions, but they can be optimistic when real STT/EOT
timing under echo/noise waits for a final transcript.

For deterministic local policy regression, run the offline policy regression
suite. It models interaction semantics at policy level: Waveshare PTT idle tap,
PTT tap-to-stop during playback, compound backchannel, false-start, echo safety,
and a real follow-up after low-evidence prefixes. It does not replace real-room
or device dogfood.

```bash
./.venv/bin/python scripts/bench_barge_in_ab.py \
  --cases benchmark/cases/shared/offline_policy_regression_enforced.yaml \
  --repeat 1 \
  --run-id offline-policy-regression
```

For local ESP32 box-3 full-duplex runtime validation, keep the runtime switch
explicit by loading the overlay:

```bash
EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML=config/overlays/box3_full_duplex.yaml
```

After a real device dogfood attempt, inspect the worker timeline evidence:

```bash
./.venv/bin/python scripts/analyze_hil_barge_in.py \
  --timeline benchmark/runs/channel-worker-turn-timeline.jsonl \
  --latest 8 \
  --require-cancel
```

For the full-duplex contract work, the timeline review should also inspect:

- `full_duplex_state_transitions`: reversible `provisional_duck` may happen early;
  irreversible `accepted_interruption` / `user_turn_committed` must have terminal
  evidence, or explicit client preempt evidence.
- `user_turn_owner_ledger`: every candidate should end as accepted, rejected,
  merged, dropped, or superseded; no candidate should be both rejected and used as
  canonical user text.
- `user_turn_superseded_finalized`: when a replacement speech is accepted after
  superseding an older pending request, the old request should have a final owner
  record instead of disappearing silently.
- `playback.stop` publish/ack: for natural-language full-duplex interrupts, a
  rejected artifact/backchannel candidate should not leave this irreversible
  output effect behind. Explicit PTT/client-preempt paths are separate controls.

The device-envelope YAML extension is intentionally benchmark-owned. It can be
used by both real dogfood suites and deterministic policy regression suites:

```yaml
device_envelope:
  enabled: true
  device:
    model: esp32_box_3
    mode: full_duplex
    audio_state_hz: 10
  agent:
    speaking_text: "..."
    tts_duration_ms: 6000
  acoustics:
    echo:
      enabled: true
      delay_ms: 80
      attenuation_db: -18
    noise:
      enabled: true
      snr_db: 20
```

Runner responsibilities stay separated:

- schema parses device-envelope metadata and expectations.
- `benchmark.device_envelope` renders synthetic mic audio and device cadence.
- `headless` replays the rendered mic audio in memory.
- `component` applies the same mic rendering to VAD/STT inputs.
- `livekit_room` publishes `client.audio_state` at the configured device cadence.
- `livekit_room` timeline expectations enforce dogfood SLOs for
  speech-start-to-suspend, speech-start-to-cancel/resume, `playback.stop`, and
  interrupted-context capture.
- `scripts/analyze_hil_barge_in.py` checks real-device timeline records that do
  not use benchmark room names.
- future HIL runners should consume the same YAML, replacing synthetic echo with
  captured playback reference audio and real device logs.
