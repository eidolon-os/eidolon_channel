# Eidolon Channel Realtime Voice Upgrade Status

Last updated: 2026-05-29

This document records what has been implemented from the realtime voice upgrade
plan, what is only partially complete, and what remains open. It is meant to be
the durable handoff point for future iterations, instead of relying on chat
history.

## Current State

The channel now has a working architecture for a reproducible realtime voice
baseline:

- structured config loading through `load_effective_config()`;
- separated turn policy modules under `eidolon/livekit/agent/turn_policy`;
- timeline observability under `eidolon/livekit/agent/observability`;
- benchmark runners under `eidolon/livekit/benchmarks`;
- generated real WAV fixtures under `benchmarks/audio/generated`;
- real LiveKit room benchmark execution through `LiveKitRoomBenchmarkRunner`;
- visual dashboard generation through `scripts/report_voice_bench_dashboard.py`.

Latest validated `direct_llm` real-room run:

```text
benchmarks/runs/timeline-room-observable-5of5-20260529/livekit_room
```

Latest validated `eidolon_agent` real-room run:

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/livekit_room
```

Latest `eidolon_agent` dashboard:

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/dashboard-with-room.html
```

Latest focused verification:

```text
77 passed
```

Latest real LiveKit room benchmark:

```text
5/5 passed
```

Covered scenarios:

- normal single turn;
- hard interrupt;
- topic switch;
- correction;
- backchannel / false interrupt.

## Completed

### Config Governance

Implemented:

- `eidolon/livekit/common/config/schema.py`
- `eidolon/livekit/common/config/loader.py`
- `eidolon/livekit/common/config/profiles.py`
- `eidolon/livekit/common/config/validators.py`
- `load_effective_config() -> EffectiveAgentConfig`

Runtime code now reads the structured effective config rather than scattered
experience env vars. `settings.yaml` owns provider, turn policy, profile, and
observability parameters. `.env` is intended for secrets, endpoints, and deploy
identity.

Important note: the project intentionally did not preserve backward-compatible
old parameter names. New config shape is the source of truth.

### Turn Intelligence Layer

Implemented:

- `eidolon/livekit/agent/turn_policy/runtime.py`
- `eidolon/livekit/agent/turn_policy/decider.py`
- `eidolon/livekit/agent/turn_policy/intent_classifier.py`
- `eidolon/livekit/agent/turn_policy/eot_config.py`

The main decision logic no longer lives directly inside `StreamingPipeline`.
The split is:

- `InterruptDecider`: pure decisions.
- `TurnPolicyRuntime`: small adapter around policy state and turn-control metadata.
- `StreamingPipeline`: LiveKit session wiring and side effects.

Implemented turn-control behavior:

- hard stop cancels current output and interrupts the session;
- topic switch emits a channel hint but does not own brain memory/topic state;
- correction emits a correction hint;
- backchannel/noise can rollback instead of hard cancelling;
- VAD-active timeout without transcript no longer blindly cancels;
- late transcript after VAD end can still update semantic turn-control decisions.

### Hard Interrupt Classifier

Implemented:

- `InterruptIntentClassifier`
- `LexiconInterruptClassifier`
- `NoopModelInterruptClassifier`
- `OnnxInterruptClassifier` placeholder

Current production path:

```text
LexiconInterruptClassifier + EOT score + VAD state + interim length
```

The ONNX classifier remains a reserved integration point only. No generic model
was introduced into the production hot path.

### Streaming Pipeline Changes

Implemented:

- `StreamingPipeline` records turn timeline marks.
- LLM metrics are bridged into the active timeline through LiveKit
  `metrics_collected`.
- `llm_first_delta_at` is derived from LiveKit LLM TTFT.
- rollback no longer flushes timeline too early, so late STT can correct
  backchannel/correction metadata before JSONL is written.
- fallback semantic path handles transcript that arrives after the initial
  duck/rollback decision.

Key timeline fields now include:

- `speech_started_at`
- `speech_stopped_at`
- `stt_stream_started_at`
- `stt_ws_connected_at`
- `stt_stream_first_audio_sent_at`
- `stt_first_audio_sent_at`
- `stt_flush_sent_at`
- `stt_provider_first_partial_at`
- `stt_provider_final_at`
- `transcript_interim_first_at`
- `transcript_final_at`
- `turn_committed_at`
- `llm_started_at`
- `llm_first_delta_at`
- `tts_first_audio_at`
- `agent_audio_playback_done_at`
- `interrupt_started_at`
- `interrupt_resolved_at`

### Observability

Implemented:

- `eidolon/livekit/agent/observability/timeline.py`
- `eidolon/livekit/agent/observability/metrics.py`
- JSONL timeline export through `observability.timeline_debug_path`
- dashboard timeline ingestion
- provider latency summaries
- decision reason summaries
- timeline case coverage checks

Important added attributes:

- `decision_reason`
- `interrupt_action`
- `rollback_drop_buffered`
- `provider_latency_ms`
- `provider_segments`
- `brain_rpc`
- `llm_metrics`
- `turn_control`

Latest `direct_llm` normal-turn latency from real LiveKit room:

```text
commit_to_llm_started_ms: 1583.6
commit_to_llm_first_delta_ms: 5173.9
commit_to_tts_first_audio_ms: 6035.2
llm_first_delta_to_tts_first_audio_ms: 861.3
```

This confirms the observability chain now captures LLM first delta, but also
shows that direct LLM TTFT/provider variance is a real experience bottleneck.

Latest `eidolon_agent` normal-turn latency from real LiveKit room:

```text
llm_ttft_ms: 267.37
llm_duration_ms: 601.45
commit_to_brain_request_started_ms: 1012.33
brain_request_write_ms: 8.49
brain_request_to_first_delta_ms: 258.56
brain_stream_duration_ms: 592.63
commit_to_tts_first_audio_ms: 1635.28
brain_first_delta_to_tts_first_audio_ms: 355.89
```

This run confirms the production brain path can be measured by the same
timeline and dashboard pipeline. The `eidolon_agent` RPC stream itself is fast;
the remaining normal-turn latency is mostly before the RPC starts and around
TTS first audio.

Provider latency segmentation is now first-class:

- timeline JSONL writes normalized `provider_segments`;
- Bailian streaming STT emits provider events for stream start, websocket
  connect, turn first audio, provider first partial, flush, and provider final;
- `eidolon_agent` RPC emits `brain_request_started`, `brain_request_sent`,
  `brain_first_delta`, and `brain_done` events;
- benchmark ingestion aggregates provider segments by stage;
- dashboard renders `Provider Latency Breakdown`;
- dashboard renders `Per-Turn Provider Segments`;
- old timeline JSONL without `provider_segments` can still be summarized from
  timestamps.

Current STT streaming interpretation:

- `stt_stream_first_audio_sent_at` is raw websocket-stream diagnostics.
- `stt_first_audio_sent_at` is the first audio chunk sent after Channel opens a
  user turn.
- STT provider-to-LiveKit transcript propagation is currently sub-millisecond
  in the latest real-room run.
- The experience-relevant STT segment is now `speech_started_at ->
  stt_provider_first_partial_at`, not websocket connect time.

### Benchmark And Baseline

Implemented:

- `benchmarks/cases/core.yaml`
- `benchmarks/audio/generated/*`
- `scripts/generate_voice_benchmark_audio.py`
- `scripts/bench_voice.py`
- `scripts/report_voice_bench.py`
- `scripts/report_voice_bench_dashboard.py`
- `scripts/compare_voice_bench.py`
- `eidolon/livekit/benchmarks/policy_runner.py`
- `eidolon/livekit/benchmarks/headless_runner.py`
- `eidolon/livekit/benchmarks/component_runner.py`
- `eidolon/livekit/benchmarks/livekit_room_runner.py`
- `eidolon/livekit/benchmarks/timeline_expectations.py`

Runner layers:

- `policy`: pure turn-policy decision regression.
- `headless`: in-memory AgentSession style replay with controlled mocks.
- `component`: real VAD/EOT/STT/TTS provider path.
- `livekit_room`: real LiveKit room boundary with real worker, real STT/TTS/LLM,
  and generated voice clips.

The real-room runner now:

- waits for the agent participant before user audio;
- uses PCM energy to detect audible agent speech instead of treating silence
  frames as speech;
- pushes audio in real 20 ms pacing instead of instantly dumping WAV bytes;
- waits for post-user agent audio when the case expects a reply;
- snapshots only new worker timeline JSONL lines into each run directory.

Latest real-room expectation validation:

```text
normal_single_turn_001: pass
hard_interrupt_001: pass
topic_switch_001: pass
correction_001: pass
backchannel_001: pass
```

Latest `eidolon_agent` STT streaming validation:

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/livekit_room
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/dashboard-with-room.html

5/5 passed
```

Notable fixes from this validation:

- correction fragments such as `是我` now hold for more interim text instead of
  cancelling too early as `normal_interrupt`;
- `是我刚...` still upgrades to correction and sets `correction_hint=True`;
- raw STT stream-first-audio is separated from per-turn first-audio latency, so
  dashboard STT latency no longer reports negative turn timing.

Latest `eidolon_agent` real-room expectation validation:

```text
normal_single_turn_001: pass
hard_interrupt_001: pass
topic_switch_001: pass
correction_001: pass
backchannel_001: pass
```

The `eidolon_agent` path was validated with:

```text
providers.brain_provider: eidolon_agent
providers.eidolon_agent.remote_target: 127.0.0.1:45051
```

The gRPC brain smoke test succeeded and the room benchmark exercised the same
generated voice cases through the real LiveKit room boundary.

### STT/TTS Streaming Understanding

The implementation now treats STT/TTS as long-lived streaming WebSocket-style
providers, not one-shot calls.

Relevant fix:

- Bailian STT streaming no longer waits forever when the receive loop finishes
  before the send loop.
- Room benchmark is paced in real time so VAD/STT/TTS timing is not distorted.

### Eidolon Agent Brain Benchmark

Implemented:

- real `eidolon_agent` gRPC smoke verification through the Channel adapter;
- real LiveKit room benchmark with `providers.brain_provider: eidolon_agent`;
- dashboard generation for the `eidolon_agent` run;
- turn-control metadata verification through timeline output.

Latest run:

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/livekit_room
```

Latest dashboard:

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/dashboard-with-room.html
```

Result:

```text
5/5 passed
```

Verified behavior:

- hard interrupt produced `decision=cancel hard_stop strong_intent`;
- topic switch produced `decision=cancel topic_switch intent:topic_switch`;
- correction produced `decision=cancel correction intent:correction`;
- backchannel produced `decision=rollback backchannel intent:backchannel`.

Important fix discovered by this benchmark:

- STT can briefly emit a single-character mistaken interim fragment such as
  `听` before later producing the intended hard-stop phrase `停一下`.
- The interrupt deadline path now holds when normalized transcript length is
  below `min_interim_chars`, instead of converting the partial fragment into a
  premature normal interrupt.
- Regression coverage was added in
  `eidolon/livekit/tests/agent/test_interrupt_decider.py`.

## Partially Complete

### OpenAI Realtime Parameter Alignment

Completed:

- VAD, EOT, interrupt, ducking, and idle parameters are centralized.
- Defaults are aligned toward OpenAI Realtime-style `server_vad` /
  semantic-VAD behavior.
- Profiles exist for the local policy system.

Still open:

- no LiveKit multilingual turn detector assisted profile yet;
- no full apples-to-apples OpenAI Realtime E2E comparison run yet.

### LiveKit Turn Handling Integration

Completed:

- Existing LiveKit AgentSession turn handling is still used.
- Channel retains lower-latency control over duck/cancel/rollback.

Still open:

- LiveKit adaptive interruption events are not yet adapted into our timeline.
- No `LiveKitAdaptiveSignalAdapter` yet.
- LiveKit multilingual turn detector is not wired as optional endpointing input.

### Noise / Echo Cancellation

Completed:

- The architecture decision is clear: noise/echo should be handled before
  VAD/STT where possible.
- Current local benchmark now exposes noise/backchannel behavior through real
  room testing.

Still open:

- no local self-hosted noise cancellation implementation;
- no `audio_input.noise_cancellation` settings block fully wired;
- no local echo cancellation benchmark case beyond current backchannel/noise
  style checks.

### Admin Benchmark Display

Completed or started:

- Benchmark output is structured enough for `eidolon_admin` to consume.
- Dashboard artifacts are generated as static HTML/JSON.

Still open for productization:

- historical run index;
- run comparison UI;
- trend charts;
- per-case drilldown in admin;
- SLO gate display as first-class admin state.

### SLO Gates

Completed:

- SLO gate infrastructure exists in `eidolon/livekit/benchmarks/slo.py`.
- Dashboard renders SLO findings.

Still open:

- decide final SLO thresholds after more stable provider runs;
- add CI or release-gate behavior;
- separate deterministic gates from real-provider warning thresholds.

## Not Done

### Preemptive LLM Generation

Status: intentionally deferred.

Reason: direct LLM is currently used mainly for comparison and benchmark work;
the real production brain is `eidolon_agent`. Preemptive generation should be
designed with the brain project and its cancellation/tool/memory semantics, not
as a direct-LLM-only feature in Channel.

### OpenTelemetry Exporter

Status: not done.

JSONL and dashboard exist. OpenTelemetry export is still future work.

Needed:

- exporter module;
- config toggle;
- span/event naming;
- test with local collector or mock exporter;
- keep JSONL as the simple local fallback.

### ONNX INT8 Intent Model

Status: not done.

The placeholder exists, but no model has been trained or integrated.

Separate task requirements:

- train or distill Chinese hard-interrupt / topic-switch / correction /
  backchannel classifier;
- export ONNX INT8;
- CPU P95 <= 30 ms;
- hard stop recall >= 98%;
- backchannel false cancel <= 2%;
- topic switch precision >= 90%;
- benchmark against current lexicon path before enabling.

### LiveKit Adaptive Interruption Adapter

Status: not done.

Needed:

- capture LiveKit interruption/adaptive signals where available;
- map them into a neutral adapter type;
- write timeline side-by-side comparison against `InterruptDecider`;
- keep Channel policy authoritative until measured evidence says otherwise.

### LiveKit Multilingual Turn Detector Assisted Profile

Status: not done.

Needed:

- optional profile such as `livekit_detector_assisted`;
- use it only for endpointing/commit help;
- do not put it into the hard interrupt hot path initially;
- benchmark long pause, hesitation, and incomplete sentence cases.

### Real Noise / Echo Cancellation

Status: not done.

For local LiveKit server, this needs a self-hosted or SDK-level solution. LiveKit
Cloud options such as Krisp / advanced noise cancellation do not directly apply
to the local server path unless deployment changes.

Needed:

- choose local strategy;
- add config;
- add noisy audio fixtures;
- run VAD false-positive and STT contamination benchmarks.

## Known Risks

### Provider Variance

Direct LLM and real STT/TTS providers show significant latency variance.
Benchmark timeouts should be wide enough to record slow requests as data rather
than convert them into false functional failures.

### Generated TTS Fixtures

Generated audio is convenient and reproducible enough for now, but it is not a
replacement for a production audio corpus. More real human audio should be added
for:

- barge-in at different loudness levels;
- overlapping speech;
- noisy room;
- far-field microphone;
- fast Mandarin;
- dialect or mixed Chinese/English.

### Lexicon-Based Intent Limits

The current lexicon path is fast and debuggable, but it is not enough for broad
natural language coverage. It should remain as a high-precision fast path even
after a small model is introduced.

### Timeline Flush Timing

Timeline writes are now less eager for rollback so late STT can update intent.
Future changes should be careful not to reintroduce early JSONL flushes before
the turn has had a chance to receive final or late interim transcripts.

## Suggested Next Milestones

### Milestone 1: Stabilize And Commit Current Baseline

Goal:

- commit the current architecture, benchmark, and observability work;
- keep latest known-good dashboard path in docs.

Recommended checks:

```bash
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_intent_classifier.py \
  eidolon/livekit/tests/agent/test_interrupt_decider.py \
  eidolon/livekit/tests/agent/test_first_signal_trigger.py \
  eidolon/livekit/tests/agent/test_timeline.py \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py

./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/core.yaml \
  --run-id <run-id> \
  --livekit-room-timeout-sec 150 \
  --livekit-room-settle-sec 2 \
  --livekit-agent-ready-timeout-sec 15
```

### Milestone 2: Harden Eidolon Agent Brain Benchmark

Goal:

- turn the passing production brain benchmark into a stable release gate.

Deliverables:

- compare latest `direct_llm` and `eidolon_agent` dashboards side by side;
- add explicit `CancelTurn(turn_id)` assertion once the brain service exposes a
  durable acknowledgement signal;
- expand provider latency breakdown beyond LLM metrics into STT/TTS streaming
  stage timing;
- run repeated samples to separate provider variance from policy regressions;
- define warning and fail thresholds for real-provider SLO gates.

### Milestone 3: Productize Benchmark In Admin

Goal:

- move from static artifacts to benchmark management.

Deliverables:

- run list;
- latest baseline pin;
- per-run dashboard embedding;
- trend comparison;
- SLO gate status.

### Milestone 4: Add OpenTelemetry Export

Goal:

- make the timeline useful outside local JSONL.

Deliverables:

- OTEL exporter;
- config switch;
- local collector smoke test;
- docs for trace fields.

### Milestone 5: Evaluate LiveKit-Assisted Turn Features

Goal:

- borrow mature LiveKit capabilities without giving up Channel hot-path control.

Deliverables:

- adaptive interruption signal adapter;
- multilingual turn detector assisted profile;
- benchmark report comparing policy-only vs assisted mode.

### Milestone 6: Noise / Echo Work

Goal:

- reduce false VAD/STT activation before turn policy receives polluted signals.

Deliverables:

- local noise/echo strategy;
- fixtures;
- SLOs for false activation and false cancel.

## Completion Estimate

Original implementation plan only:

```text
82-86% complete
```

Including the broader product-experience plan against top realtime voice agents:

```text
65-70% complete
```

The reason for the gap is that the core Channel architecture and baseline
mechanism are now in place and the production `eidolon_agent` path has passed a
real-room benchmark. Production-grade evaluation still needs repeated
`eidolon_agent` sampling, admin productization, OTEL export, and optional
LiveKit assisted turn/noise capabilities.
