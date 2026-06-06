# V1 Attention Admission Rollout

> Date: 2026-06-06
> Status: staged implementation plan
> Scope: Eidolon V1 realtime companion quality

## Goal

Keep Eidolon's realtime, interruptible companion experience while reducing false reactions to non-target speech, environmental speech, backchannels, and agent echo.

The target behavior is:

```text
Do not react just because there is speech.
React when the speech is likely directed at Eidolon or explicitly interrupts it.
```

This plan deliberately builds on the existing `eidolon_channel` mechanisms:

- `StreamingPipeline`
- `TurnPolicyRuntime`
- `InterruptDecider`
- `OutputController`
- FireRed pVAD and EOT
- STT provider timeline events
- existing headless/component/room benchmark harnesses
- existing turn-control metadata path into `eidolon_agent`

It must not introduce a second competing turn pipeline.

## Non-Goals

- Do not replace LiveKit `AgentSession`.
- Do not replace `OutputController`.
- Do not call an LLM in the realtime interrupt hot path.
- Do not make speaker verification a hard identity lock in V1.
- Do not try to solve arbitrary overlapping multi-speaker recognition in V1.

## Current Baseline

The current channel already has important building blocks:

- Framework audio-activity auto-interrupt is disabled, so Eidolon policy is the authority.
- `OutputController` can duck, rollback, or cancel TTS output.
- `InterruptDecider` classifies hard stop, topic switch, correction, backchannel, noise, normal interrupt, and uncertain.
- `TurnPolicyRuntime` converts policy decisions into metadata.
- Bailian STT has an opt-in VAD-gated forwarding path.
- Timeline and benchmark infrastructure already record many latency and decision events.

The main gap is earlier than `InterruptDecider`:

```text
LiveKit user_state=speaking
  -> currently can enter duck/interrupt path too eagerly
```

V1 needs a light admission layer before ducking/interruption.

## Stage Gates

Each stage must be independently shippable. After each stage:

1. Run unit tests.
2. Run focused component/headless benchmarks.
3. Run at least one real LiveKit room smoke.
4. Run dogfood in a real room.
5. Compare against the previous accepted baseline.
6. Decide whether to proceed, tune, rollback, or stop.

Do not start the next stage until the previous stage has an accepted report.

## Current Status Snapshot

Updated: 2026-06-06

### Completed

- Stage 0 froze the attention-admission benchmark baseline.
- Stage 1 added client audio hygiene and the `client.audio_state` data-channel hint path.
- Stage 2 added `AttentionAdmission` in observe-only mode, reusing the existing
  `StreamingPipeline`, `TurnPolicyRuntime`, and `InterruptDecider` path.
- Stage 2.1 added `TranscriptEvidenceGate` to stop short STT artifacts such as
  `If`, `对`, or `也` from causing irreversible `normal_interrupt` cancellation
  before a later hard-stop transcript arrives.
- Stage 2.2 stabilized real-room evidence capture, made gRPC brain startup
  observable, added timeout wiring, and fixed the room benchmark participant
  identity contract (`metadata.kind=user|device` plus resolvable identity).
- Stage 2.2 added a weak-signal follow-up guard: after a short artifact,
  backchannel-like fragment, or noise-like hold, a following normal-interrupt
  candidate within a short window is held instead of immediately canceled.
- Stage 2.3 fixed Bailian TTS pool shutdown/task cleanup while preserving the
  prewarmed-connection performance strategy.
- Stage 2.4 audited STT provider lifecycles. There is no TTS-style prewarmed
  connection pool in STT, but Bailian FunASR abnormal close paths were hardened.
- Stage 2.5 added an explicit enforce-mode benchmark suite and taught the
  room runner to refresh `client.audio_state` during each audio step, so
  playback-time admission does not depend on a stale one-shot hint.

### Accepted Reports

Deterministic and local benchmark reports accepted so far:

- Stage 0 policy/headless: `5/5`, 3 repeats.
- Stage 1 policy/headless: `5/5`.
- Stage 2 observe-only policy/headless: `5/5`, 3 repeats.
- Stage 2.1 policy/headless: `6/6`, 3 repeats.
- Stage 2.1 focused tests: `81 passed`.
- Stage 2.1 config + benchmark infrastructure tests: `34 passed`.
- Stage 2.2 adapter/timeline/benchmark tests: `40 passed`, `48 passed`.
- Stage 2.2 policy/headless after weak-signal guard: `6/6`, 3 repeats.
- Stage 2.3 TTS pool/stage/benchmark tests: `54 passed`.
- Stage 2.4 Bailian STT lifecycle/transcript-gate tests: `31 passed`.
- Stage 2.4 SenseTime STT persistent/reliability tests: `32 passed`.
- Stage 2.5 attention admission focused tests: `42 passed`.
- Stage 2.5 enforce-mode policy benchmark: `3/3`, 3 repeats.
- Stage 2.5 baseline policy regression after enforce harness changes: `6/6`,
  3 repeats.

Real LiveKit room status:

- Stage 2 room smoke showed the room/data-channel path works but exposed
  first-interim instability.
- Stage 2.1 room smoke passed all interrupt/noise/backchannel cases.
- Stage 2.2 room smoke with `--livekit-participant-identity manson` is `6/6`;
  `normal_user_turn_when_agent_idle_001` now passes and proves the full
  `brain_request_sent -> brain_first_delta -> tts_first_audio` path.
- `cough_noise_then_ambient_speech_holds_001` now passes by asserting
  `forbid_actions: ["cancel"]`; both `hold` and `rollback` are acceptable
  false-interrupt outcomes.
- Stage 2.3 room smoke is `6/6` and command output is clean: no
  `acquire on a closed pool` and no unhandled TTS task exception.
- Stage 2.5 default observe-mode room smoke is `5/6`; it exposed that a cough
  with no STT transcript cannot trigger the weak-signal guard, so later ambient
  speech can still cancel when `attention.enforce=false`.
- Stage 2.5 enforce-mode room smoke is `3/3` for the focused suite:
  playback-time ambient speech does not cancel, hard-stop still cancels, and
  missing client state preserves the legacy path.
- The temporary enforce-mode worker was stopped after the smoke test and the
  supervisor-managed default channel worker was restored.

### Current Working Hypothesis

The previous normal-idle failure was caused by the benchmark participant not
following the runtime identity contract. The channel resolver requires
`participant.metadata.kind` and an identity that admin can resolve. A random
`voice-bench-*` identity cannot resolve to an active user/agent.

After passing a real dogfood user identity, normal idle reaches:

```text
brain_request_sent
brain_first_delta
tts_provider_first_audio
tts_first_audio
```

The remaining product question is now rollout tuning rather than broken brain
startup: full ambient speech while the agent is speaking is blocked in
enforce-mode when fresh client playback state is present, but real Web dogfood
must verify missed-interrupt risk before enabling this by default.

### Next Execution Queue

1. Run real Web-client dogfood with `turn_policy.attention.enforce=true`,
   focusing on background speech, hard-stop latency, missed normal turns, and
   stale/missing client state.
2. Consolidate scattered keyword/lexicon checks into one configurable,
   observable intent/profile layer.
3. Add richer decision observability for every interrupt path:
   transcript evidence, attention admission, EOT confidence, final action, and
   latency.
4. Only after real telemetry shows deterministic gates are insufficient,
   evaluate a tiny local classifier in shadow mode. Do not put a remote LLM in
   the realtime hot path.

## Metrics

Minimum metrics to track per run:

- false duck count while agent is speaking
- false cancel count while agent is speaking
- real interrupt latency
- false-interrupt rollback latency
- turn commit count from background speech
- empty-ASR VAD-end count
- user-visible audio gap during false interrupt
- backchannel rollback rate
- agent echo / self-transcription incidents
- first audio latency regression

Suggested SLOs:

```text
hard interrupt p95:             <= 150 ms for explicit interrupt/PTT
duck-to-decision p95:           <= 500 ms for semantic candidate
false duck reduction:           >= 50% vs baseline in noisy fixtures
false cancel:                   0 in backchannel/noise fixtures
background turn commits:        0 in non-directed background fixtures
first-audio latency regression: <= +50 ms p95
```

## Stage 0: Freeze Baseline

### Purpose

Create the comparison point before changing behavior.

### Implementation

No behavior change.

Add or standardize benchmark scenarios:

- agent speaking + nearby background conversation
- agent speaking + TV/podcast speech
- agent speaking + user says backchannel: "嗯", "好的", "知道了"
- agent speaking + real hard stop: "停一下", "别说了", "打住"
- agent speaking + correction: "不是", "等一下", "我刚才说错了"
- idle room + background conversation
- wake/follow-up real user turn
- agent self-echo fixture if available

### Tests

- Existing `turn_policy` unit tests.
- Existing `OutputController` tests.
- Existing headless benchmark runner.
- Real LiveKit smoke with timeline enabled.

### Acceptance

- Produce a markdown report with current counts and latencies.
- Save raw timeline JSONL / benchmark artifacts.
- No product behavior change.

## Stage 1: Client Audio Hygiene and Explicit Signals

### Purpose

Improve realtime input quality and provide explicit low-latency user-intent signals without changing server-side policy yet.

### Client Work

Web:

- Enable browser audio constraints:
  - `echoCancellation`
  - `noiseSuppression`
  - `autoGainControl` with A/B validation
- Add explicit interrupt/PTT signal if missing.
- Emit lightweight `client.audio_state` data messages:

```json
{
  "type": "client.audio_state",
  "input_mode": "auto",
  "ptt": false,
  "manual_interrupt": false,
  "playback_state": "agent_speaking",
  "rms": 0.42,
  "snr_hint": 0.61
}
```

ESP32:

- Prefer AFE/AEC/NS when available.
- Emit minimal audio state:
  - wake word detected
  - button/PTT
  - local playback state
  - mic mute state

### Server Work

- Add a data message receiver in `StreamingPipeline` or a small adjacent adapter.
- Store the latest per-participant `ClientAudioState`.
- Do not let these hints affect behavior yet, except explicit manual interrupt if it is already a product behavior.

### Tests

- Unit test parsing and state expiry.
- LiveKit room smoke for receiving data messages.
- Web smoke for audio constraints and signal send.

### Acceptance

- No regression in baseline behavior.
- Client state appears in timeline/debug artifacts.
- Explicit interrupt remains fast.

## Stage 2: AttentionAdmission Without Speaker Verification

### Purpose

Prevent weak/background speech from entering the duck/interruption path while preserving real interrupts.

### Implementation

Add `AttentionAdmission` under `eidolon.livekit.agent.turn_policy`.

It should be pure or nearly pure, similar to `InterruptDecider`.

Inputs:

- current pipeline state
- latest client audio state
- follow-up window
- wake word / PTT / manual interrupt
- cooldown
- latest VAD state
- optional transcript preview

Output:

```text
IGNORE
OBSERVE
DUCK_AND_DECIDE
HARD_INTERRUPT
```

Integrate only at the existing admission point:

```text
StreamingPipeline._on_user_state_changed(new="speaking")
  if agent state is SPEAKING:
    admission = AttentionAdmission.decide(...)
    HARD_INTERRUPT   -> existing interrupt path
    DUCK_AND_DECIDE  -> existing _duck_and_arm_timeout()
    OBSERVE/IGNORE   -> do not duck
```

Do not replace `_run_eot_check`.
Do not bypass `InterruptDecider` once ducking has started.

### Initial Rules

Hard:

- manual interrupt
- PTT
- button interrupt
- wake word

Likely:

- follow-up window active plus strong speech signal
- first transcript carries hard-stop/correction/topic-switch lexical signal

Weak:

- VAD only, no directed signal, agent is speaking
- client says playback is active and there is no explicit interrupt
- rapid VAD flapping inside cooldown

### Tests

- Unit tests for each admission output.
- Streaming tests that background speech does not call `_duck_and_arm_timeout`.
- Real interrupt tests still call existing cancel path.
- Backchannel tests still rollback when admitted.

### Benchmarks

Run Stage 0 fixtures.

### Acceptance

- False duck count drops significantly in background fixtures.
- Real interrupt p95 stays within budget.
- No increase in missed real interrupt dogfood reports.

## Stage 3: Echo and Self-Transcription Suppression

### Purpose

Reduce agent self-echo and playback-caused false admission.

### Implementation

Use existing state and text signals:

- client `playback_state=agent_speaking`
- `OutputController.played_seconds`
- current/pushed TTS text if available
- STT interim text

Add a cheap similarity check:

```text
if agent is speaking
and STT interim is highly similar to recently pushed TTS text
and no hard attention signal:
  classify as echo/self-transcription
  rollback or ignore
```

Keep it simple:

- normalized substring / token overlap first
- no embedding model in hot path

### Tests

- TTS text echoed into STT interim should not cancel.
- Real correction with partial overlap should still cancel if lexical hard signal exists.

### Benchmarks

- Add echo fixtures.
- Run all previous fixtures.

### Acceptance

- Echo incidents decrease.
- No real hard-stop regression.

## Stage 2.1 Execution Record: Transcript Evidence Gate

Date: 2026-06-06

Status: accepted for deterministic unit, policy, and headless benchmarks.

### Purpose

Fix the real-room failure where a hard interrupt "别说了" first arrived as a
short latin-only interim "If". The old first-signal path treated any interim
with `len >= min_interim_chars` as `normal_interrupt` and cancelled before the
later correct hard-stop transcript arrived.

### Design Decision

Do not add an LLM or remote model to the realtime hot path yet.

Instead, add a deterministic `TranscriptEvidenceGate`:

- Hard controls still keep their fast path:
  - PTT/manual interrupt
  - hard-stop/topic-switch/correction lexicon matches
- Reversible actions may still happen early.
- Irreversible `normal_interrupt -> cancel` now needs transcript evidence.
- Short latin-only interim artifacts in a Chinese room, such as `If`, become
  `HOLD` instead of `CANCEL`.
- Substantive CJK interim text can still cancel quickly.
- Final transcripts count as sufficient evidence.

Configuration:

```yaml
turn_policy:
  interrupt:
    transcript_evidence_gate_enabled: true
    min_normal_interim_cjk_chars: 3
    latin_artifact_hold_max_chars: 4
```

### Changes

- Added `eidolon.livekit.agent.turn_policy.evidence.TranscriptEvidenceGate`.
- Wired `is_final` into `TurnPolicyRuntime.decide_from_transcript`.
- Updated `InterruptDecider` so the first-signal normal-interrupt cancel path
  consults transcript evidence before cancelling.
- Updated the policy benchmark runner to process multiple interims per step,
  so it can represent `["If", "别说了"]` instead of only the first interim.
- Added benchmark case `hard_stop_after_short_latin_artifact_001`.

### Validation

Focused tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_interrupt_decider.py \
  eidolon/livekit/tests/agent/test_first_signal_trigger.py \
  eidolon/livekit/tests/agent/test_transcript_evidence.py \
  eidolon/livekit/tests/agent/test_attention_admission.py \
  eidolon/livekit/tests/agent/test_timeline.py -q
```

Result:

```text
81 passed
```

Config + benchmark infrastructure:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/common/test_effective_config.py \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
34 passed
```

Policy benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-1-transcript-evidence-policy-20260606 \
  --repeat 3 \
  --skip-preflight
```

Result:

```text
pass_rate: 6/6
repeats: 3
hard_stop_after_short_latin_artifact_001: 3/3
interrupt_decision_ms p50/p95/max: 80/160/160
```

Headless benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-1-transcript-evidence-headless-20260606 \
  --repeat 3 \
  --skip-preflight
```

Result:

```text
pass_rate: 6/6
repeats: 3
hard_stop_after_short_latin_artifact_001: 3/3
```

Artifacts:

```text
benchmarks/runs/stage2-1-transcript-evidence-policy-20260606/policy/report.md
benchmarks/runs/stage2-1-transcript-evidence-headless-20260606/headless/report.md
```

### Observations

- This keeps hard-stop latency fast when the transcript is immediately correct.
- The artifact case costs one interim hop: policy latency moves from 80 ms to
  160 ms in that fixture, still inside the 500 ms budget.
- This is a better first step than adding a model: it is deterministic,
  explainable, cheap, and removes the most dangerous hard-coded behavior.
- A model can still be added later as a shadow classifier if collected timeline
  evidence shows rules are insufficient.

## Stage 2.2 Execution Record: Room Smoke Evidence Stabilization

Date: 2026-06-06

Status: accepted.

### Purpose

Turn the remaining real-room `5/6` result into useful evidence instead of a
muddy failure. The failing case is `normal_user_turn_when_agent_idle_001`.

### Findings

- The interrupt, noise, artifact, and backchannel room cases pass.
- `normal_user_turn_when_agent_idle_001` gets real STT final text:
  `帮我详细介绍一下这个方案。`
- The worker's long-lived timeline shows this turn reaches:
  - `speech_stopped_at`
  - `turn_committed_at`
  - `llm_started_at`
  - `tts_request_started_at`
- The same timeline does not show:
  - `brain_request_sent_at`
  - `brain_first_delta_at`
  - `tts_provider_first_audio_at`
  - `tts_first_audio_at`
- Channel logs also showed a Bailian TTS invalid-text error during the room
  smoke window:
  `[InvalidParameter] Please ensure input text is valid.`
- After adding earlier `brain_request_started` observability, the normal-idle
  failure was classified as a benchmark identity-contract issue:
  the random `voice-bench-*` participant had no `participant.metadata.kind` and
  could not resolve through admin.
- A valid room participant must provide both:
  - `participant.metadata.kind`: `user` or `device`
  - `participant.identity`: an admin-resolvable user/device id

### Changes

- Added `--livekit-timeline-flush-grace-sec` to `scripts/bench_voice.py`.
  Default is 5 seconds, so `session_closed` timeline flushes have time to reach
  the captured per-run `turn_timeline.jsonl`.
- Moved the `brain_request_started` provider event in
  `EidolonAgentGrpcLlmStream._run` to the beginning of adapter execution. This
  makes token/session/conversation-id startup visible instead of invisible.
  `brain_request_sent` still means the gRPC `StartTurn` was actually written.
- Applied `conn_options.timeout` to both gRPC session open and `StartTurn`.
  If runtime token/admin resolve or gRPC startup hangs, the LLM stream now
  fails fast with `APIConnectionError` instead of silently waiting for the
  outer room timeout.
- Added `--livekit-participant-identity` and `--livekit-participant-kind` to
  the room benchmark runner. Cases that expect agent replies now fail early
  with a clear message if no resolvable identity is supplied.
- Added LiveKit participant metadata to the benchmark access token, so the
  room smoke path matches web/ESP32 runtime tagging.
- Changed livekit-room preflight in `eidolon_agent` brain mode to only preflight
  STT/TTS. Dynamic brain token resolution requires a real LiveKit room
  participant and is proven by the room timeline instead.
- Added `weak_signal_followup_hold_ms` to the interrupt policy. Any short
  weak-signal hold can suppress a following `normal_interrupt` cancel candidate
  inside the configured window. Strong intents still pass through.
- Added `forbid_actions` to benchmark expectations so false-interrupt cases can
  assert product-critical behavior ("must not cancel") without overfitting to
  whether the runtime internally chose `hold` or `rollback`.

### Validation So Far

Focused tests before the gRPC adapter observability change:

```text
./.venv/bin/python -m pytest eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
29 passed
```

Interrupt/evidence tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_interrupt_decider.py \
  eidolon/livekit/tests/agent/test_first_signal_trigger.py \
  eidolon/livekit/tests/agent/test_transcript_evidence.py -q
```

Result:

```text
60 passed
```

Room smoke with 2-second timeline grace still captured `5/6`; the long-lived
worker timeline confirmed the normal-idle turn exists but was flushed later
than the per-run capture.

After moving `brain_request_started` earlier and using 5-second timeline grace,
room smoke captured all 6 worker timeline records. The normal-idle case now
shows:

```text
turn_committed_at
llm_started_at
brain_request_started_at
tts_request_started_at
```

It still does not show:

```text
brain_request_sent_at
brain_first_delta_at
tts_provider_first_audio_at
tts_first_audio_at
```

This narrows the remaining normal-idle issue to gRPC session/token/resolve or
`StartTurn` startup before the request is written.

Adapter + benchmark/timeline tests after timeout wiring:

```text
./.venv/bin/python -m pytest eidolon/livekit/tests/agent/test_eidolon_agent_grpc_llm.py -q
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py \
  eidolon/livekit/tests/agent/test_timeline.py -q
```

Result:

```text
19 passed
42 passed
```

Additional validation after participant identity and preflight fixes:

```text
./.venv/bin/python -m pytest eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_eidolon_agent_grpc_llm.py \
  eidolon/livekit/tests/agent/test_timeline.py -q
```

Result:

```text
30 passed
32 passed
```

Room benchmark identity guard:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-identity-guard-check-20260606 \
  --skip-preflight \
  --livekit-room-timeout-sec 1 \
  --livekit-agent-ready-timeout-sec 1
```

Result:

```text
livekit_room cases expecting agent replies require --livekit-participant-identity
```

Real-room smoke with a resolvable user:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-room-smoke-with-user-identity-20260606 \
  --livekit-participant-identity manson \
  --livekit-participant-kind user \
  --livekit-room-timeout-sec 45 \
  --livekit-agent-ready-timeout-sec 12 \
  --livekit-timeline-flush-grace-sec 5
```

Result:

```text
pass_rate: 5/6
normal_user_turn_when_agent_idle_001: PASS
hard_stop_after_short_latin_artifact_001: PASS
owner_hard_stop_while_agent_speaking_001: PASS
ambient_normal_speech_currently_interrupts_001: PASS
short_backchannel_rolls_back_001: PASS
cough_noise_then_ambient_speech_currently_interrupts_001: FAIL because expected cancel but observed hold
```

Normal-idle timeline now proves the full chain:

```text
conversation_id: livekit:manson:voice-bench-normal_user_turn_when_agent_idle_001-...
commit_to_brain_request_started: 622 ms
brain_request_to_first_delta: 435 ms
commit_to_tts_first_audio: 1428 ms
brain last_event: brain_done
tts last_event: tts_provider_first_audio
```

### Next

Weak-signal guard validation:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py \
  eidolon/livekit/tests/agent/test_turn_policy_runtime.py \
  eidolon/livekit/tests/common/test_effective_config.py -q
```

Result:

```text
40 passed
```

Policy/headless final baseline:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-forbid-actions-policy-20260606 \
  --repeat 3

./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-forbid-actions-headless-20260606 \
  --repeat 3
```

Result:

```text
policy: 6/6, 3 repeats
headless: 6/6, 3 repeats
```

Final real-room smoke:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-forbid-actions-room-smoke-20260606 \
  --livekit-participant-identity manson \
  --livekit-participant-kind user \
  --livekit-room-timeout-sec 45 \
  --livekit-agent-ready-timeout-sec 12 \
  --livekit-timeline-flush-grace-sec 5
```

Result:

```text
pass_rate: 6/6
cough_noise_then_ambient_speech_holds_001: PASS, action=hold
normal_user_turn_when_agent_idle_001: PASS
real_call_verified: true for all cases
```

### Remaining Follow-Up

- Keep `ambient_normal_speech_currently_interrupts_001` as the remaining
  known-gap case: complete non-directed ambient speech still cancels today.
- Run dogfood with `turn_policy.attention.enforce=true`.

## Stage 2.3 Execution Record: Bailian TTS Pool Shutdown Stabilization

Date: 2026-06-06

Status: accepted.

### Purpose

Fix the teardown/retry noise observed during real-room smoke:

```text
BailianTTSPool acquire on a closed pool
Task exception was never retrieved
no audio frames were pushed for text: 你好，测试。
```

The pool's intent is still correct: keep warm WebSocket connections ready so
cancel/reply cycles do not pay full provider handshake latency. The bug was in
lifecycle boundaries, not in the prewarming strategy.

### Findings

- `TTSConnectionPool.shutdown()` waited for in-flight refill factories instead
  of canceling them. During shutdown, prewarming should stop immediately.
- `BailianTTS.shutdown()` could close the pool while a framework-owned
  `SynthesizeStream` retry was still unwinding. A later retry then attempted to
  acquire from a closed pool.
- `TtsStage.synthesize()` wrapped LiveKit's `ChunkedStream` but did not close
  the underlying stream when the consumer stopped early, leaving provider task
  exceptions unconsumed.
- The text `你好，测试。` came from benchmark preflight, not from the user room
  interaction.

### Changes

- Pool shutdown now cancels in-flight refill tasks, waits for tracked cleanup
  tasks, then drains warm connections.
- `in_flight_refills` now reports only real refill tasks, not dispose tasks.
- Bailian TTS sets a `_closing` flag during shutdown. Late streams during
  shutdown end cleanly without acquiring a new pool connection.
- `TtsStage.synthesize()` now always closes the underlying LiveKit stream and
  consumes any completed task exception.
- Benchmark TTS preflight explicitly closes the stage synthesize generator.

### Validation

Unit tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py \
  eidolon/livekit/tests/pipeline/test_tts_stage.py \
  eidolon/livekit/tests/tts/test_pool.py \
  eidolon/livekit/tests/tts/bailian/test_tts_shutdown.py -q
```

Result:

```text
54 passed
```

Real-room smoke:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-2-tts-pool-final-room-smoke-20260606 \
  --livekit-participant-identity manson \
  --livekit-participant-kind user \
  --livekit-room-timeout-sec 45 \
  --livekit-agent-ready-timeout-sec 12 \
  --livekit-timeline-flush-grace-sec 5
```

Result:

```text
pass_rate: 6/6
command output: no closed-pool error, no unhandled TTS task exception
```

## Stage 2.4 Execution Record: STT Lifecycle Audit

Date: 2026-06-06

Status: accepted for provider lifecycle tests.

### Purpose

Check whether the Bailian TTS pool shutdown issue has an equivalent in STT.

### Findings

- STT does not currently use a TTS-style multi-connection prewarmed pool.
- `SttStage.recognize_streaming()` already closes provider streams in a
  `finally` block.
- SenseTime STT uses one persistent WebSocket and has explicit
  `disconnect()` / `_force_close()` coverage for warmup, shutdown, reconnect,
  cleanup timeout, and recovery.
- Bailian FunASR opens per-stream/per-recognize WebSockets. It did not have the
  same closed-pool acquire failure, but two abnormal paths were too loose:
  handshake failure after `_ws` was assigned but before `_connected=True`, and
  batch/stream failure before background receive/send tasks fully unwound.

### Changes

- `BailianConnectionManager.close()` now closes an assigned WebSocket even if
  the run-task handshake never reached `_connected=True`.
- Bailian batch `_recognize_impl()` now cancels/awaits the receive task and
  closes the connection from a `finally` block.
- Bailian streaming `_run()` now cancels/awaits send/receive tasks, closes the
  connection, and stops the VAD gate from a `finally` block across normal,
  retryable-error, and cancellation paths.

### Validation

Bailian STT and transcript gate:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/stt/bailian/test_stt_lifecycle.py \
  eidolon/livekit/tests/stt/bailian/test_stt.py \
  eidolon/livekit/tests/stt/bailian/test_stt_retry.py \
  eidolon/livekit/tests/stt/test_transcript_gate.py -q
```

Result:

```text
31 passed
```

SenseTime STT persistent/reliability:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/stt/sensetime/test_stt_persistent.py \
  eidolon/livekit/tests/stt/sensetime/test_stt_reliability.py -q
```

Result:

```text
32 passed
```

## Stage 2.5 Execution Record: Attention Enforce Smoke

Date: 2026-06-06

Status: accepted for policy benchmark and focused LiveKit room smoke; pending
real Web-client dogfood before product rollout.

### Purpose

Validate the concrete V1 product fix for the noisy-room pain point:

```text
When Eidolon is speaking, fresh client playback state plus non-directed
ambient speech should not cancel the agent.
Explicit hard stop should still cancel quickly.
Missing client state should preserve the legacy path.
```

### Changes

- Added `benchmarks/cases/attention_admission_enforced.yaml` as an explicit
  enforce-mode suite. It is skipped by default benchmark discovery and must be
  run intentionally with `--cases ... --attention-enforce`.
- Added `client_playback_state`, `client_ptt`, `client_manual_interrupt`, and
  `client_mic_muted` fields to benchmark user steps.
- Extended the policy runner to call `TurnPolicyRuntime.admit_attention()`
  before `InterruptDecider`. With `attention.enforce=true`, `ignore`/`observe`
  decisions skip the interrupt path and keep the final action as `none`.
- Added `--attention-enforce` to `scripts/bench_voice.py` so policy benchmark
  runs can enable enforcement without editing `config/settings.yaml`.
- Updated the LiveKit room runner to publish `client.audio_state` repeatedly
  during captured audio, not only once at step start. This avoids false stale
  state during longer utterances.
- Added `client_playback_state: none` for the no-client-state case to prove
  the legacy path still works.

### Validation

Focused tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py \
  eidolon/livekit/tests/agent/test_attention_admission.py -q
```

Result:

```text
42 passed
```

Enforce-mode policy benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_enforced.yaml \
  --run-id stage2-5-attention-enforce-policy-v3-20260606 \
  --repeat 3 \
  --skip-preflight \
  --attention-enforce
```

Result:

```text
pass_rate: 3/3
repeats: 3
```

Baseline policy regression:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-5-attention-enforce-baseline-policy-20260606 \
  --repeat 3 \
  --skip-preflight
```

Result:

```text
pass_rate: 6/6
repeats: 3
```

Default observe-mode room smoke:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-5-post-stt-room-smoke-20260606 \
  --livekit-participant-identity manson \
  --livekit-participant-kind user \
  --livekit-room-timeout-sec 45 \
  --livekit-agent-ready-timeout-sec 12 \
  --livekit-timeline-flush-grace-sec 5
```

Result:

```text
pass_rate: 5/6
```

The failed case was `cough_noise_then_ambient_speech_holds_001`. The cough did
not produce a transcript, so the weak-signal guard had no evidence to hold.
The later ambient speech then reached the legacy cancel path while
`attention.enforce=false`.

Focused enforce-mode room smoke:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_enforced.yaml \
  --run-id stage2-5-attention-enforce-room-smoke-v2-20260606 \
  --livekit-participant-identity manson \
  --livekit-participant-kind user \
  --livekit-room-timeout-sec 45 \
  --livekit-agent-ready-timeout-sec 12 \
  --livekit-timeline-flush-grace-sec 5 \
  --lenient-realcall
```

Result:

```text
pass_rate: 3/3
```

Room smoke notes:

- Ambient playback speech produced `attention_admission.action=observe`,
  `client_state_used=true`, `enforced=true`, and no cancel action.
- Hard-stop produced `timeline_actions=cancel` and
  `timeline_intents=hard_stop`.
- Missing client state produced `attention_admission.action=duck_and_decide`
  with reason `no_client_state`, preserving the legacy cancel path.
- `--lenient-realcall` was used because this interrupt-only suite has no normal
  idle brain turn, so it does not emit brain RPC proof by design.
- The temporary enforce-mode worker was stopped after the run, and the
  supervisor-managed default channel worker was restored.

### Observations

- This directly addresses the main V1 noisy-room failure mode when a fresh
  `client.audio_state` signal is present.
- The fix is low-latency and deterministic; no LLM or remote classifier enters
  the hot interrupt path.
- The remaining risk is product-level: stale/missing client state and real Web
  microphone behavior can still differ from synthetic room playback.

### Post-Reflection Refinements

After reviewing the Stage 2.5 changes for architecture drift, the following
cleanup was applied:

- `TurnPolicyRuntime` no longer owns weak-signal timing inline. The follow-up
  hold logic is isolated in a private stabilizer, keeping runtime closer to an
  adapter/orchestrator.
- Weak-signal detection no longer treats every `HOLD` as weak evidence. Only
  explicit transcript-evidence, better-transcript, noise, and backchannel hold
  reasons can arm the follow-up window.
- Policy benchmark decisions now use one composite record per interim:
  `attention_admission` plus optional `decision`, instead of two adjacent
  records for the same interim.
- LiveKit room benchmark `client.audio_state` now carries step-level
  `client_ptt`, `client_manual_interrupt`, and `client_mic_muted` flags through
  the real data channel.
- Attention admission timeline attrs now preserve an
  `attention_admission_events` history list while keeping the latest
  `attention_admission` attr for compatibility.
- Transcript-time attention admission now prefers the `speaker_id`-matched
  `client.audio_state` before falling back to the freshest state. This keeps
  V1 behavior unchanged for single-client rooms while preventing another
  participant's newer state from steering the current speaker's admission.

Validation after cleanup:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_turn_policy_runtime.py \
  eidolon/livekit/tests/agent/test_attention_admission.py \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
49 passed
```

Policy benchmark regression:

```text
stage2-5-reflection-baseline-policy-20260606: 6/6, 3 repeats
stage2-5-reflection-enforce-policy-20260606: 3/3, 3 repeats
stage2-5-speaker-binding-baseline-policy-20260606: 6/6, 3 repeats
stage2-5-speaker-binding-enforce-policy-20260606: 3/3, 3 repeats
```

Remaining architecture item:

- VAD-level `user_state=speaking` still has no participant identity in the
  current LiveKit event type, so the first no-transcript admission continues to
  use freshest-state fallback. Transcript-time admission is now speaker-bound.

## Stage 4: Owner Voice Profile as Soft Attention Signal

### Purpose

Use the owner's voiceprint to reduce non-owner speech admission without making voiceprint a hard lock.

### Existing Assets

FireRed pVAD already includes ECAPA speaker embedding machinery:

- `SpeakerEmbExtractor`
- `PvadProcessor.update_speaker_embedding`
- `VADStream.update_speaker`

V1 should reuse these assets rather than adding an unrelated speaker stack.

### Implementation

Add a `SpeakerProfile` concept:

```text
user_id
embedding_centroid
sample_count
accept_threshold
reject_threshold
updated_at
```

Enrollment:

- Admin/Web flow reads 3-5 short phrases.
- Store embeddings server-side.
- Allow deleting/re-enrolling.

Hot path:

- Collect 0.8-1.5s speech window.
- Extract embedding asynchronously.
- Compute cosine similarity to owner centroid.
- Feed result into `AttentionAdmission`.

Policy:

```text
owner match strong -> raises attention score
owner reject       -> lowers attention score
unknown            -> no hard decision
```

Do not block PTT, wake word, or hard stop just because voiceprint is uncertain.

### Tests

- Unit tests for score thresholds and unknown handling.
- Enrollment storage/load tests.
- Graceful degradation when ECAPA/SpeechBrain is unavailable.

### Benchmarks

- Owner vs non-owner fixtures.
- Short utterance fixture.
- Noisy/low-SNR fixture.

### Acceptance

- Non-owner background speech admission drops.
- Owner real interrupts are not missed in dogfood.
- Speaker extraction does not add hot-path latency to explicit interrupt.

## Stage 5: Optional STT Gate Tuning

### Purpose

Reduce STT cost and noise exposure after admission behavior is stable.

### Implementation

Use existing `BailianSTTConfig.gate_enabled` and `_gate.py`.

Tune:

- preroll
- tail window
- VAD high/low thresholds
- RMS threshold

### Tests

- First-word preservation.
- Short utterances.
- False-negative VAD scenarios.

### Acceptance

- Cost/noise reduction without first-word loss or quality regression.

## Stage 6: Product Modes

### Purpose

Expose stable behavior choices to users.

Modes:

```text
Auto
  natural conversation, attention admission enabled

Focus / PTT
  office/noisy environment, explicit input emphasized

Quiet Companion
  conservative speaking-state interruption, low false duck tolerance
```

### Tests

- Config loading.
- Mode-to-policy mapping.
- Web/ESP32 UI smoke.

### Acceptance

- Users can recover from noisy environments without changing config files.

## Rollback Strategy

Each stage should have a config kill switch.

Suggested toggles:

```yaml
turn_policy:
  attention:
    enabled: false
    speaker_profile_enabled: false
    echo_suppression_enabled: false
```

Default rollout:

1. Land disabled or observe-only.
2. Enable in dogfood.
3. Promote to default after benchmark and real-room acceptance.

## Decision Log

- Duck/mute is retained as the output-control mechanism, but it must not be triggered by raw VAD alone.
- Voiceprint is useful but soft; it is not an authentication lock in V1.
- Client signals are hints; server policy remains the source of truth.
- LLM is not part of realtime admission or interruption.

## Stage 0 Execution Record

Date: 2026-06-06

Status: accepted for deterministic local baseline.

### Changes

Added a focused Stage 0 benchmark suite:

```text
benchmarks/cases/attention_admission_baseline.yaml
```

Cases:

- `owner_hard_stop_while_agent_speaking_001`
- `ambient_normal_speech_currently_interrupts_001`
- `short_backchannel_rolls_back_001`
- `cough_noise_then_ambient_speech_holds_001`
- `normal_user_turn_when_agent_idle_001`

`ambient_normal_speech_currently_interrupts_001` remains tagged `known_gap`.
It intentionally freezes current behavior where complete non-directed speech
during agent speaking is admitted as `normal_interrupt` and cancels/interrupts.
`cough_noise_then_ambient_speech_holds_001` is now a Stage 2.2 desired-behavior
regression case: short noise or weak ambient evidence should hold instead of
canceling. In benchmark terms it uses `forbid_actions: ["cancel"]` so both
`hold` and `rollback` count as acceptable false-interrupt outcomes.

### Validation

Schema/unit test:

```text
./.venv/bin/python -m pytest eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
28 passed
```

Policy benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage0-attention-admission-policy-3x-20260606 \
  --repeat 3
```

Result:

```text
pass_rate: 5/5
repeats: 3
per-case pass_rate: 100%
interrupt_decision_ms p50/p95/max: 80/80/80
```

Artifacts:

```text
benchmarks/runs/stage0-attention-admission-policy-3x-20260606/policy/report.md
benchmarks/runs/stage0-attention-admission-policy-3x-20260606/policy/metrics.json
benchmarks/runs/stage0-attention-admission-policy-3x-20260606/policy/report.html
```

Headless benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage0-attention-admission-headless-3x-20260606 \
  --repeat 3
```

Result:

```text
pass_rate: 5/5
repeats: 3
per-case pass_rate: 100%
elapsed_ms p50/p95/max: 1699/2178.6/2180
user_final_count p50/p95/max: 1/2/2
agent_message_count p50/p95/max: 0/1/1
```

Artifacts:

```text
benchmarks/runs/stage0-attention-admission-headless-3x-20260606/headless/report.md
benchmarks/runs/stage0-attention-admission-headless-3x-20260606/headless/metrics.json
benchmarks/runs/stage0-attention-admission-headless-3x-20260606/headless/report.html
```

### Observations

- Current policy correctly preserves explicit owner-style hard stop: `别说了` -> `cancel`, `hard_stop`.
- Current policy correctly avoids hard cancel for short backchannel: `好的` -> `rollback`, `backchannel`.
- Current policy currently treats a normal complete utterance during agent speaking as a real interruption: `那它的主要风险是什么` -> `cancel`, `normal_interrupt`. This is the main V1 attention-admission gap.
- Short noise alone is held, but if normal speech follows, the normal speech still cancels. Admission needs to keep observing weak/noisy starts without immediately exposing the output path to false duck/cancel.
- Headless 3x emitted mock-harness async logs such as `_MockSTTStream is closed`; final runner exit code was 0 and all cases passed. Treat this as harness noise unless it becomes a failing/flaky case.

### Not Yet Run

- Real provider component benchmark for this new suite.
- Real LiveKit room smoke with timeline enabled.
- Live dogfood in a noisy physical room.

## Stage 1 Execution Record

Date: 2026-06-06

Status: accepted for Web + server observe-only implementation.

### Changes

Channel:

- Added `eidolon.livekit.agent.client_audio_state`.
- Added `StreamingPipeline` room data observer for topic `client.audio_state`.
- Stores latest per-participant `ClientAudioState` in `_client_audio_states`.
- Copies the latest received state into turn timeline attrs as `client_audio_state`.
- Does not alter VAD, STT, ducking, interruption, or commit behavior.

Web client:

- Added `useClientAudioStatePublisher`.
- Main realtime room and companion room now publish `client.audio_state` every 500 ms while connected.
- Payload includes:
  - `input_mode`
  - `ptt`
  - `manual_interrupt`
  - `playback_state`
  - `mic_muted`
  - `client_ts_ms`
- Main realtime room and companion room now request browser capture with:
  - `echoCancellation: true`
  - `noiseSuppression: true`
  - `autoGainControl: true`

No keyword/lexicon changes were made in this stage.

### Validation

Channel tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_timeline.py \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
41 passed
```

Web build/type check:

```text
npm run build
```

Result:

```text
Compiled successfully
```

Policy benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage1-client-audio-state-policy-20260606
```

Result:

```text
pass_rate: 5/5
interrupt_decision_ms p50/p95/max: 80/80/80
```

Headless benchmark:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage1-client-audio-state-headless-20260606
```

Result:

```text
pass_rate: 5/5
elapsed_ms p50/p95/max: 1705/2162.4/2181
```

Artifacts:

```text
benchmarks/runs/stage1-client-audio-state-policy-20260606/policy/report.md
benchmarks/runs/stage1-client-audio-state-headless-20260606/headless/report.md
```

### Observations

- Stage 1 did not change the frozen attention baseline behavior.
- Browser capture constraints and data-channel hints are now available as inputs for Stage 2.
- The headless mock harness again emitted `_MockSTTStream is closed` async logs, but final runner exit code was 0 and all cases passed.

### Not Yet Run

- ESP32 client `client.audio_state` publisher.
- Real LiveKit room smoke confirming Web packets appear in timeline JSONL.
- Live dogfood in a noisy physical room.

## Stage 2 Execution Record

Date: 2026-06-06

Status: accepted for deterministic unit + observe-only baseline regression tests.

### Changes

Added `AttentionAdmission` under `eidolon.livekit.agent.turn_policy`.

Added `client.audio_state` publishing to the real `livekit_room` benchmark participant,
so room smoke tests now exercise the same data-channel hint path as Web/ESP32
clients instead of only publishing microphone audio.

Behavior:

- If no fresh `client.audio_state` exists, preserve the old duck/interrupt path.
- If the client reports `playback_state=agent_speaking` and there is no direct signal, classify it as `observe`.
- If the client reports `ptt` or `manual_interrupt`, admit as `hard_interrupt`.
- If transcript preview contains an existing hard-stop intent, admit as `hard_interrupt`.
- If transcript preview contains existing topic-switch/correction intent, admit to the existing duck/EOT/decider path.
- No keyword/lexicon entries were changed.

Rollout safety:

- `turn_policy.attention.enabled=true` means decisions are computed and written to timeline.
- `turn_policy.attention.enforce=false` by default means those decisions do not change product behavior yet.
- Enforced suppression is available for dogfood/A-B by setting `turn_policy.attention.enforce=true`.
- This was corrected after self-review: the first Stage 2 patch made the new admission behavior active by default, which was too aggressive for a realtime interrupt path.

Configuration:

```yaml
turn_policy:
  attention:
    enabled: true
    enforce: false
    client_state_max_age_ms: 2000
    require_direct_signal_during_playback: true
    ignore_when_mic_muted: true
```

Integration point:

```text
StreamingPipeline._on_user_state_changed(new="speaking")
  -> AttentionAdmission
     HARD_INTERRUPT   -> existing interrupt path
     DUCK_AND_DECIDE  -> existing _duck_and_arm_timeout()
     OBSERVE/IGNORE   -> no duck when enforce=true
     enforce=false    -> old duck/interrupt path, timeline only

StreamingPipeline._on_user_transcribed(...)
  -> AttentionAdmission with transcript preview
     OBSERVE/IGNORE   -> skip _run_eot_check() when enforce=true
     otherwise        -> existing EOT/InterruptDecider path
     enforce=false    -> existing EOT/InterruptDecider path, timeline only
```

This means Web clients that publish `client.audio_state` now produce `attention_admission` observability. Product behavior remains unchanged until `turn_policy.attention.enforce=true` is explicitly enabled for dogfood or A/B. Headless tests, ESP32 without publisher, or any client without fresh state keep the previous behavior in both observe-only and enforced modes.

### Validation

Focused tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/agent/test_attention_admission.py \
  eidolon/livekit/tests/agent/test_turn_policy_runtime.py \
  eidolon/livekit/tests/agent/test_timeline.py -q
```

Result:

```text
23 passed
```

Config + benchmark infrastructure tests:

```text
./.venv/bin/python -m pytest \
  eidolon/livekit/tests/common/test_effective_config.py \
  eidolon/livekit/tests/benchmarks/test_voice_benchmark.py -q
```

Result:

```text
33 passed
```

Policy baseline regression:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner policy \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-attention-admission-observe-policy-20260606 \
  --repeat 3 \
  --skip-preflight
```

Result:

```text
pass_rate: 5/5
repeats: 3
interrupt_decision_ms p50/p95/max: 80/80/80
```

Headless baseline regression:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner headless \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-attention-admission-observe-headless-20260606 \
  --repeat 3 \
  --skip-preflight
```

Result:

```text
pass_rate: 5/5
repeats: 3
elapsed_ms p50/p95/max: 1700/2179.9/2182
```

Artifacts:

```text
benchmarks/runs/stage2-attention-admission-observe-policy-20260606/policy/report.md
benchmarks/runs/stage2-attention-admission-observe-headless-20260606/headless/report.md
```

Real LiveKit room smoke:

```text
./.venv/bin/python scripts/bench_voice.py \
  --runner livekit_room \
  --cases benchmarks/cases/attention_admission_baseline.yaml \
  --run-id stage2-attention-admission-room-smoke-20260606 \
  --repeat 1 \
  --skip-preflight \
  --lenient-realcall
```

Result:

```text
pass_rate: 2/5
```

Signal verification:

```text
benchmark events include client_audio_state_published
timeline attention_admission.client_state_used: true
timeline attention_admission.enforced: false
```

Artifacts:

```text
benchmarks/runs/stage2-attention-admission-room-smoke-20260606/livekit_room/report.md
benchmarks/runs/stage2-attention-admission-room-smoke-20260606/livekit_room/turn_timeline.jsonl
```

### Observations

- The frozen baseline behavior is unchanged because Stage 2 defaults to observe-only.
- The new Web path now supplies enough signal to classify playback-time ambient speech before ducking/EOT.
- In observe-only mode, real hard-stop text and ambient normal speech still flow through the existing path.
- With `enforce=true`, the Web path can suppress playback-time ambient speech before ducking/EOT; that must be validated in real-room dogfood before enabling by default.
- Normal content while the agent is speaking becomes conservative only in enforced mode unless accompanied by explicit PTT/manual signal or hard semantic intent. This remains a deliberate V1 tradeoff until owner voiceprint / richer directed-speech signals land.
- Real room smoke proved the data-channel hint path reaches the worker timeline, but the full room suite is not stable yet.
- Real room hard-stop showed an existing hot-path weakness: the first STT interim for "别说了" arrived as "If", so the old observe-only path cancelled as `normal_interrupt` before the later correct transcript arrived.
- Real room normal idle turn timed out waiting for post-user agent audio in this run, so room-level dogfood acceptance remains open.

### Not Yet Run

- Live noisy-room dogfood.
- ESP32 publisher and hardware AEC/NS validation.
