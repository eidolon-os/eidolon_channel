# Realtime Voice Latency Analysis

Last updated: 2026-05-29

This document records the current latency diagnosis for the real
`eidolon_agent` LiveKit-room benchmark and the concrete optimization plan.

## Latest Run

```text
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/livekit_room
benchmarks/runs/stt-turn-audio-correction-room-final-20260529/dashboard-with-room.html
```

Result:

```text
5/5 passed
```

Covered cases:

- `normal_single_turn_001`
- `hard_interrupt_001`
- `topic_switch_001`
- `correction_001`
- `backchannel_001`

## Main Diagnosis

The system is not uniformly slow. The current latency profile has three
separate problems:

- interrupt P95 is pulled up by one slow semantic-interrupt case;
- normal-turn first audio is dominated by STT finalization and the gap before
  the brain request starts;
- current marks show where latency clusters, but not yet every provider-level
  streaming sub-step.

## Current Provider Segments

Latest real-room values after adding brain RPC and STT streaming marks:

```text
STT turn first audio: 45.2-98.8 ms
STT speech -> provider first partial: 207.1-357.5 ms
STT provider partial -> LiveKit interim: 0.2-0.8 ms
STT provider final -> LiveKit final: 0.5-0.7 ms
LiveKit room benchmark: 5/5 passed
```

Important interpretation:

```text
STT is a long-lived websocket stream, not a per-turn request/response call.
Therefore stream-open and stream-first-audio are diagnostic marks only.
The experience-critical STT latency is measured per turn from speech start to
provider first partial/final, then from provider event to LiveKit transcript.
```

This run shows LiveKit transcription propagation is effectively negligible
inside the process boundary. The meaningful STT delay is provider-side partial
recognition, usually a few hundred milliseconds for the current generated clips.

## Normal Turn Breakdown

For `normal_single_turn_001`:

```text
STT turn first audio: 95.4 ms
STT first audio -> provider first partial: 195.0 ms
STT speech -> provider first partial: 290.3 ms
STT provider partial -> LiveKit interim: 0.8 ms
STT final: 3191.4 ms
speech stop -> commit: 0.3 ms
commit -> brain RPC start: 1012.3 ms
brain RPC start -> request sent: 8.5 ms
brain request sent -> first delta: 258.6 ms
brain first delta -> TTS first audio: 355.9 ms
commit -> first audio: 1635.3 ms
```

The `eidolon_agent` stream itself is not the bottleneck. The request write and
first delta are fast. The larger remaining gap is before the brain RPC starts,
which currently includes waiting for final transcript / AgentSession turn
progression after Channel commits the turn.

## Completed In This Iteration

Implemented:

- `brain_request_started_at`
- `brain_request_sent_at`
- `brain_first_delta_at`
- `brain_done_at`
- `brain_rpc` timeline attrs with provider, turn id, request id, and
  conversation id

Also added STT-confusion canonicalization for hot-path semantic decisions:

- `半个...` is treated as the common `换个...` topic-switch confusion;
- `是我刚...` is treated as a correction cue for the `不是，我刚...` clip.
- ambiguous `是我` correction-prefix fragments now `HOLD` for more interim
  text instead of immediately cancelling as `normal_interrupt`.

Latest validation:

```text
ruff check: passed
pytest focused timeline/decider/classifier/benchmark/STT: 77 passed
livekit_room + eidolon_agent: 5/5 passed
```

## Optimization Plan

### 1. Add Provider-Level Brain RPC Marks

Status: done.

Add marks:

```text
brain_request_started_at
brain_first_delta_at
brain_done_at
```

Use these to distinguish:

```text
turn_committed_at -> llm_started_at
llm_started_at -> brain_request_started_at
brain_request_started_at -> brain_first_delta_at
brain_request_started_at -> brain_done_at
```

This confirmed the gap is not inside the `eidolon_agent` generation stream:

```text
brain request sent -> first delta: 137.1 ms
brain request sent -> done: 426.1 ms
```

The remaining normal-turn gap is earlier:

```text
turn committed -> brain RPC start: 931.5 ms
```

### 2. Add STT Streaming Marks

Status: done.

Goal: determine whether slow first interim is caused by audio delivery,
provider partial latency, or LiveKit transcription propagation.

Implemented marks:

```text
stt_stream_started_at
stt_ws_connected_at
stt_stream_first_audio_sent_at
stt_first_audio_sent_at
stt_flush_sent_at
stt_provider_first_partial_at
stt_provider_final_at
transcript_interim_first_at
transcript_final_at
```

Implementation note: `stt_first_audio_sent_at` is now the first provider audio
chunk after Channel has opened a user turn. The raw websocket stream's first
audio packet is recorded separately as `stt_stream_first_audio_sent_at`, because
with a long-lived stream it may be silence/pre-roll and is not a valid turn
latency anchor.

### 3. Add TTS Streaming Marks

Goal: separate provider first audio from LiveKit output playback.

Planned marks:

```text
tts_request_started_at
tts_stream_opened_at
tts_provider_first_audio_at
livekit_first_audio_published_at
playback_started_at
```

### 4. Reduce Semantic Interrupt Tail Latency

Focus case:

```text
correction_001
```

Likely strategy:

- keep hard stop as the fastest, most aggressive path;
- add a fast correction/topic-switch phrase-prefix path;
- keep single-character or ambiguous fragments in `HOLD` to avoid false cancel;
- add per-case SLO assertions for correction and topic switch.

### 5. Repeat Benchmark Sampling

Status: partially done.

Single-run results are useful for diagnosis but not enough for stable SLOs.
`--repeat N` now runs the whole suite N times and aggregates:

```text
./.venv/bin/python scripts/bench_voice.py --runner livekit_room --repeat 5 --run-id <id>
```

Implemented:

- per-case p50/p95 and `stdev` (jitter) from N repeats;
- pooled `summary.metrics` over every case x repeat sample for SLO gates;
- per-case pass rate and a `flaky` count surfaced in the dashboard findings.

Still open:

- slowest segment ranking across repeats;
- SLO trend over time;
- regression diff against a pinned baseline for the `livekit_room` layer;
- promote real-provider SLO gates from advisory to hard once >=5-repeat runs
  show stable percentiles.

## vs Industry Top-Tier

Update 2026-06-07: `tts_request_started_at` is emitted when the LiveKit TTS
stream opens. That can happen before the brain has produced any text, so
`tts_request_to_provider_first_audio_ms` is a composite "stream opened -> first
provider audio" metric, not pure provider TTFB. The provider-facing TTS TTFB SLO
now uses `tts_first_text_sent_to_provider_first_audio_ms`.

Update 2026-06-07: interrupt latency now has a split interpretation. Tier 0
hard stops are still judged from VAD/speech start to `interrupt_resolved_at`,
because a hard stop should cancel as soon as the direct stop phrase is
available. Tier 1 semantic redirects/corrections additionally use
`interrupt_started_at -> interrupt_resolved_at` as the channel execution SLO.
The full `speech_started_at -> interrupt_resolved_at` value remains important,
but it includes STT evidence availability. In the
`stable-recheck-smoke-20260607` run, topic/correction were still slow from
speech start (963 / 1154 ms), while admitted-to-resolved execution was fast
(about 136 / 141 ms). That points to early transcript quality/timing rather
than slow channel policy execution.

Reference baseline, May 2026 (platform口径 = user stop -> first agent audio,
excludes telephony network). Sources: Twilio Core-Latency 2025.11, OpenAI
gpt-realtime, LiveKit pipeline blog, Gradium/Podcastle TTS 2026.

Measured `eidolon_agent` + Bailian, real LiveKit room, `--repeat 5` (25 turns),
2026-05-30:

| Segment (metric) | Target p50/p95 | Measured p50/p95 | Verdict |
| --- | ---: | ---: | --- |
| STT final: commit -> provider final (`stt_final_after_commit_ms`) | 350 / 500 | **990 / 1042** | ❌ ~2.8x — #1 lever |
| Brain TTFT: request sent -> first delta (`brain_request_to_first_delta`) | 375 / 750 | **152 / 159** | ✅ top-tier |
| TTS stream-open -> provider audio (`tts_request_to_provider_first_audio_ms`) | diagnostic | **649 / 734** | composite: includes waiting for first brain text |
| TTS publish: provider audio -> agent audio (`tts_provider_first_audio_to_agent_audio_ms`) | 50 / 120 | **2 / 2** | ✅ negligible |
| Interrupt: VAD start -> resolved (`vad_start_to_interrupt_resolved`) | 500 / 650 | 699 / 1424 | ⚠️ p95 pulled by 1 flaky correction case |
| E2E commit -> first audio (`commit_to_tts_first_audio`) | 800 / 1100 | **1718 / 1755** | ❌ ~2.1x |

Key data-backed findings:

- The brain (`eidolon_agent`) is already **top-tier (~152ms TTFT)** — do not optimize it.
- E2E 1718ms is almost entirely two segments: **STT finalization (~990ms)** and
  the post-brain TTS path. TTS publish is ~2ms.
- New TTS split marks (`tts_first_text_sent_at`) show the old
  `tts_request_to_provider_first_audio_ms` number included waiting for the
  first brain text. In the 2026-06-07 smoke, provider-facing TTFB was
  `tts_first_text_sent_to_provider_first_audio_ms ~= 291ms`, while
  `tts_request_to_first_text_sent_ms ~= 735ms` was mostly upstream brain/token
  wait. Optimize the provider/aggregator path against the new first-text metric.
- `correction_001` is flaky (4/5): one repeat the STT interim classified as
  `normal_interrupt` instead of `correction` — a semantic-robustness issue, not
  latency.

These thresholds are encoded as tiered SLO gates (`target` + `acceptable`) in
`eidolon/livekit/benchmarks/slo.py`. Provider micro-gates stay advisory until a
run has `>= 20` repeats (`--repeat 20`), so a smoke run cannot claim or fail
top-tier on a single sample.

Milestone: `commit -> first_audio` 1635 ms -> p95 <= 1100 ms. The biggest lever
is the ~1s endpoint wait now isolated as `stt_final_after_commit_ms`; closing it
(interim-preemptive brain request, faster STT endpointing) plus driving
`tts_first_text_sent_to_provider_first_audio_ms` under 250 ms is **Phase 2**
(runtime), measured by the new marks above.

Every real-provider run must also pass real-call verification
(`eidolon/livekit/benchmarks/realcall.py`): provider != mock, minimum audio
bytes, and per-turn brain-gRPC / STT-stream evidence. A run that cannot prove
real calls fails instead of masquerading as a pass.

## Phase 2 Results: Preemptive (Speculative) Generation

Real-room A/B (`--repeat 5`, eidolon_agent + Bailian), 2026-05-30:

| variant | commit→first_audio p50/p95 | preemptive trig/reuse/disc | notes |
| --- | ---: | --- | --- |
| OFF (baseline) | 1718 / 1755 | 0/0/0 | brain after commit |
| native preemptive only | 1631 / 1758 | 0/0/0 | fires on the late FINAL → no gain |
| **+ STT PREFLIGHT (shipped)** | **1574 / 1617** | 5/4/1 | brain fully hidden; 5/5, clean discard |
| + preemptive_tts | 1721 / 1875 | 5/3/2 | no gain (see below); reverted |
| **+ STT eos=400ms (shipped)** | **820 / 960** | 1/5 | STT FINAL 961→96ms; **top-tier band** |

Findings:

- Native `preemptive_generation` alone is a no-op for Bailian: its trigger is
  gated on a stable transcript, and Bailian emits no preflight and its FINAL
  lands ~1s after speech stop. Emitting `PREFLIGHT_TRANSCRIPT` on a stable
  interim (`speech_stream.py`) is what makes preemption fire early.
- With PREFLIGHT, the brain is **fully hidden** (~150ms saved → 1718→1574),
  safely (5/5 pass, reuse 4/5, the 1 discard clean). Shipped: `turn_policy.
  preemptive.enabled=true, preemptive_tts=false`.
- `preemptive_tts=true` gives **no further gain**: playback (`tts_first_audio`)
  is scheduled only at the framework's `on_end_of_turn`, which still waits for
  the STT FINAL (~961ms). Preemptive_tts pre-*synthesizes* but plays only on
  confirm, so it cannot beat the final-wait; it only added discards. Reverted.

Bottom line: preemption pre-computes the response but **playback is gated on the
STT FINAL**, so `commit→first_audio (1574) ≈ stt_final_after_commit (961) +
TTS_TTFB (649)`. The win therefore came from cutting the gating cost, not more
preemption: lowering FunASR `max_sentence_silence` 800→400ms collapsed the FINAL
wait **961→96ms** and **commit→first_audio 1574→820 p50 / 960 p95** — into the
top-tier band (p95 meets the ≤1100 target tier), still 5/5 / 0 flaky.

Then the TTS first-sentence flush (`aggregator_first_sentence_flush_any_punct`):
sending the first TTS segment on an early prefix+punctuation instead of waiting
for the 12-char soft-min cut it further to **759 p50 / 804 p95**.

Net Phase 2: **commit→first_audio 1718 → 664ms p50 / 777 p95 (~61%)** over a
hardened 20-repeat run (100 sessions, 5/5 / 0 flaky) — both inside the industry
top-tier *target* tier (p50 ≤800, p95 ≤1100), via STT PREFLIGHT (hide brain) +
faster STT endpointing (cut FINAL wait) + TTS first-sentence flush. The
`commit_to_first_audio` SLO gate is now `required` (acceptable tier) and passes
under `--enforce-slo`.

eos safety — validated: a paused-speech probe (mid-clause pause 300/500/700ms)
splits into two finals **identically at eos=400/600/800**, so `max_sentence_silence`
does NOT control mid-utterance splitting (FunASR's internal ~300ms segmentation
does) — it only governs trailing-silence finalization (the measured win).
Lowering eos to 400 therefore does **not** introduce or worsen premature splits;
it is safe vs the default 800. (Probe uses synthesized speech + digital silence;
real-audio sign-off for filled-pause hesitation is still worthwhile.) The
first-flush makes the first spoken segment short — a prosody/UX trade, raise the
first-sentence dials if needed. Residual floor: TTS provider synthesis
(~500–600ms TTFB), which needs a faster provider path, not channel tuning.

### Preemptive on/off A/B — default OFF (2026-05-31)

Same session, back-to-back, eos=400 fixed, only `turn_policy.preemptive.enabled`
toggled (`--repeat 12` each):

| arm | commit→first_audio p50/p95 | preemptive trig/reuse/discard |
| --- | ---: | --- |
| preemptive OFF | 725 / 880 | — |
| preemptive ON | 736 / 847 | 15 / 11 / 4 (waste 27%) |

The two are identical within noise (n≈12) — **preemptive adds no measurable
first-audio benefit once eos is low**, because the fast FINAL (~96ms after
commit) leaves nothing to overlap. Meanwhile it wastes 27–58% of speculative
brain turns (clean cancels, but real upstream load). eos=400 and preemption are
overlapping levers; lowering eos already captured the win, so preemption is
**now default OFF** (`PreemptivePolicyConfig.enabled=False`). The machinery
(STT PREFLIGHT, config gate, reuse/waste instrumentation) stays available —
enable it only for deployments forced to keep a high endpointing silence (slow
FINAL), where the overlap pays off. Earlier attempt to cut the waste by raising
the stability window (320→550ms) backfired (waste 58→69%): a window longer than
the eos silence overshoots the FINAL.

## Short-Term Targets

```text
interrupt_resolution P95 <= 900 ms
stt_first_interim P95 <= 1000 ms
commit_to_brain_rpc_started P95 <= 500 ms
brain_rpc_first_delta P95 <= 500 ms
tts_first_audio P95 <= 500 ms
commit_to_first_audio P95 <= 1600 ms
```

## Mid-Term Targets

```text
hard_interrupt P95 <= 500-650 ms
normal user stop -> first audio <= 1200 ms
brain first delta <= 300-500 ms
TTS first audio <= 300-500 ms
backchannel false cancel <= 2%
```
