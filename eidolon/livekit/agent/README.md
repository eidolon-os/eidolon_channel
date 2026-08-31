# Eidolon Channel Agent Code Map

This package is the LiveKit-backed channel worker. The root keeps only public
entrypoints; implementation code lives under named boundary packages.

## Entry Points

- `server.py` starts the worker, resolves room/session metadata, and constructs
  the mode-specific pipeline.
- `full_duplex/` is the full-duplex realtime path. New code imports
  `StreamingPipeline` from `eidolon.livekit.agent.full_duplex`.
- `half_duplex/` is the push-to-talk path. New code imports
  `HalfDuplexPttPipeline` from `eidolon.livekit.agent.half_duplex`.
- `factory.py` builds shared STT, TTS, VAD, LLM, voiceprint, and runtime services.

## Boundary Packages

- `integration/` contains direct external-contract adapters built on public
  LiveKit events/options and LiveKit data-channel payload parsing.
- `session/` contains per-session handlers and effect appliers: room data,
  provider events, idle, semantic interruption effects, commit guards, shared
  EOT helpers, session-local `eidolon.control` envelope helpers, full-duplex
  transcript echo gating, message helpers, and user-turn coordination.
  `ProviderEventObserver` owns provider event observer installation, pending
  STT replay, STT turn-audio observation, and timeline recording; full-duplex
  pipeline code should use that owner instead of reintroducing pipeline proxy
  methods. Import helpers from their concrete modules; `session/__init__.py`
  is intentionally not a broad facade.
- `full_duplex/` contains the full-duplex realtime AgentSession pipeline:
  open-mic VAD/STT/EOT, natural interruption ownership, backchannel/false
  interruption handling, user-state/transcript event normalization,
  user-state/transcript entry routing, VAD speech segment lifecycle, transcript
  admission/echo gating before turn evidence, accepted transcript recording,
  semantic interrupt trigger gating, the observable full-duplex turn contract
  state machine, interruption output side effects, voiceprint-gated user-turn
  completion, framework completed-turn gating,
  interrupted context ledger wiring, AgentSession run/start/shutdown/close
  lifecycle, output ducking install/arming, `client.audio_state`
  freshness/playback views, and explicit client preempt handling for deliberate
  full-duplex controls.
  `StreamingPipeline.run()` and `StreamingPipeline.shutdown()` are the public
  entry points; lifecycle steps such as room teardown and proactive background
  consumer management live in `full_duplex/lifecycle.py`, accepted transcript
  side effects live in `full_duplex/transcript_recorder.py`, user-turn
  completion and voiceprint commit gates live in `full_duplex/turn_completion.py`,
  full-duplex contract transitions live in `full_duplex/state_machine.py`,
  owner ledger decisions live in `session/user_turn_coordinator.py`, and
  LiveKit completed-turn hook gating lives in
  `full_duplex/framework_completed_turn.py`.
- `half_duplex/` contains the half-duplex PTT pipeline: hold-scoped audio
  recording, one-shot STT, PTT control status helpers, and the PTT controller
  that does not consume streaming transcript events or EOT.
- `turn_policy/` contains owner logic and pure turn decisions: intent/evidence,
  attention admission, tier policy chain, and runtime state.
- `output/` contains output-side components: playback controller, ducking state,
  and filler playback.
- `providers/` contains provider-neutral stage wrappers for STT, TTS, VAD, and
  LLM. Concrete model integrations and model resources still live under
  `eidolon.livekit.plugins`.
- `shared/` contains runtime primitives used by both mode pipelines, such as
  `BasePipeline`, `PipelineState`, callbacks, and turn-id generation.
- `runtime/` contains worker runtime resolution: participant metadata parsing,
  Kernel Device Mount consumption, System Data Runtime Authority consumption,
  and the session-scoped Agent token resolver.
- `context/` contains conversation-context ledger helpers.
- `observability/` contains timeline/metrics helpers and `ChannelTurnEventSink`.
  The sink is a best-effort in-process observer for bounded session/turn phase,
  milestone, and terminal facts. It never opens System Data, publishes global
  audit, or exposes transcript/audio; a future metrics/tracing adapter may
  consume its vocabulary asynchronously.
- `session/agent_output_coordinator.py` owns the one committed response timeline.
  A new STT/VAD candidate may coexist with it during barge-in, while Brain/TTS/
  playback events and non-recoverable output errors remain bound to the response
  until it completes, fails, or is interrupted.
- `eidolon_agent_rpc/` contains the remote Eidolon Agent LLM/proactive bridge.
- `speaker_verification/` contains voiceprint service/store orchestration.
  Model providers and model resources live under `eidolon.livekit.plugins`.

New internal code should import concrete modules directly. Use
`eidolon.livekit.agent.providers` for STT/TTS/VAD/LLM stage wrappers and
`eidolon.livekit.agent.shared` for pipeline primitives. The root
`eidolon.livekit.agent` package no longer re-exports mode pipelines and should
not be used as a compatibility facade.
