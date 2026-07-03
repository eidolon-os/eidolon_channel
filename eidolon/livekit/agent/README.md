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

- `integration/` contains direct external-contract adapters: LiveKit framework
  patches and LiveKit data-channel payload parsing.
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
  admission/echo gating before turn evidence, accepted transcript evidence
  recording, semantic interrupt trigger gating, interruption output side effects,
  voiceprint-gated user-turn completion, framework completed-turn alignment,
  interrupted context ledger wiring, AgentSession run/start/shutdown lifecycle,
  output ducking install/arming, `client.audio_state` freshness/playback views,
  and explicit client preempt handling for deliberate full-duplex controls.
  `StreamingPipeline.run()` and `StreamingPipeline.shutdown()` are the public
  entry points; lifecycle steps such as room teardown and proactive background
  consumer management live in `full_duplex/lifecycle.py`, while user-turn
  completion and voiceprint commit gates live in `full_duplex/turn_completion.py`.
- `half_duplex/` contains the half-duplex PTT pipeline: hold-scoped audio
  recording, one-shot STT, PTT control status helpers, and the PTT controller
  that does not consume streaming transcript events or EOT.
- `turn_policy/` contains owner logic and pure turn decisions: intent/evidence,
  attention admission, tier policy chain, and runtime state.
- `output/` contains output-side components: playback controller, ducking state,
  and filler playback.
- `pipeline/` contains provider-neutral stage wrappers for STT, TTS, VAD, and
  LLM.
- `runtime/` contains worker runtime resolution: metadata parsing and admin
  resolve helpers.
- `context/` contains conversation-context ledger helpers.
- `observability/` contains timeline and metrics helpers.
- `eidolon_agent_rpc/` contains the remote Eidolon Agent LLM/proactive bridge.
- `speaker_verification/` contains voiceprint service/store orchestration.
  Model providers and model resources live under `eidolon.livekit.plugins`.

New internal code should import concrete modules directly. The root
`eidolon.livekit.agent` package no longer re-exports mode pipelines and should
not be used as a compatibility facade.
