# Eidolon Channel Agent Code Map

This package is the LiveKit-backed channel worker. The root keeps only public
entrypoints; implementation code lives under named boundary packages.

## Entry Points

- `server.py` starts the worker, resolves room/session metadata, and constructs
  the pipeline.
- `streaming.py` is the realtime AgentSession wrapper. It wires LiveKit events
  to small session helpers and should stay mostly orchestration.
- `batch.py` is the non-realtime/manual audio path.
- `factory.py` builds shared STT, TTS, VAD, LLM, voiceprint, and runtime services.

## Boundary Packages

- `integration/` contains direct external-contract adapters: LiveKit framework
  patches and LiveKit data-channel payload parsing.
- `session/` contains per-session handlers and effect appliers: room data,
  client controls, half-duplex PTT turn ownership, interaction modes, provider
  events, idle, semantic interruption effects, commit guards, shared EOT helpers,
  session-local `eidolon.control` helpers, the streaming/manual PTT adapter,
  full-duplex transcript echo gating, message helpers, and user-turn
  coordination. Import helpers from their concrete modules; `session/__init__.py`
  is intentionally not a broad facade.
- `half_duplex/` contains the optional segment-based PTT pipeline: hold-scoped
  audio recording, one-shot STT, and the PTT controller that does not consume
  streaming transcript events or EOT.
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
`eidolon.livekit.agent` package remains only a small public compatibility
surface for pipeline entrypoints.
