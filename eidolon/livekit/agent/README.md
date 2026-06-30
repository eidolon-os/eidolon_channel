# Eidolon Channel Agent Code Map

This package is the LiveKit-backed channel worker. The root keeps only the
public entrypoints and legacy import shims; real implementation code should live
under a named boundary package.

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
  client controls, interaction modes, provider events, idle, semantic
  interruption effects, commit guards, shared EOT helpers, message helpers, and
  user-turn coordination.
- `turn_policy/` contains owner logic and pure turn decisions: intent/evidence,
  attention admission, tier policy chain, and runtime state.
- `output/` contains output-side components: playback controller, ducking state,
  and filler playback.
- `pipeline/` contains provider-neutral stage wrappers for STT, TTS, VAD, and
  LLM.
- `runtime/` contains worker runtime resolution: metadata parsing, admin
  resolve, token/runtime helpers.
- `context/` contains conversation-context ledger helpers.
- `observability/` contains timeline and metrics helpers.
- `eidolon_agent_rpc/` contains the remote Eidolon Agent LLM/proactive bridge.
- `speaker_verification/` contains voiceprint service/provider code.

## Compatibility Shims

The root files `_framework_patches.py`, `client_audio_state.py`,
`output_controller.py`, `filler.py`, and `interrupt_decider.py` are legacy
import shims. New code should import from the boundary packages above.
