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
- Online music uses the same authenticated Chat stream for tool calls, results,
  and cancellation. `runtime/music.py` composes the optional capability;
  `session/music.py` orchestrates commands; `output/music.py` owns playback and
  `output/music_cache.py` owns atomic downloads. The Lizheng adapter lives in
  `eidolon.livekit.plugins.music`, behind `eidolon.livekit.common.music` contracts.
  Playback uses the existing AgentSession audio output and interruption path.
  Only sessions with speech output and the remote Agent bridge advertise music.
  Shutdown drains tool requests and stops playback before closing the session.
  Credentials are optional; initialization failure disables music without
  disabling conversation. Cache paths derive from the Host's `EIDOLON_CACHE_ROOT`.
- `speaker_verification/` contains voiceprint service/store orchestration.
  Model providers and model resources live under `eidolon.livekit.plugins`.

New internal code should import concrete modules directly. Use
`eidolon.livekit.agent.providers` for STT/TTS/VAD/LLM stage wrappers and
`eidolon.livekit.agent.shared` for pipeline primitives. The root
`eidolon.livekit.agent` package no longer re-exports mode pipelines and should
not be used as a compatibility facade.

### Capability boundary: online music

Music is an optional tool capability hosted by Channel. Its current transport is
ChannelTool over the authenticated Chat stream, not a second MCP server. A future
MCP adapter must call the same capability implementation, not duplicate its player.

- Eidolon OS owns current-request interpretation, dialogue context, tool transport,
  cancellation routing, output arbitration and interruption classification. It
  does not search song catalogues or infer a music provider's resource state.
- The music capability owns bounded catalogue queries, candidate identities,
  provider authentication, media resolution, cache, preparation and playback.
  Search results are non-exhaustive candidates; the tool cannot silently substitute
  an alternative or drop an explicit version/language constraint.
- Agent resolves current references/consent against the preceding exchange. Prior
  dialogue alone never authorizes execution. No music-specific phrase matching or
  automatic execution on a transcript is allowed.
- `ChannelToolResult.outcome_state` describes the command (`accepted`, `completed`,
  `failed`); `content.status` describes the music resource (`preparing`, `accepted`,
  `playing`, `stopped`, `interrupted`, `completed`, `failed`, `not_found`). A successful
  status read is completed even when the resource is absent or failed. Legacy
  producers without an outcome are never presumed completed by the transport.
- `music.play` acceptance does not prove audible playback. `playing` means decoded
  frames have begun entering the Channel output, not endpoint acknowledgement.
  `music.status` supplies subsequent facts; no speculative success expression is
  permitted for an accepted command. This version does not push resource updates
  to Agent; callers must not assume such events exist.
- Preparation and playback share one task owner. Stop, replacement, timeout and
  shutdown cancel that owner, including an in-flight download. A repeated current
  playback reference cannot restart it or change its source. A stale stop reference
  cannot stop a newer task. Completed tool invocation and ongoing playback are
  separate lifetimes; normal turn completion does not terminate music.
- Direct voice interruption uses the existing OS audio path. The player observes
  that speech handle's interruption, so tool status reflects the same outcome.
  Neither the provider adapter nor the device classifies user dialogue.

Boundary tests cover preparation cancellation, stale references, retries, framework
interruption, exact versus unverified catalogue candidates, preserved constraints,
and command outcomes across protobuf. Model interpretation must additionally be
checked with varied conversations; deterministic tests do not prove semantic accuracy.

Recognized speech can arrive without a local VAD start. The existing transcript
boundary restores provider evidence before admission. A newer source span (or a
new revision with a higher sequence in the same stream) can open a new candidate
after the previous candidate is terminal; repeated wording does not make it a
duplicate. Late revisions and pending turns keep their existing owner. This uses
the same speech lifecycle and interruption orchestrator without fabricating VAD
or voiceprint audio. Providers without source evidence retain the VAD path.

Music remains a native AgentSession speech handle: OS forced interruption cancels
it and closes the decoder; no second music-specific interruption path is needed.
The real-session regression also verifies subsequent output remains usable.

`music.play` accepts a song ID, not a model-selected playback instance. The
capability derives the instance reference from the tool invocation's turn/call
identity. A new invocation starts a new instance even for the same song; retries
of the current invocation do not resurrect a terminal instance. The existing
Agent dispatcher deduplicates commands, including old requests after replacement.
Returned references target `music.stop`; cancellation compares the originating
invocation's reference so an old request cannot stop a newer playback.

When the intent provider fails or exceeds its deadline, an admitted nonempty
lexical final transcript degrades to ordinary interruption: cancel current output
and route the utterance through the existing user-turn/voiceprint/Agent path.
This is an availability policy, not a guessed stop intent. Successful semantic
uncertainty still waits; provisional/empty input and superseded requests do not
acquire cancellation authority. During an outage, even an acknowledgement may
interrupt output. No capability-specific keyword rules or second classifier are used.
