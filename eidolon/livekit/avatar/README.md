# Avatar — digital-human (talking-head) video for the LiveKit channel

Bridges an external audio-driven talking-head service into LiveKit's standard
**avatar-worker** mechanism, so a display-capable client (web/app) sees the
agent's face move while it speaks. Audio-only sessions are unaffected.

## Data flow

```
channel agent (eidolon)                         avatar worker (avatar-<room>)
  TTS PCM ──▶ DataStreamAudioOutput ─(DataStream)─▶ DataStreamAudioReceiver
                                                       └▶ EidolonDHVideoGenerator.push_audio
                                                            accumulate 1 segment → AudioSegmentEnd
  session.output.audio = DataStreamAudioOutput            └▶ POST /api/stream_video (PCM→WAV)
  (room audio output disabled)                             └▶ decode+retime (PyAV) → frames
                                                       AvatarRunner + AVSynchronizer
                                                       └▶ publishes synced audio+video track
web/app client ◀───────────── subscribes to avatar-<room> video+audio ──────────┘
```

## The service contract (measured — Phase 0, 2026-07-16)

`POST http://<host>/api/stream_video` (multipart: `audio` required WAV, optional
`image` JPEG, `device_info` JSON, `format` mp4|ts). Streams back fragmented
MP4 / MPEG-TS: **H.264 Constrained Baseline (yuv420p) + AAC-LC**.

- First byte ~130–180 ms; a 5 s clip fully generated in ~1.9 s (progressive).
- Audio returns at **24 kHz mono** regardless of input rate.
- `device_info` controls size/fps (requesting 512×512 → 448×448; `prefer_fps` honored
  in the container's declared rate).
- **The actual emitted frame count does NOT match the declared fps and varies
  with clip duration** (~15.5–18 fps for a requested 25). So the decoder
  **retimes** video to a constant `fps` against the authoritative audio duration
  and interleaves audio+video by time — lip-sync holds regardless of source rate.
  Do not "trust" the container fps.

## Modules

- `service_client.py` — async HTTP client (`DigitalHumanServiceClient`), `pcm16_to_wav`.
- `decoder.py` — `decode_stream` / `decode_container`: PyAV demux+decode, **retime to constant fps**, I420 + int16 PCM.
- `video_generator.py` — `EidolonDHVideoGenerator(VideoGenerator)`: segment accumulation, idle loop (keeps the face alive during generation latency), barge-in `clear_buffer` (aborts in-flight request + flushes).
- `worker.py` — `AvatarWorker`: connects the `avatar-<room>` participant, wires `AvatarRunner`. Channel spawns it in-process per avatar session (M1).

## Config

`config/settings.yaml` `avatar:` block (see `common/config/schema.py::AvatarConfig`).
`enabled` is a global kill-switch; per-session use is decided at runtime from the
joining participant's metadata (`resolve_avatar_requested`). `fps` is the constant
retime target (also requested from the service).

## Validation status

- **Validated against the real service** (standalone, no room): PCM→WAV→POST→decode→rtc
  frames; I420 byte size exact (`w*h*3//2`); audio 24 kHz; retiming exact
  (`frames == fps*duration`); clean audio/video interleave; idle-before-speech;
  `clear_buffer` abort. See `scratchpad/test_bridge.py`, `test_videogen.py`.
- **Deferred to the verify phase**: full room integration (AvatarRunner publish +
  DataStream + client subscribe) — run inside the actual channel agent + a browser
  client. A synthetic 3-participant test is blocked in the dev session by a WS
  signal-handshake quirk (server 1.11.0 vs client rtc 1.1.12 on the manual
  participant-connect path); it is not a code defect (token/secret verified valid).

## Enabling it + live end-to-end verify

The full loop is wired across three repos (all default-off / audio-only):

1. **channel** `config/settings.yaml` → add an `avatar:` block:
   ```yaml
   avatar:
     enabled: true
     service_url: http://111.2.199.31:61320
     width: 448
     height: 448
     fps: 25.0
   ```
   Restart the channel worker so it reloads config + this code.
2. **hub** already forwards `?avatar=1` → participant metadata `avatar: true` (see
   `hub/api/routers/system/config.py`). No config needed.
3. **web** (`eidolon_client_web`): set `NEXT_PUBLIC_AVATAR=1` in `.env.local`, `npm run dev`,
   open `/companion`, start a conversation.

Expected: the particle head shows first, then cross-fades to the photoreal
talking-head video (`avatar-<room>` track) once the agent speaks; barge-in stops
the face. With `avatar` off (default) the session is byte-for-byte the current
audio-only + particle-head behaviour.

Per-session/runtime: a client that does not send `avatar=1` stays audio-only in
the same channel fleet — the decision is made per connection from participant
metadata (`resolve_avatar_requested`), not a global switch.

## Fast-follow (not M1)

- Progressive decode (start decoding on the first ~150 ms chunk instead of
  buffering the whole segment) — internal to `decoder.py`, interface unchanged.
- Idle "breathing" loop instead of a static idle frame.
- Session reuse (`/api/start_stream` + `/video/{session_id}`) to avoid per-segment overhead.
- Move the worker to its own process/container for CPU isolation.
