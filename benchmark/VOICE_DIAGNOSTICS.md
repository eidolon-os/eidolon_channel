# Limited voice diagnostics

`voice_diagnostic_capture` is an explicitly invoked diagnostic utility, not an
enrollment endpoint or a conversation input/gate. It records only the specified
participant's first microphone publication. It cannot publish audio/data or
explicitly dispatch an agent. Automatic subscription is disabled. Replacement
tracks are not concatenated into the original capture.

## Operator procedure

1. Obtain consent for this specific test window. Confirm the existing device room
   and participant through LiveKit's participant inventory; do not guess or create
   a test room. Confirm ordinary conversation remains on the rollback baseline.
   A connected participant and microphone publication do **not** prove an active
   conversation: BOX-3 emits zero frames while its microphone input gate is closed.
   Have the user activate the normal conversation immediately before the window,
   verify the current session is active, and explicitly signal when to speak.
   Earlier successful conversation is not evidence that the session is still active.
2. Run on the Pi5 with its installed Channel environment and existing LiveKit
   credentials. Do not print credentials, tokens, audio, transcripts or embeddings.
   Create a dedicated private parent directory (mode `0700`) outside the release
   and enrolled-recording store. Do not copy diagnostic audio to the development
   machine or repository.
3. Tell the user when to speak; start only once they are ready. For example:

   ```sh
   python -m benchmark.voice_diagnostic_capture \
     --start-authorized-window \
     --room "$VERIFIED_ROOM" --participant "$VERIFIED_DEVICE" \
     --private-parent /var/lib/eidolon/voice-diagnostics \
     --wall-seconds 45 --audio-seconds 30 --retain-seconds 600
   ```

   The capture deadline includes connection time. Hard limits are 60 wall seconds,
   30 seconds of decoded PCM, and 900 seconds of temporary file retention. The
   default retention is 600 seconds after capture. `CAPTURE_WINDOW_OPEN` marks the
   connected window; `CAPTURE_SAVED` marks that subscription has ended and includes
   the exact private directory and deletion deadline. Do not start a new test
   automatically when no microphone/audio was received.
4. Analyze on the Pi5 during that retention period. Files are `audio.wav` and
   `capture.json`, mode `0600`, inside a fresh `0700` directory. PCM is mono 16 kHz
   decoded by LiveKit; receipt timestamps are included to expose arrival gaps.
   Do not assume this subscriber sees the same sample/turn boundaries as the agent.
   The capture is a bounded window, not an automatically detected complete utterance.
   Reject all-zero PCM before model inference; duration alone is not usable audio.
   Establish effective speech before interpreting any embedding score. If capture
   prerequisites failed, report an invalid diagnostic attempt, delete its files and
   resolve the cause before requesting another test window.
5. Once analysis finishes, create `analysis.done` inside that exact directory.
   The capture process removes its temporary directory and emits `CAPTURE_DELETED`.
   If analysis never signals completion, the retention deadline removes it.
   SIGINT/SIGTERM also unwind cleanup. Confirm deletion before reporting completion;
   after an uncatchable kill or host power loss, inspect and remove this run's exact
   directory manually. Never delete enrolled audio or glob unrelated diagnostics.

Only aggregate quality, scores, model/preprocessing identity and timing belong in
the acceptance report. Existing phone trial audio can serve as a diagnostic
reference with identical preprocessing; it is not a replacement for independent
enrollment/calibration/test sessions. Never tune a production threshold using this
single positive example. The tool does not enable voice protection or change the
working welcome cue, input, ASR, interruption or response pipeline.

## Verification

```sh
.venv/bin/python -m pytest eidolon/livekit/tests/benchmark/test_voice_diagnostic_capture.py -q
```

Fake RTC tests check explicit subscription, no publication/dispatch grants,
audio/wall limits, missing microphone, cancellation, restrictive files and cleanup
both after errors and at retention expiry. These tests are not a real-device
capture or cross-device speaker-verification acceptance result.
