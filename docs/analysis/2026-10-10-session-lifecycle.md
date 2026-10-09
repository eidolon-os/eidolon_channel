# Session shutdown and audio-only welcome

The BOX-3 incident involved a Provider dispatch withdrawal followed by LiveKit
removing the worker after its three-second grace period. The application
entrypoint waited for the pipeline, which waited for AgentSession closure;
LiveKit waits for the entrypoint before closing its primary AgentSession.

The job entrypoint now waits only for successful session startup. A job-owned
task runs the pipeline; its completion requests job shutdown, and a shutdown
callback joins its resource cleanup. Startup failures still propagate through
the entrypoint. No LiveKit package files or private SDK state are patched.
Closing sessions use the public zero final-transcript wait option: withdrawing
a conversation is not an invitation to finish a new input turn.

A pure audio welcome has an empty text segment. LiveKit skips empty deltas, so
its transcription sink otherwise never marks the segment complete. The queued
welcome audio iterator explicitly captures and flushes an empty text segment
through the public sink before yielding its first audio frame. It does not
insert whitespace or fabricated text into subtitles/history. Synchronization
remains enabled for subsequent spoken replies.

Regression coverage includes entrypoint/close ordering, startup failure and
cancellation, and actual AgentSession + TranscriptSynchronizer playback of a
cue followed by a normal subtitle-bearing segment. Device close provenance
cannot be reconstructed from USER_INITIATED alone; that remains an SDK reason,
not proof of a physical button action.
