# Interrupted suspended speech must finish before the next reply

## Evidence and root cause

The Mobile + Pi5 session `mobile-4345d0ef-aa27dedb-00000001` on
2026-10-16 (Asia/Shanghai) produced final ASR at 18:18:13.673, cancellation
at 14.510, and `playback.stop` at 14.515. At 19.514, LiveKit reported
`speech not done in time after interruption, cancelling the speech arbitrarily.`
The replacement reply entered thinking at 19.550 and produced TTS through
31.518, but never entered speaking. The following request did speak normally.
Source evidence: workspace `diagnostics/2026-10-16-box3-compare/`.

Installed `livekit-agents` is 1.7.1 on both Mac and Pi5. Their RoomIO `_output.py`
has SHA-256 `44844576aa51de0056734a43f7d05f810fb4f90ffa041303785d06870c4317ad`.
The Channel lock uses the registry package, not an editable LiveKit fork;
`eidolon-sdk` is the local editable dependency. No dependency change is needed.

`OutputController.flush()` defers downstream flush while a suspended local tail
exists. If synthesis has already ended, its producer will never call flush
again. Previously cancellation invalidated `_flush_pending` and discarded that
tail, then called only downstream `clear_buffer()`. In installed RoomIO, clear
sets an interruption event; **flush starts the worker that consumes it, clears
the native queue, and emits playback_finished**. With no flush worker, the SDK
waits until its five-second SpeechHandle cancellation timeout. The unconsumed
interruption event then discards the next reply until that reply's flush.

The previous native-queue tests flushed *before* suspension and therefore
already had a flush worker. They could not catch this path.

## Change

Track the admitted output segment separately from pending local writes. Normal
flush, cancel, and clear share one segment finalizer. Discarding pending writes
cannot discard the obligation to finish the segment. Forwarded segments retain
the SDK's own playback position and interruption completion; wholly local
segments finish with position zero because there is no downstream event to
forward. Repeated SDK cleanup does not finish the same segment twice. The
native queue stays paused until normal reset/resume, so cancellation does not
release old speech.

No private SDK mutation, new timer, phrase list, threshold adjustment, or SDK
upgrade is involved.

## Automated validation

`test_output_native_pause.py` adds deferred-flush completion cases and real
AgentSession + TranscriptSynchronizer + RoomIO integration. Only RTC AudioSource
is replaced with an in-memory recorder; generated PCM never reaches a speaker.
Cases cover synthesis already ended / blocked by capacity, with / without a
forwarded prefix, exactly one completion, and complete next-reply PCM with no
old samples. Existing rollback tests continue to preserve unheard PCM.

All 12 native-output cases and 139 related output, intent, commit-guard and
endpointing regressions pass. Loading the pre-fix controller from Git in a
separate test process makes 5 of the 6 new cases fail; the remaining case has
an active producer whose cancellation finally supplies the missing flush.
This negative control does not modify the working source or installed SDK.

## Remaining endpointing latency

The second request's final at 18:18:35.757 received EOT probability 0.104 at
35.777 and entered thinking at 38.115. The installed SDK selects its maximum
endpointing delay for scores below the configured threshold. Its remaining
sleep is computed from the last-speaking timestamp, **not restarted after
Channel settlement**. Thus this is a separate learned-completeness limitation,
not evidence that the five-second cancellation timeout is an EOT delay.
The repository already has explicit expected-failure contracts for complete
short Chinese turns scoring below threshold. Changing that threshold without
an accuracy corpus would also affect unfinished utterances and is outside this
root-cause fix. Real-device latency after deployment remains to be measured;
automated completion bounds are not acoustic latency measurements.
