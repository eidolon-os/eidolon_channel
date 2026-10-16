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

## Regression provenance

Replaying historical controllers against the same installed SDK locates the
missing deferred-flush completion at `67cf3b0` (2026-09-06). Its parent starts
the native flush worker and completes the interrupted segment; `67cf3b0` does
neither. Both sides of the later native-pause change `14f101c` (2026-10-14)
also fail this check. `eb50405` restores completion. This comparison isolates
the controller change; it does not recreate every historical deployment's SDK.

## Pi5 deployment and BOX-3 follow-up

With explicit user approval, Ops activated
`pi5-playout-finalization-20261016-1`, carrying Channel
`eb5040569350f293909a38ff02a6cc0e47da99d4`. Only Channel advanced relative to
the prior release. Release doctor was healthy; the separate App-ready check
passed all 26 checks. An offline probe using that release's actual Python
runtime, AgentSession, TranscriptSynchronizer and RoomIO passed all four
prefix/no-prefix × producer-ended/blocked cases. Cancel-to-speech-done took
0.538–1.178 ms, and all 1600 next-reply PCM samples matched. These are process
completion measurements using an in-memory AudioSource, not acoustic latency.

The user's subsequent BOX-3 + Pi5 session
`esp32-a3c6ced5-687a95ab-00000002` started at 18:36:20 on 2026-10-16.
Host timestamps (Asia/Shanghai) show:

| Request | ASR final | Cancel | Old speech → listening | Thinking | Speaking |
| --- | --- | --- | --- | --- | --- |
| 给我讲一个笑话 | 18:36:24.925 | — | — | 18:36:25.001 | 18:36:27.388 |
| 换一个笑话 | 18:36:37.809 | 18:36:38.427 | 18:36:38.439 | 18:36:40.036 | 18:36:41.936 |
| 停，不要说了 | 18:36:55.138 | 18:36:55.564 | 18:36:55.573 | — | — |

For replacement, final-to-cancel was 618 ms, cancel-to-old-speech-finished
12 ms, cancel-to-thinking 1609 ms, and thinking-to-speaking 1900 ms.
Final-to-new-speaking was 4127 ms. For stop, final-to-cancel was 426 ms and
old speech finished 9 ms later. No SDK interruption timeout occurred, and the
replacement did enter speaking. Host speaking events do not acknowledge
physical speaker output. The stop transcript was correct in this run.

Both cancellations discarded zero *local* buffered frames; the first TTS
had completed before ducking. Therefore this run confirms ordinary native-tail
cancellation and next-turn scheduling after deployment, but does not itself
replay the original Mobile failure's seven locally held frames. The automated
regressions cover that deferred-flush path. This run also confirms the separate
remaining endpointing wait: the replacement scored 0.104, then spent roughly
1.56 seconds between EOT prediction and thinking.

Follow-up evidence is in workspace
`diagnostics/2026-10-16-box3-compare/box3-after-playout-fix-journal.log`,
`result-box3-after-playout-fix.json`, and `playout-probe-pi5-after.json`.
