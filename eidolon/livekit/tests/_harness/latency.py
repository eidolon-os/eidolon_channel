"""Read-only timing probe over the existing production/SDK replay harness.

Input reference: last 20 ms frame with RMS >= 120/32768, measured when delivered
at real-time pace. It is a reproducible acoustic estimate, not human annotation.
Output reference: first audible frame of the reply segment after user commit.
Device/RTC playback is outside this in-memory measurement.
"""
import time

from .audio import pcm_rms


class ReplyLatencyProbe:
    def __init__(self, pipeline, handle):
        self.pipeline = pipeline
        self.handle = handle
        self.timelines = []
        self.predictions = []
        self.final_predictions = []
        self.frame_offset = len(handle.audio_in.frame_delivered_at)
        self.detector = handle.agent.turn_detection
        # Streaming detectors emit predictions from their audio stream instead
        # of exposing a text prediction method. The acoustic/sink clock is the
        # same; their observed prediction events can populate `predictions`.
        self.predict = getattr(self.detector, 'predict_end_of_turn', None)
        self.eot_model = pipeline._get_eot_model()
        self.update_asr = self.eot_model.update_asr

        def observed_asr(text, is_final):
            score = self.update_asr(text, is_final=is_final)
            if is_final:
                self.final_predictions.append((time.monotonic(), score, text))
            return score

        self.eot_model.update_asr = observed_asr

        async def observed_predict(ctx, **kwargs):
            score = await self.predict(ctx, **kwargs)
            threshold = await self.detector.unlikely_threshold('zh')
            self.predictions.append((time.monotonic(), score, threshold))
            self._remember_timeline()
            return score

        if self.predict is not None:
            self.detector.predict_end_of_turn = observed_predict
        handle.session.on('user_state_changed', self._remember_timeline)

    def _remember_timeline(self, *args):
        timeline = self.pipeline._timeline
        if timeline is not None and all(timeline is not t for t in self.timelines):
            self.timelines.append(timeline)

    def close(self):
        if self.predict is not None:
            self.detector.predict_end_of_turn = self.predict
        self.eot_model.update_asr = self.update_asr
        self.handle.session.off('user_state_changed', self._remember_timeline)

    def report(self, pcm, *, sample_rate=16000):
        frame_bytes = sample_rate // 50 * 2
        voiced = [i for i, pos in enumerate(range(0, len(pcm), frame_bytes))
                  if pcm_rms(pcm[pos:pos + frame_bytes]) >= 120 / 32768]
        assert voiced, 'no voiced reference frames'
        # Each delivered PCM packet contains an already captured 20 ms frame.
        end = self.handle.audio_in.frame_delivered_at[self.frame_offset + voiced[-1]]
        users = [e for e in self.handle.events.of_type('conversation_item_added')
                 if getattr(e.payload.item, 'role', None) == 'user']
        assert len(users) == 1, 'latency sample requires exactly one committed user turn'
        committed = users[0].timestamp
        audible = [s.first_audible_at for s in self.handle.audio_out.segments
                   if s.started_at >= committed and s.first_audible_at is not None and not s.cleared]
        assert audible, 'no audible reply after user commit'
        first = min(audible)
        self._remember_timeline()
        marks = self.timelines[-1].timestamps if self.timelines else {}
        complete = [at for at, score, threshold in self.predictions
                    if threshold is not None and score >= threshold and at <= committed]
        threshold = self.pipeline._turn_policy.eot.eot_unlikely_threshold
        final_complete = [at for at, score, text in self.final_predictions
                          if score >= threshold and text == users[0].payload.item.text_content]
        def ms(at):
            return None if at is None else round((at - end) * 1000, 1)
        return {
            'reference': 'last voiced 20ms input frame to first audible reply frame at memory sink',
            'stop_to_reply_audio_ms': ms(first),
            'complete_to_reply_audio_ms': round((first - final_complete[-1]) * 1000, 1) if final_complete else None,
            'sdk_complete_to_reply_audio_ms': round((first - complete[-1]) * 1000, 1) if complete else None,
            'commit_to_reply_audio_ms': round((first - committed) * 1000, 1),
            'from_acoustic_stop_ms': {
                **{name: ms(at) for name, at in marks.items()},
                'sdk_user_committed': ms(committed), 'reply_audible': ms(first),
            },
            'eot_predictions': [{'after_stop_ms': ms(at), 'score': score, 'threshold': threshold}
                                for at, score, threshold in self.predictions],
            'final_eot_predictions': [{'after_stop_ms': ms(at), 'score': score, 'text': text}
                                      for at, score, text in self.final_predictions],
        }
