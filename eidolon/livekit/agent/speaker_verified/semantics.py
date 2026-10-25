"""Use existing numerical EOT and intent services, without ordinary turn effects."""

import math

from ..turn_policy.intent_classifier import InterruptIntent, intent_requires_reply


class CandidateSemantics:
    def __init__(self, eot, *, classifier, assistant_text):
        self._eot, self._classifier, self._assistant_text = eot, classifier, assistant_text

    async def is_complete(self, text: str) -> bool:
        # This public score-only path never records rejected text in EOT
        # history/debug files. The ordinary endpointing adapter uses it too.
        score = self._eot.score_candidate(text)
        threshold = await self._eot.unlikely_threshold("zh")
        return (
            math.isfinite(score)
            and threshold is not None
            and math.isfinite(threshold)
            and 0 <= score <= 1
            and 0 <= threshold <= 1
            and score >= threshold
        )

    async def classify(self, text: str, output) -> str:
        context = self._assistant_text(output) if output is not None else ""
        if self._classifier is None or (output is not None and not output.done() and not context):
            return "no_action"
        # A stop-only request stays stop-only if playback has just completed.
        # Absence of output is not evidence that a new reply was requested.
        result = await self._classifier.classify(text, assistant_text=context)
        if result.intent == InterruptIntent.HARD_STOP:
            return "stop_output"
        if intent_requires_reply(result.intent):
            return "reply"
        return "no_action"
