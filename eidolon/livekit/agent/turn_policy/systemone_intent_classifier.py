"""SystemOne choice API adapter; no dialogue policy or generation inside."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from .intent_classifier import (
    INTENT_INSTRUCTIONS,
    INTENT_LABELS,
    IntentClassification,
    InterruptIntentClassifier,
    intent_state,
)


class SystemOneInterruptClassifier(InterruptIntentClassifier):
    """Reuse a bounded HTTP pool for structured, stateless intent evidence."""

    _CRITERIA = {
        "keep": "保持当前播报：附和、确认听到、希望继续当前内容；撤回其他要求后希望继续也属于此类。",
        "stop": "只停止当前播报，不生成新的回答。",
        "reply": "停止当前播报并回应新的信息、问题、纠正或讲解方向。语法不完整不等于不能打断。",
        "wait": "话意尚未形成、指向不明或缺少必要上下文。",
    }

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_sec: float = 1.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(timeout_sec=timeout_sec)
        self._endpoint = endpoint
        self._model = model
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            timeout=timeout_sec, trust_env=False, follow_redirects=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )
        self._closed = False

    async def classify(self, text: str, *, assistant_text: str) -> IntentClassification:
        if self._closed:
            raise RuntimeError("intent classifier is closed")
        body = {
            "model": self._model,
            "state": intent_state(text, assistant_text),
            "questions": {"intent": {
                "type": "choice", "instructions": INTENT_INSTRUCTIONS,
                "criteria": self._CRITERIA,
            }},
        }
        # Total budget includes pool admission as well as HTTP read/write time.
        async with asyncio.timeout(self._timeout_sec):
            response = await self._client.post(self._endpoint, json=body, follow_redirects=False)
            response.raise_for_status()
            return self._parse(response.json())

    @staticmethod
    def _parse(data: Any) -> IntentClassification:
        try:
            answer = data["answers"]["intent"]
            label = answer["choice"]
            probabilities = answer["probabilities"]
            if (
                answer["type"] != "choice" or label not in INTENT_LABELS
                or not isinstance(probabilities, dict) or set(probabilities) != set(INTENT_LABELS)
            ):
                raise ValueError("invalid intent choice schema")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1
                for value in probabilities.values()
            ):
                raise ValueError("invalid intent probabilities")
            if (
                abs(sum(probabilities.values()) - 1) > .001
                or probabilities[label] + 1e-6 < max(probabilities.values())
            ):
                raise ValueError("inconsistent intent probabilities")
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid SystemOne intent response") from exc
        # Raw scores are observable, but never impose an uncalibrated 0.8 gate.
        return IntentClassification(
            INTENT_LABELS[label], probabilities[label], "systemone",
            "discrete_choice_raw_probability", dict(probabilities),
        )

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client:
                await self._client.aclose()
