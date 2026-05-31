"""Service-level objective gates for benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MetricSource = Literal["summary", "timeline"]
Tier = Literal["target", "acceptable"]


@dataclass(frozen=True)
class SloGate:
    name: str
    runner: str
    source: MetricSource
    metric: str
    statistic: str
    max_value: float
    tier: Tier = "acceptable"
    required: bool = False
    # A percentile is meaningless on a handful of samples. When min_samples > 0
    # and the metric has fewer samples, the gate reports `advisory` instead of
    # hard pass/fail, so a low-repeat smoke run cannot claim (or fail) top-tier.
    min_samples: int = 0
    description: str = ""


# Industry top-tier reference, May 2026. Sources: Twilio Core-Latency 2025.11
# (STT 350/500, LLM TTFT 375/750, TTS TTFB 100/250, platform turn gap
# 885/1100), OpenAI gpt-realtime (p50<800 / p95<1.4s), LiveKit semantic turn
# detector (<75ms P99), Cartesia Sonic / ElevenLabs Flash TTS TTFB ~75-90ms.
# Each segment gets a "target" (top-tier) and "acceptable" (upper-limit) gate.
#
# Provider micro-gates (timeline source) require `min_samples` real repeats
# before they hard-assert. The two experience gates (summary source) always
# evaluate so even a single real-room run flags a gross regression.
_PERCENTILE_MIN_SAMPLES = 20


def _tier_pair(
    *,
    base_name: str,
    runner: str,
    source: MetricSource,
    metric: str,
    statistic: str,
    target: float,
    acceptable: float,
    description: str,
    min_samples: int = 0,
    required_target: bool = False,
    required_acceptable: bool = False,
) -> tuple[SloGate, ...]:
    return (
        SloGate(
            name=f"{base_name}_target",
            runner=runner,
            source=source,
            metric=metric,
            statistic=statistic,
            max_value=target,
            tier="target",
            min_samples=min_samples,
            required=required_target,
            description=f"{description} (top-tier target)",
        ),
        SloGate(
            name=base_name,
            runner=runner,
            source=source,
            metric=metric,
            statistic=statistic,
            max_value=acceptable,
            tier="acceptable",
            min_samples=min_samples,
            required=required_acceptable,
            description=f"{description} (acceptable limit)",
        ),
    )


# Enforcement ratchet (2026-05-30): only gates the system already meets with a
# real-room sample are `required=True` (a required failure exits the run under
# --enforce-slo). Today that is the two summary experience ceilings, which the
# system passes (~1300<=1400, ~2649<=2800) and which hard-judge on every run.
# The target-tier and the timeline provider gates (STT-final, TTS-TTFB, E2E)
# stay required=False — they are honest Phase-2 goals the preemptive-generation
# + TTS work will close, then they get promoted to required.
DEFAULT_SLO_GATES: tuple[SloGate, ...] = (
    # E2E experience (room participant口径, includes LiveKit transport).
    *_tier_pair(
        base_name="room_user_done_to_next_audio",
        runner="livekit_room",
        source="summary",
        metric="user_done_to_agent_audio_after_user_done_ms",
        statistic="p95",
        target=1100.0,
        acceptable=1400.0,
        description="Room user-audio-done to next agent audio P95",
        required_acceptable=True,
    ),
    SloGate(
        name="room_publish_to_first_audio",
        runner="livekit_room",
        source="summary",
        metric="publish_to_agent_audio_first_ms",
        statistic="p95",
        max_value=2800.0,
        tier="acceptable",
        required=True,
        description="Room publish-to-first-agent-audio P95",
    ),
    # E2E platform口径 (worker timeline, excludes participant network).
    *_tier_pair(
        base_name="commit_to_first_audio_p50",
        runner="livekit_room",
        source="timeline",
        metric="commit_to_tts_first_audio",
        statistic="p50",
        target=800.0,
        acceptable=1100.0,
        description="Commit -> first agent audio P50",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    *_tier_pair(
        base_name="commit_to_first_audio_p95",
        runner="livekit_room",
        source="timeline",
        metric="commit_to_tts_first_audio",
        statistic="p95",
        target=1100.0,
        acceptable=1400.0,
        description="Commit -> first agent audio P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    # Endpoint wait: commit -> STT provider final (the ~1s normal-turn lever).
    *_tier_pair(
        base_name="stt_final_after_commit",
        runner="livekit_room",
        source="timeline",
        metric="stt_final_after_commit_ms",
        statistic="p95",
        target=350.0,
        acceptable=500.0,
        description="Commit -> STT provider final P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    # Brain TTFT.
    *_tier_pair(
        base_name="brain_first_delta",
        runner="livekit_room",
        source="timeline",
        metric="brain_request_to_first_delta",
        statistic="p95",
        target=375.0,
        acceptable=750.0,
        description="Brain request sent -> first delta P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    # TTS TTFB and publish.
    *_tier_pair(
        base_name="tts_ttfb",
        runner="livekit_room",
        source="timeline",
        metric="tts_request_to_provider_first_audio_ms",
        statistic="p95",
        target=100.0,
        acceptable=250.0,
        description="TTS request -> provider first audio P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    *_tier_pair(
        base_name="tts_provider_to_agent_audio",
        runner="livekit_room",
        source="timeline",
        metric="tts_provider_first_audio_to_agent_audio_ms",
        statistic="p95",
        target=50.0,
        acceptable=120.0,
        description="TTS provider audio -> agent audio P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
    # Interrupt resolution.
    *_tier_pair(
        base_name="room_interrupt_resolved",
        runner="livekit_room",
        source="timeline",
        metric="vad_start_to_interrupt_resolved",
        statistic="p95",
        target=500.0,
        acceptable=650.0,
        description="VAD start -> interrupt resolution P95",
        min_samples=_PERCENTILE_MIN_SAMPLES,
    ),
)


def evaluate_slo_gates(
    runner_payload: dict[str, Any],
    gates: tuple[SloGate, ...] = DEFAULT_SLO_GATES,
) -> list[dict[str, Any]]:
    """Evaluate SLO gates that apply to a dashboard runner payload."""

    results: list[dict[str, Any]] = []
    runner_name = str(runner_payload.get("name") or "")
    for gate in gates:
        if gate.runner != runner_name:
            continue
        metric_values = _metric_values(runner_payload, gate)
        value = metric_values.get(gate.statistic) if metric_values else None
        count = metric_values.get("count") if metric_values else None
        missing = value is None
        sample_count = count if isinstance(count, (int, float)) else 0
        advisory = gate.min_samples > 0 and sample_count < gate.min_samples
        if missing:
            passed = not gate.required
        elif advisory:
            # Too few samples to assert a percentile: report, do not fail.
            passed = True
        else:
            passed = isinstance(value, (int, float)) and value <= gate.max_value
        results.append(
            {
                "name": gate.name,
                "runner": gate.runner,
                "source": gate.source,
                "metric": gate.metric,
                "statistic": gate.statistic,
                "tier": gate.tier,
                "value": value,
                "max_value": gate.max_value,
                "required": gate.required,
                "min_samples": gate.min_samples,
                "count": sample_count,
                "advisory": advisory,
                "passed": passed,
                "missing": missing,
                "description": gate.description,
            }
        )
    return results


def enforcement_failures(slo_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Required gates that hard-failed (the set that should fail a gated run).

    Advisory (too few samples) and missing-metric results never block, so a
    low-repeat or partial run cannot spuriously fail enforcement.
    """

    return [
        result
        for result in slo_results
        if result.get("required")
        and not result.get("passed")
        and not result.get("advisory")
        and not result.get("missing")
    ]


def _metric_values(
    runner_payload: dict[str, Any],
    gate: SloGate,
) -> dict[str, Any] | None:
    if gate.source == "summary":
        metrics = runner_payload.get("summary", {}).get("metrics", {})
    else:
        metrics = runner_payload.get("timeline", {}).get("latencies", {})
    value = metrics.get(gate.metric)
    return value if isinstance(value, dict) else None
