"""Real-provider call verification for the voice benchmark.

A benchmark can only claim "top-tier" if it provably exercised real STT / TTS /
LLM(brain) calls — not mocks, silence, or a dead worker. This module turns that
into hard evidence: provider identity, minimum audio volume, and (for the real
room) brain-gRPC / STT-stream markers in the per-turn timeline. A run that
cannot prove real calls is marked failed rather than allowed to masquerade as a
pass.

It mirrors the provider/factory path used by ``scripts/smoke_realtime_stack.py``
(``SharedStageFactory.from_config``) for the pre-flight check, and reuses the
existing timeline JSONL attrs (``brain_rpc``, ``stt_stream``, audio-byte counts)
as evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from .provider_smoke import check_stt as provider_check_stt
from .provider_smoke import check_tts as provider_check_tts
from .schema import CaseResult, RunResult
from .timeline import load_timeline_records

# Real provider identifiers used by SharedStageFactory.from_config. Anything
# carrying a mock marker (or absent from this set) is treated as not-real.
REAL_PROVIDER_NAMES = frozenset(
    {
        "eidolon_agent",
        "eidolon_agent_rpc",
        "direct_llm",
        "bailian",
        "sensetime",
        "firered",
        "firered_pvad",
        "silero",
    }
)
MOCK_MARKERS = ("mock", "noop", "fake", "stub", "scripted", "dummy")

# A real Bailian TTS clip for a short sentence is well over this; a mock/empty
# synthesizer is not. 0.25s of 16kHz mono PCM16 is 8000 bytes.
MIN_COMPONENT_TTS_BYTES = 2000
MIN_ROOM_AGENT_AUDIO_BYTES = 8000

_ROOM_CASE_RE = re.compile(r"^voice-bench-(?P<case>.+)-[0-9a-f]{8}$")


def _is_mock(name: str) -> bool:
    lowered = (name or "").lower()
    if not lowered:
        return False
    if any(marker in lowered for marker in MOCK_MARKERS):
        return True
    return lowered not in REAL_PROVIDER_NAMES


def verify_provider_config(provider_config: dict[str, str]) -> list[str]:
    """Fail if any configured provider is a mock / unknown (non-real) provider."""

    failures: list[str] = []
    for role, name in sorted((provider_config or {}).items()):
        if not name:
            continue
        if _is_mock(name):
            failures.append(f"{role} provider {name!r} is not a real provider")
    return failures


def _base_case_id(case_id: str) -> str:
    # component results use "<case>:<component>"; rooms use the bare case id.
    return case_id.split(":", 1)[0]


def group_records_by_case(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        attrs = record.get("attrs") if isinstance(record.get("attrs"), dict) else {}
        room_name = attrs.get("room_name")
        if not isinstance(room_name, str):
            continue
        match = _ROOM_CASE_RE.match(room_name)
        if not match:
            continue
        grouped.setdefault(match.group("case"), []).append(record)
    return grouped


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_brain_rpc_evidence(record: dict[str, Any]) -> bool:
    """Whether a turn record proves a real eidolon_agent brain gRPC stream."""

    attrs = _mapping(record.get("attrs"))
    brain_rpc = _mapping(attrs.get("brain_rpc"))
    timestamps = _mapping(record.get("timestamps"))
    return (
        str(brain_rpc.get("provider") or "").startswith("eidolon_agent")
        and bool(brain_rpc.get("request_id"))
        and "brain_request_sent_at" in timestamps
        and "brain_first_delta_at" in timestamps
    )


def _verify_room_timeline(
    provider_config: dict[str, str],
    case_records: list[dict[str, Any]],
) -> list[str]:
    """Per-case room evidence: a worker turn record with the right STT stream.

    Brain-RPC evidence is intentionally NOT checked here. Interrupt / rollback
    cases capture the *decision* turn (action=cancel/rollback), which has no
    brain generation, so requiring brain marks per case is wrong. The brain is
    proven once per run instead (see ``apply_real_call_verification``); per case
    we rely on agent-audio bytes + the STT stream that transcribed the user.
    """

    failures: list[str] = []
    if not case_records:
        failures.append("no worker timeline records (cannot prove real call)")
        return failures

    if any(_is_control_only_explicit_client_preempt(record) for record in case_records):
        return failures
    if any(_has_half_duplex_ptt_segment_evidence(record) for record in case_records):
        return failures

    stt_provider = (provider_config or {}).get("stt", "")
    if stt_provider:
        stt_ok = any(
            str(_mapping(_mapping(r.get("attrs")).get("stt_stream")).get("provider") or "")
            == stt_provider
            for r in case_records
        )
        if not stt_ok:
            stt_ok = any(_has_transcript_timeline_evidence(r) for r in case_records)
        if not stt_ok:
            failures.append(f"no STT stream evidence for provider {stt_provider!r}")

    return failures


def _is_control_only_explicit_client_preempt(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    return bool(attrs.get("control_only")) and bool(
        _mapping(attrs.get("explicit_client_interrupt")).get("ptt")
    )


def _has_half_duplex_ptt_segment_evidence(record: dict[str, Any]) -> bool:
    attrs = _mapping(record.get("attrs"))
    if attrs.get("pipeline") != "half_duplex_ptt_segment":
        return False
    segment = _mapping(attrs.get("ptt_segment"))
    terminal = _mapping(segment.get("terminal"))
    action = str(terminal.get("action") or "")
    if action == "commit":
        return str(segment.get("stt_mode") or "") not in {"", "none"} and bool(
            str(segment.get("transcript_preview") or "").strip()
        )
    if action == "reject":
        # tap-to-stop / empty-hold paths deliberately do not call STT; the real
        # evidence is the room data/control timeline, not a provider transcript.
        return str(terminal.get("reason") or "") in {
            "tap_to_stop",
            "empty_hold",
            "short_press",
            "stt_empty",
        }
    return False


def _has_transcript_timeline_evidence(record: dict[str, Any]) -> bool:
    """Fallback STT proof for immediate control turns.

    Very fast cancel/rollback turns may flush before the provider observer has
    copied ``stt_stream.provider`` into the timeline. A transcript mark plus a
    recorded decision transcript still proves the worker processed real room
    STT events for this turn.
    """

    timestamps = _mapping(record.get("timestamps"))
    if not ("transcript_interim_first_at" in timestamps or "transcript_final_at" in timestamps):
        return False
    attrs = _mapping(record.get("attrs"))
    decision = _mapping(attrs.get("decision"))
    return bool(decision.get("transcript_preview") or attrs.get("attention_admission"))


def verify_real_call(
    case_result: CaseResult,
    *,
    runner: str,
    provider_config: dict[str, str],
    case_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return real-call failures for one case (empty means proven-real)."""

    failures = verify_provider_config(provider_config)
    metrics = case_result.metrics

    if runner == "component":
        component = case_result.case_id.rpartition(":")[2]
        if component == "tts":
            audio_bytes = metrics.get("tts_audio_bytes")
            if not isinstance(audio_bytes, (int, float)) or audio_bytes < MIN_COMPONENT_TTS_BYTES:
                failures.append(
                    f"tts_audio_bytes={audio_bytes} < {MIN_COMPONENT_TTS_BYTES} (mock/empty TTS?)"
                )
        elif component == "stt":
            if not metrics.get("stt_nonempty") and not metrics.get("stt_empty_allowed"):
                failures.append("stt produced empty transcript (no real STT output)")
        elif component == "vad":
            inference = metrics.get("vad_inference_count")
            if not isinstance(inference, (int, float)) or inference <= 0:
                failures.append("vad emitted no inference events")
    elif runner == "livekit_room":
        audio_bytes = metrics.get("agent_audio_bytes")
        agent_audio_expected = str(metrics.get("expected_agent_audio_response") or "auto")
        if agent_audio_expected != "none" and (
            not isinstance(audio_bytes, (int, float)) or audio_bytes < MIN_ROOM_AGENT_AUDIO_BYTES
        ):
            failures.append(
                f"agent_audio_bytes={audio_bytes} < {MIN_ROOM_AGENT_AUDIO_BYTES} (dead/mock agent?)"
            )
        failures.extend(_verify_room_timeline(provider_config, case_records or []))

    return failures


def apply_real_call_verification(
    run: RunResult,
    *,
    strict: bool,
    timeline_path: Path | None = None,
    require_brain_evidence: bool = True,
) -> None:
    """Annotate each case with a real-call verdict.

    ``strict`` (default for real-provider runners) turns failures into case
    errors so the run fails. Otherwise failures are recorded advisory in
    ``metrics['real_call_warnings']``. Every case gets a boolean
    ``metrics['real_call_verified']``.
    """

    records_by_case: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    if timeline_path is not None:
        all_records = load_timeline_records(timeline_path)
        records_by_case = group_records_by_case(all_records)

    # Run-level brain proof: the eidolon_agent gRPC only needs to be shown to
    # really stream once across the run (interrupt/rollback turns legitimately
    # have none). If it never appears, every case shares the failure.
    brain_provider = (run.provider_config or {}).get("brain", "")
    run_brain_failure: str | None = None
    if (
        require_brain_evidence
        and run.runner == "livekit_room"
        and brain_provider == "eidolon_agent"
    ):
        if not any(_has_brain_rpc_evidence(record) for record in all_records):
            run_brain_failure = (
                "no real eidolon_agent brain RPC evidence in the whole run "
                "(provider/request_id + brain_request_sent/first_delta marks)"
            )

    for case in run.cases:
        case_records = records_by_case.get(_base_case_id(case.case_id), [])
        failures = verify_real_call(
            case,
            runner=run.runner,
            provider_config=run.provider_config,
            case_records=case_records,
        )
        if run_brain_failure:
            failures.append(run_brain_failure)
        case.metrics["real_call_verified"] = not failures
        if not failures:
            continue
        if strict:
            case.errors.extend(f"real-call: {failure}" for failure in failures)
            case.passed = False
        else:
            case.metrics["real_call_warnings"] = "; ".join(failures)


async def preflight_real_stack(
    checks: tuple[str, ...] = ("llm", "stt", "tts"),
    *,
    llm_timeout_sec: float = 20.0,
    stt_timeout_sec: float = 30.0,
    tts_timeout_sec: float = 45.0,
    attempts: int = 2,
) -> dict[str, Any]:
    """Exercise the configured real provider stack once before benchmarking.

    Uses the same ``SharedStageFactory.from_config`` path as the worker (no
    mocks). Returns ``{"ok": bool, "config": {...}, "results": [...]}`` so the
    benchmark can abort early with a clear message if a provider is unreachable.

    Each check gets up to ``attempts`` tries: a cold connection to an upstream
    model gateway can spike (e.g. a one-off 28s TCP stall) even when the steady
    state is ~130ms, so the first attempt warms the connection and a transient
    spike does not false-abort the whole run.
    """

    import asyncio
    import time

    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.factory import SharedStageFactory
    from eidolon.livekit.agent.providers.llm import LlmInput
    from eidolon.livekit.common.config import load_effective_config

    async def _check_llm(factory: Any) -> dict[str, Any]:
        out = await factory.llm.chat(
            LlmInput(text="请用中文简短回复：ping"),
            conn_options=APIConnectOptions(max_retry=0, timeout=llm_timeout_sec),
        )
        return {"chars": len(out.text), "preview": out.text[:80]}

    async def _check_tts(factory: Any) -> dict[str, Any]:
        return await provider_check_tts(factory)

    async def _check_stt(factory: Any) -> dict[str, Any]:
        return await provider_check_stt(factory)

    cfg = load_effective_config()
    if cfg.providers.brain_provider == "eidolon_agent" and "llm" not in checks:
        factory = SimpleNamespace(
            stt=SharedStageFactory._build_stt(cfg),
            tts=SharedStageFactory._build_tts(cfg),
            llm=SimpleNamespace(llm=None),
        )
    else:
        factory = SharedStageFactory.from_config(cfg)
    fns: dict[str, tuple[Callable[[Any], Awaitable[dict[str, Any]]], float]] = {
        "llm": (_check_llm, llm_timeout_sec + 5.0),
        "tts": (_check_tts, tts_timeout_sec),
        "stt": (_check_stt, stt_timeout_sec),
    }
    results: list[dict[str, Any]] = []
    try:
        for name in checks:
            fn, timeout = fns[name]
            started = time.monotonic()
            ok = False
            detail: dict[str, Any] = {}
            used_attempts = 0
            for attempt in range(max(1, attempts)):
                used_attempts = attempt + 1
                try:
                    detail = await asyncio.wait_for(fn(factory), timeout=timeout)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 - surface as data, never crash preflight
                    detail = {"error_type": type(exc).__name__, "error": str(exc)[:300]}
            results.append(
                {
                    "name": name,
                    "ok": ok,
                    "attempts": used_attempts,
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "detail": detail,
                }
            )
    finally:
        await factory.stt.shutdown()
        await factory.tts.shutdown()
        if getattr(factory.llm, "llm", None) is not None and hasattr(factory.llm.llm, "aclose"):
            await factory.llm.llm.aclose()

    return {
        "ok": all(result["ok"] for result in results),
        "config": {
            "brain_provider": cfg.providers.brain_provider,
            "stt_provider": cfg.providers.stt_provider,
            "tts_provider": cfg.providers.tts_provider,
            "vad_provider": cfg.providers.vad_provider,
        },
        "results": results,
    }


async def preflight_runtime_identity(
    *,
    identity: str,
    kind: str,
    runtime_authority: Any | None = None,
    owner_id: str = "",
    resolver: Any | None = None,
) -> dict[str, Any]:
    """Prove a room participant can resolve before publishing test audio.

    Reaching RTC/STT without being able to mint the brain token is not
    equivalent to a real device session. Keep this boundary separate from the
    provider health preflight so a failure is attributed to its actual cause.
    """

    participant_id = identity.strip()
    participant_kind = kind.strip().lower()
    if not participant_id:
        return {"ok": False, "kind": participant_kind, "error": "identity is empty"}
    if participant_kind not in {"device", "user", "owner"}:
        return {
            "ok": False,
            "kind": participant_kind,
            "identity": participant_id,
            "error": f"unsupported participant kind {participant_kind!r}",
        }

    scoped_owner_id = owner_id.strip()
    if participant_kind == "device" and not scoped_owner_id:
        return {
            "ok": False,
            "kind": participant_kind,
            "identity": participant_id,
            "error": "device preflight requires owner_id for Kernel namespace scope",
        }
    if participant_kind in {"user", "owner"} and not scoped_owner_id:
        scoped_owner_id = participant_id

    owned_services = None
    try:
        if resolver is None:
            if runtime_authority is None:
                raise RuntimeError("runtime_authority config is required")
            from eidolon.livekit.agent.factory import _build_runtime_services

            owned_services = _build_runtime_services(runtime_authority)
            resolver = owned_services
        participant = SimpleNamespace(
            identity=participant_id,
            metadata=json.dumps({"kind": participant_kind, "owner_id": scoped_owner_id}),
        )
        room = SimpleNamespace(remote_participants={participant_id: participant})
        context = await resolver.resolve_room(room)
        return {
            "ok": True,
            "kind": participant_kind,
            "identity": participant_id,
            "owner_id": context.owner_id,
            "companion_id": context.companion_id,
            "device_id": context.device_id,
        }
    except Exception as exc:  # noqa: BLE001 - serialize the preflight failure
        return {
            "ok": False,
            "kind": participant_kind,
            "identity": participant_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if owned_services is not None:
            await owned_services.aclose()


def provider_config_from_cfg(cfg: Any) -> dict[str, str]:
    """Extract the {brain,stt,tts,vad} provider names from effective config."""

    providers = cfg.providers
    return {
        "brain": providers.brain_provider,
        "stt": providers.stt_provider,
        "tts": providers.tts_provider,
        "vad": providers.vad_provider,
    }
