"""Typed schema for realtime voice benchmark cases and results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


RunnerName = Literal["policy", "headless", "component", "livekit_room"]

# Room names are "{prefix}-{case_id}-{8-hex}". The livekit_room runner builds
# them and timeline_expectations parses them back, so both sides share this.
ROOM_NAME_PREFIX = "voice-bench"


@dataclass(frozen=True)
class AudioClip:
    id: str
    text: str
    path: str
    intent: str = "normal"


@dataclass(frozen=True)
class UserStep:
    text: str
    audio: str
    start_ms: int
    duration_ms: int = 700
    interims: tuple[str, ...] = ()
    eot_scores: tuple[float, ...] = ()
    final_delay_ms: int = 80
    vad_probability: float = 0.9
    agent_speaking: bool = True
    client_playback_state: str = "unknown"
    client_ptt: bool = False
    client_manual_interrupt: bool = False
    client_mic_muted: bool = False


@dataclass(frozen=True)
class AgentReply:
    when: str
    reply: str


@dataclass(frozen=True)
class DogfoodDevice:
    """Device-side behavior for human+device dogfood scenarios.

    This is benchmark DSL only. Runtime code should continue to consume the
    actual wire packets, not import benchmark types.
    """

    model: str = "generic"
    mode: str = "full_duplex"
    audio_state_hz: float = 2.0
    playback_ack: str = "simulated"
    state_jitter_ms: int = 0


@dataclass(frozen=True)
class DogfoodAgentPlayback:
    """Synthetic agent playback used to model mic echo in dogfood cases."""

    speaking_text: str = ""
    tts_duration_ms: int = 0
    echo_source: str = "synthetic_voice"


@dataclass(frozen=True)
class DogfoodEcho:
    enabled: bool = False
    delay_ms: int = 80
    attenuation_db: float = -18.0


@dataclass(frozen=True)
class DogfoodNoise:
    enabled: bool = False
    type: str = "room"
    snr_db: float | None = None
    amplitude: float = 0.0


@dataclass(frozen=True)
class DogfoodAcoustics:
    """Acoustic conditions for synthetic dogfood mic rendering."""

    echo: DogfoodEcho = field(default_factory=DogfoodEcho)
    noise: DogfoodNoise = field(default_factory=DogfoodNoise)


@dataclass(frozen=True)
class DogfoodSpec:
    """Human + device dogfood extension for a benchmark case.

    Existing policy/headless/component runners can ignore it safely. Room and
    future HIL runners use it to emulate device state cadence and acoustic mess.
    """

    enabled: bool = False
    device: DogfoodDevice = field(default_factory=DogfoodDevice)
    agent: DogfoodAgentPlayback = field(default_factory=DogfoodAgentPlayback)
    acoustics: DogfoodAcoustics = field(default_factory=DogfoodAcoustics)


@dataclass(frozen=True)
class Expectations:
    action: str = "none"
    intent: str = "uncertain"
    decision_action: str = ""
    decision_intent: str = ""
    voiceprint: str = "any"
    brain: str = "any"
    rejected_turn_brain: str = "any"
    agent_audio_response: str = "auto"
    canonical_contains: tuple[str, ...] = ()
    allow_attention_actions: tuple[str, ...] = ()
    forbid_actions: tuple[str, ...] = ()
    topic_switch_hint: bool = False
    correction_hint: bool = False
    agent_audio_cancelled: bool = False
    min_user_finals: int = 1
    min_agent_messages: int = 0
    max_interrupt_decision_ms: float | None = None
    max_interrupt_resolution_after_started_ms: float | None = None
    # Turn-segmentation correctness (timeline source). A natural pause that is
    # wrongly split shows up as extra brain requests; a multi-turn flow that
    # silently drops a turn shows up as missing ones.
    min_brain_requests: int | None = None
    max_brain_requests: int | None = None
    # End-of-turn responsiveness bound for committed turns (timeline source).
    # Per-case because merge-window cases wait longer by design, which would
    # pollute a suite-wide percentile gate.
    max_speech_stop_to_commit_ms: float | None = None
    # Room-participant bound on user-audio-done -> next agent audio. Doubles as
    # the resume-latency bound for false-interruption recovery cases.
    max_user_done_to_agent_audio_ms: float | None = None
    # Dogfood / timeline-level assertions. They may be enforced by room timeline
    # expectations or future HIL runners; deterministic runners simply carry
    # them through reports.
    max_speech_start_to_suspend_ms: float | None = None
    max_speech_start_to_cancel_ms: float | None = None
    max_speech_start_to_resume_ms: float | None = None
    playback_stop_sent: bool | None = None
    ptt_terminal_action: str = ""
    ptt_terminal_reason: str = ""
    no_full_assistant_context_commit: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    suite: str
    description: str
    audio_clips: tuple[AudioClip, ...]
    user_steps: tuple[UserStep, ...]
    agent_replies: tuple[AgentReply, ...] = ()
    dogfood: DogfoodSpec = field(default_factory=DogfoodSpec)
    expectations: Expectations = field(default_factory=Expectations)
    tags: tuple[str, ...] = ()
    timeout_sec: float = 15.0


@dataclass(frozen=True)
class BenchmarkSuite:
    suite_id: str
    cases: tuple[BenchmarkCase, ...]


@dataclass
class CaseResult:
    case_id: str
    suite: str
    runner: RunnerName
    passed: bool
    metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    git_sha: str
    runner: RunnerName
    profile: str
    cases: list[CaseResult]
    provider_config: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for case in self.cases:
                f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def _tuple_of(cls, raw: Any) -> tuple:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"expected list for {cls.__name__}, got {type(raw).__name__}")
    return tuple(cls(**item) for item in raw)


def _dogfood_spec(raw: Any) -> DogfoodSpec:
    if raw is None:
        return DogfoodSpec()
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping for DogfoodSpec, got {type(raw).__name__}")
    device = DogfoodDevice(**dict(raw.get("device") or {}))
    agent = DogfoodAgentPlayback(**dict(raw.get("agent") or {}))
    acoustics_raw = dict(raw.get("acoustics") or {})
    acoustics = DogfoodAcoustics(
        echo=DogfoodEcho(**dict(acoustics_raw.get("echo") or {})),
        noise=DogfoodNoise(**dict(acoustics_raw.get("noise") or {})),
    )
    enabled = bool(raw.get("enabled", True))
    return DogfoodSpec(
        enabled=enabled,
        device=device,
        agent=agent,
        acoustics=acoustics,
    )


def load_suite(path: str | Path) -> BenchmarkSuite:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cases: list[BenchmarkCase] = []
    for case_raw in raw.get("cases", []):
        expect_raw = dict(case_raw.get("expect") or {})
        if isinstance(expect_raw.get("canonical_contains"), list):
            expect_raw["canonical_contains"] = tuple(expect_raw["canonical_contains"])
        expectations = Expectations(**expect_raw)
        cases.append(
            BenchmarkCase(
                case_id=case_raw["case_id"],
                suite=case_raw.get("suite") or raw.get("suite_id") or p.stem,
                description=case_raw.get("description", ""),
                audio_clips=_tuple_of(AudioClip, case_raw.get("audio_clips")),
                user_steps=_tuple_of(UserStep, case_raw.get("user_steps")),
                agent_replies=_tuple_of(AgentReply, case_raw.get("agent_replies")),
                dogfood=_dogfood_spec(case_raw.get("dogfood")),
                expectations=expectations,
                tags=tuple(case_raw.get("tags") or ()),
                timeout_sec=float(case_raw.get("timeout_sec") or 15.0),
            )
        )
    return BenchmarkSuite(
        suite_id=raw.get("suite_id") or p.stem,
        cases=tuple(cases),
    )


def load_suites(paths: list[str | Path]) -> list[BenchmarkSuite]:
    return [load_suite(path) for path in paths]
