"""Voice benchmark infrastructure tests."""

from __future__ import annotations

import statistics

from eidolon.livekit.benchmarks.compare import compare_metrics
from eidolon.livekit.benchmarks.dashboard import DashboardRunner, write_dashboard
from eidolon.livekit.benchmarks.realcall import (
    apply_real_call_verification,
    verify_provider_config,
    verify_real_call,
)
from eidolon.livekit.benchmarks.report import (
    aggregate,
    aggregate_runs,
    write_repeated_reports,
)
from eidolon.livekit.benchmarks.schema import CaseResult, RunResult, load_suite
from eidolon.livekit.benchmarks.slo import (
    DEFAULT_SLO_GATES,
    SloGate,
    enforcement_failures,
    evaluate_slo_gates,
)
from eidolon.livekit.benchmarks.timeline import (
    TimelineCapture,
    load_timeline_records,
    summarize_timeline_records,
)
from eidolon.livekit.benchmarks.timeline_expectations import apply_timeline_expectations


def test_load_core_benchmark_suite() -> None:
    suite = load_suite("benchmarks/cases/core.yaml")

    assert suite.suite_id == "core_voice_baseline"
    assert {case.case_id for case in suite.cases} >= {
        "normal_single_turn_001",
        "hard_interrupt_001",
        "backchannel_001",
    }


def test_aggregate_case_metrics() -> None:
    summary = aggregate(
        [
            CaseResult(
                case_id="a",
                suite="s",
                runner="policy",
                passed=True,
                metrics={"interrupt_decision_ms": 100},
            ),
            CaseResult(
                case_id="b",
                suite="s",
                runner="policy",
                passed=False,
                metrics={"interrupt_decision_ms": 200},
            ),
        ]
    )

    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["metrics"]["interrupt_decision_ms"]["p50"] == 150


def _repeat_runs() -> list[RunResult]:
    def run(m_a: float, b_passed: bool, m_b: float) -> RunResult:
        return RunResult(
            run_id="r",
            git_sha="sha",
            runner="policy",
            profile="p",
            cases=[
                CaseResult("a", "s", "policy", True, metrics={"m": m_a}),
                CaseResult(
                    "b",
                    "s",
                    "policy",
                    b_passed,
                    metrics={"m": m_b},
                    errors=[] if b_passed else ["boom"],
                ),
            ],
        )

    return [run(100, True, 200), run(110, False, 260)]


def test_aggregate_runs_tracks_per_case_jitter_and_flakiness() -> None:
    summary = aggregate_runs(_repeat_runs())

    assert summary["repeats"] == 2
    assert summary["total"] == 2
    # only case "a" passed in every repeat
    assert summary["passed"] == 1
    assert summary["flaky"] == 1

    case_a = summary["per_case"]["a"]
    assert case_a["runs"] == 2
    assert case_a["pass_rate"] == 1.0
    assert case_a["metrics"]["m"]["stdev"] == statistics.stdev([100, 110])

    case_b = summary["per_case"]["b"]
    assert case_b["passed"] == 1
    assert case_b["pass_rate"] == 0.5
    assert "boom" in case_b["errors"]

    # pooled distribution carries every (case x repeat) sample for SLO gates
    assert summary["metrics"]["m"]["count"] == 4


def test_write_repeated_reports_emits_stability_view(tmp_path) -> None:
    out = tmp_path / "policy"
    payload = write_repeated_reports(_repeat_runs(), out)

    assert payload["summary"]["repeats"] == 2
    assert len(payload["cases"]) == 4
    assert (out / "metrics.json").exists()
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "repeats: `2`" in markdown
    assert "Per-Case Stability" in markdown


def test_dashboard_flags_flaky_case_from_repeats(tmp_path) -> None:
    run_dir = tmp_path / "policy"
    write_repeated_reports(_repeat_runs(), run_dir)

    payload = write_dashboard(
        runners=[DashboardRunner(name="policy", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert any("flaky case b" in item["text"] for item in payload["findings"])
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "2 repeats per case" in html


def test_verify_provider_config_rejects_mock() -> None:
    assert verify_provider_config({"brain": "eidolon_agent", "stt": "bailian"}) == []
    failures = verify_provider_config({"brain": "mock_llm", "stt": "bailian"})
    assert any("brain" in f for f in failures)


def test_verify_real_call_component_tts_bytes() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    low = CaseResult(
        "c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 10}
    )
    assert verify_real_call(low, runner="component", provider_config=cfg)
    ok = CaseResult(
        "c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 50_000}
    )
    assert verify_real_call(ok, runner="component", provider_config=cfg) == []


def test_verify_real_call_room_per_case_checks_stt_and_audio() -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}
    # interrupt-style turn: no brain marks, but real STT stream + agent audio.
    rec = {
        "attrs": {
            "room_name": "voice-bench-x-12345678",
            "stt_stream": {"provider": "bailian"},
            "interrupt_action": "cancel",
        },
        "timestamps": {},
    }
    ok = CaseResult("x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000})
    # per-case no longer requires brain marks (those are run-level)
    assert verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[rec]) == []
    # low audio fails per-case
    low = CaseResult("x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 10})
    assert verify_real_call(low, runner="livekit_room", provider_config=cfg, case_records=[rec])
    # wrong/absent STT stream fails per-case
    no_stt = {"attrs": {"room_name": "voice-bench-x-12345678"}, "timestamps": {}}
    assert verify_real_call(ok, runner="livekit_room", provider_config=cfg, case_records=[no_stt])


def test_apply_real_call_verification_run_level_brain(tmp_path) -> None:
    cfg = {"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian", "vad": "firered"}

    def make_run() -> RunResult:
        return RunResult(
            run_id="r", git_sha="s", runner="livekit_room", profile="p",
            cases=[
                CaseResult("normal_x", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}),
                CaseResult("hard_y", "s", "livekit_room", True, metrics={"agent_audio_bytes": 40_000}),
            ],
            provider_config=cfg,
        )

    stt = '"stt_stream":{"provider":"bailian"}'
    brain = '"brain_rpc":{"provider":"eidolon_agent_rpc","request_id":"eidolon-abc"}'
    marks = '"brain_request_sent_at":1.0,"brain_first_delta_at":1.2'

    # one normal turn proves the brain; the interrupt turn has none -> all pass
    good = tmp_path / "good_turn_timeline.jsonl"
    good.write_text(
        f'{{"attrs":{{"room_name":"voice-bench-normal_x-12345678",{brain},{stt}}},"timestamps":{{{marks}}}}}\n'
        f'{{"attrs":{{"room_name":"voice-bench-hard_y-abcdef12",{stt}}},"timestamps":{{}}}}\n',
        encoding="utf-8",
    )
    run = make_run()
    apply_real_call_verification(run, strict=True, timeline_path=good)
    assert all(c.passed for c in run.cases)
    assert all(c.metrics["real_call_verified"] for c in run.cases)

    # brain never streams anywhere -> run-level failure on every case
    bad = tmp_path / "bad_turn_timeline.jsonl"
    bad.write_text(
        f'{{"attrs":{{"room_name":"voice-bench-normal_x-12345678",{stt}}},"timestamps":{{}}}}\n'
        f'{{"attrs":{{"room_name":"voice-bench-hard_y-abcdef12",{stt}}},"timestamps":{{}}}}\n',
        encoding="utf-8",
    )
    run2 = make_run()
    apply_real_call_verification(run2, strict=True, timeline_path=bad)
    assert all(not c.passed for c in run2.cases)
    assert all("brain RPC evidence in the whole run" in " ".join(c.errors) for c in run2.cases)


def test_apply_real_call_verification_strict_fails_mocked_run() -> None:
    run = RunResult(
        run_id="r",
        git_sha="sha",
        runner="component",
        profile="p",
        cases=[
            CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 50_000})
        ],
        provider_config={"brain": "mock_llm", "stt": "bailian", "tts": "bailian"},
    )
    apply_real_call_verification(run, strict=True)
    assert run.cases[0].passed is False
    assert run.cases[0].metrics["real_call_verified"] is False
    assert any(e.startswith("real-call:") for e in run.cases[0].errors)


def test_dashboard_flags_unverified_real_call(tmp_path) -> None:
    run = RunResult(
        run_id="rc",
        git_sha="sha",
        runner="component",
        profile="real_components",
        cases=[
            CaseResult("c:tts", "s:tts", "component", True, metrics={"tts_audio_bytes": 5})
        ],
        provider_config={"brain": "eidolon_agent", "stt": "bailian", "tts": "bailian"},
    )
    apply_real_call_verification(run, strict=True)

    run_dir = tmp_path / "component"
    write_repeated_reports([run], run_dir)
    payload = write_dashboard(
        runners=[DashboardRunner(name="component", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert any("real call NOT verified" in item["text"] for item in payload["findings"])
    assert "providers: brain=eidolon_agent" in html


def test_compare_detects_p95_regression() -> None:
    baseline = {
        "summary": {
            "failed": 0,
            "metrics": {"latency_ms": {"p95": 100}},
        }
    }
    candidate = {
        "summary": {
            "failed": 0,
            "metrics": {"latency_ms": {"p95": 130}},
        }
    }

    result = compare_metrics(baseline, candidate, max_p95_regression_pct=10)

    assert result["passed"] is False
    assert result["failures"]


def test_component_runner_name_is_supported() -> None:
    summary = aggregate(
        [
            CaseResult(
                case_id="normal_single_turn_001:vad",
                suite="core_latency:vad",
                runner="component",
                passed=True,
                metrics={"vad_elapsed_ms": 42, "vad_detected_speech": True},
            )
        ]
    )

    assert summary["passed"] == 1
    assert summary["metrics"]["vad_elapsed_ms"]["p95"] == 42


def test_livekit_room_state_marks_agent_connected() -> None:
    from eidolon.livekit.benchmarks.livekit_room_runner import _RoomCaseState

    state = _RoomCaseState(started=0.0, events=[])
    assert not state.agent_connected.is_set()

    state.mark("participant_connected_at")

    assert state.agent_connected.is_set()


def test_timeline_records_are_summarized(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        "\n".join(
            [
                (
                    '{"turn_id":"t1","durations_ms":{"commit_to_tts_first_audio":100},'
                    '"timestamps":{"speech_started_at":1.0,"transcript_final_at":1.2},'
                    '"attrs":{"decision_reason":"intent:hard_stop",'
                    '"room_name":"voice-bench-hard_interrupt_001-1234abcd",'
                    '"timeline_flush_reason":"interrupt_cancel",'
                    '"provider_latency_ms":{"stt_final_ms":200}}}'
                ),
                (
                    '{"turn_id":"t2","durations_ms":{"commit_to_tts_first_audio":300},'
                    '"timestamps":{"speech_started_at":2.0,"transcript_final_at":2.4},'
                    '"attrs":{"room_name":"voice-bench-normal_single_turn_001-deadbeef",'
                    '"timeline_flush_reason":"agent_audio_playback_done",'
                    '"provider_latency_ms":{"stt_final_ms":400}}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(tmp_path))

    assert summary["count"] == 2
    assert summary["latencies"]["commit_to_tts_first_audio"]["p95"] == 290
    assert summary["latencies"]["stt_final_ms"]["p50"] == 300
    assert round(summary["provider_segments"]["stt_final"]["p50"]) == 300
    assert summary["record_summaries"][0]["case_id"] == "hard_interrupt_001"
    assert summary["decision_reasons"]["intent:hard_stop"] == 1
    assert summary["flush_reasons"]["agent_audio_playback_done"] == 1
    assert summary["cases"]["hard_interrupt_001"] == 1
    assert summary["cases"]["normal_single_turn_001"] == 1


def test_timeline_capture_writes_only_new_lines(tmp_path) -> None:
    source = tmp_path / "worker_timeline.jsonl"
    source.write_text('{"turn_id":"old"}\n', encoding="utf-8")
    capture = TimelineCapture.start(str(source))

    with source.open("a", encoding="utf-8") as f:
        f.write('{"turn_id":"new-1"}\n')
        f.write('{"turn_id":"new-2"}\n')

    output = tmp_path / "run" / "turn_timeline.jsonl"
    written = capture.write_new_lines(output)

    assert written == 2
    assert "old" not in output.read_text(encoding="utf-8")
    assert len(load_timeline_records(output)) == 2


def test_dashboard_includes_timeline_section(tmp_path) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

    run_dir = tmp_path / "candidate"
    write_reports(
        RunResult(
            run_id="timeline-test",
            git_sha="abc123",
            runner="policy",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="policy",
                    passed=True,
                    metrics={"elapsed_ms": 1},
                )
            ],
        ),
        run_dir,
    )
    (run_dir / "turn_timeline.jsonl").write_text(
        (
            '{"turn_id":"t1","durations_ms":{"commit_to_tts_first_audio":100},'
            '"attrs":{"timeline_flush_reason":"agent_audio_playback_done"}}\n'
        ),
        encoding="utf-8",
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="policy", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert payload["runners"][0]["timeline"]["count"] == 1
    assert "policy Turn Timeline" in html
    assert "commit_to_tts_first_audio" in html
    assert "Case Coverage" in html
    assert "Provider Latency Breakdown" in html
    assert "Per-Turn Provider Segments" in html
    assert "STT Audio" in html


def test_slo_gate_detects_timeline_regression() -> None:
    runner = {
        "name": "livekit_room",
        "summary": {"metrics": {}},
        "timeline": {
            "latencies": {
                "vad_start_to_interrupt_resolved": {
                    "p95": 800.0,
                }
            }
        },
    }

    results = evaluate_slo_gates(
        runner,
        gates=(
            SloGate(
                name="interrupt",
                runner="livekit_room",
                source="timeline",
                metric="vad_start_to_interrupt_resolved",
                statistic="p95",
                max_value=650.0,
            ),
        ),
    )

    assert results[0]["passed"] is False
    assert results[0]["value"] == 800.0


def test_slo_gate_advisory_below_min_samples() -> None:
    gate = SloGate(
        name="g",
        runner="livekit_room",
        source="summary",
        metric="m",
        statistic="p95",
        max_value=650.0,
        tier="target",
        min_samples=20,
    )
    low = {"name": "livekit_room", "summary": {"metrics": {"m": {"p95": 900.0, "count": 3}}}}
    res_low = evaluate_slo_gates(low, gates=(gate,))[0]
    assert res_low["advisory"] is True
    assert res_low["passed"] is True
    assert res_low["tier"] == "target"

    enough = {"name": "livekit_room", "summary": {"metrics": {"m": {"p95": 900.0, "count": 25}}}}
    res_full = evaluate_slo_gates(enough, gates=(gate,))[0]
    assert res_full["advisory"] is False
    assert res_full["passed"] is False


def test_enforcement_failures_only_required_hard_fails() -> None:
    # The default ratchet enforces the two summary experience ceilings.
    enforced = {
        g.name for g in DEFAULT_SLO_GATES if g.required and g.source == "summary"
    }
    assert "room_user_done_to_next_audio" in enforced
    assert "room_publish_to_first_audio" in enforced

    # required gate that hard-fails -> enforcement failure
    runner = {
        "name": "livekit_room",
        "summary": {
            "metrics": {
                "user_done_to_agent_audio_after_user_done_ms": {"p95": 1800.0, "count": 25},
                "publish_to_agent_audio_first_ms": {"p95": 1000.0, "count": 25},
            }
        },
    }
    results = evaluate_slo_gates(runner)
    fails = enforcement_failures(results)
    assert [f["name"] for f in fails] == ["room_user_done_to_next_audio"]

    # an advisory (too few samples) required gate must NOT block
    advisory = evaluate_slo_gates(
        {"name": "livekit_room", "summary": {"metrics": {}}, "timeline": {"latencies": {
            "commit_to_tts_first_audio": {"p50": 9999.0, "p95": 9999.0, "count": 3},
        }}}
    )
    assert enforcement_failures(advisory) == []


def test_timeline_exposes_tts_and_endpoint_segments(tmp_path) -> None:
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","timestamps":{'
            '"turn_committed_at":1.0,"stt_provider_final_at":1.8,'
            '"tts_request_started_at":2.0,"tts_provider_first_audio_at":2.15,'
            '"tts_first_audio_at":2.20},'
            '"attrs":{"room_name":"voice-bench-normal_single_turn_001-1234abcd"}}\n'
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline_records(load_timeline_records(timeline_path))
    segments = summary["provider_segments"]

    assert round(segments["tts_request_to_first_audio"]["p50"]) == 150
    assert round(segments["tts_provider_to_agent_audio"]["p50"]) == 50
    assert round(segments["stt_final_after_commit"]["p50"]) == 800


def test_dashboard_findings_include_slo_failure(tmp_path) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="slo-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={"user_done_to_agent_audio_after_user_done_ms": 2000},
                )
            ],
        ),
        run_dir,
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert payload["findings"]
    assert any("SLO" in item["text"] for item in payload["findings"])


def test_dashboard_renders_slo_gate_table(tmp_path) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="slo-table-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="case-1",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={"user_done_to_agent_audio_after_user_done_ms": 900},
                )
            ],
        ),
        run_dir,
    )

    write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "livekit_room SLO Gates" in html
    assert "room_user_done_to_next_audio" in html
    assert "PASS" in html


def test_dashboard_warns_when_livekit_room_timeline_coverage_is_partial(
    tmp_path,
) -> None:
    from eidolon.livekit.benchmarks.report import write_reports

    run_dir = tmp_path / "livekit_room"
    write_reports(
        RunResult(
            run_id="coverage-test",
            git_sha="abc123",
            runner="livekit_room",
            profile="test",
            cases=[
                CaseResult(
                    case_id="covered",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={},
                ),
                CaseResult(
                    case_id="missing",
                    suite="suite",
                    runner="livekit_room",
                    passed=True,
                    metrics={},
                ),
            ],
        ),
        run_dir,
    )
    (run_dir / "turn_timeline.jsonl").write_text(
        (
            '{"turn_id":"t1","attrs":'
            '{"room_name":"voice-bench-covered-1234abcd"}}\n'
        ),
        encoding="utf-8",
    )

    payload = write_dashboard(
        runners=[DashboardRunner(name="livekit_room", candidate=run_dir)],
        output_path=tmp_path / "dashboard.html",
    )

    assert any("timeline coverage missing" in item["text"] for item in payload["findings"])
    assert any("missing" in item["text"] for item in payload["findings"])


def test_livekit_room_timeline_expectations_fail_unexpected_cancel(
    tmp_path,
) -> None:
    suite = load_suite("benchmarks/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="normal_single_turn_001",
                suite="core_latency",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-normal_single_turn_001-1234abcd",'
            '"interrupt_action":"cancel","decision":{"intent":"normal_interrupt"}}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is False
    assert "expected no interrupt action" in run.cases[0].errors[0]


def test_livekit_room_timeline_expectations_pass_expected_hard_stop(
    tmp_path,
) -> None:
    suite = load_suite("benchmarks/cases/core.yaml")
    run = RunResult(
        run_id="expectation-test",
        git_sha="abc123",
        runner="livekit_room",
        profile="test",
        cases=[
            CaseResult(
                case_id="hard_interrupt_001",
                suite="interrupt",
                runner="livekit_room",
                passed=True,
            )
        ],
    )
    timeline_path = tmp_path / "turn_timeline.jsonl"
    timeline_path.write_text(
        (
            '{"turn_id":"t1","attrs":{"room_name":'
            '"voice-bench-hard_interrupt_001-1234abcd",'
            '"interrupt_action":"cancel","decision":{"intent":"hard_stop"}}}\n'
        ),
        encoding="utf-8",
    )

    apply_timeline_expectations(run, [suite], timeline_path)

    assert run.cases[0].passed is True
    assert not run.cases[0].errors
