"""The startup check that says whether this process runs what the lock pins.

The incident this guards is 2026-09-07: livekit-agents 1.6.4 in a venv whose
repository declared ``>=1.7.1,<1.8`` and whose lock pinned 1.7.1.  Every layer
above reported success — the Provider answered 200, the device showed
「正在聆听」 — and the only evidence was a ``TypeError`` 1.5 s into every
session.  So the headline tests here are about what gets *refused*, and about
the list of packages being derived rather than typed, because a typed list is
the thing that would have rotted before the next incident.
"""

from __future__ import annotations

import logging
import textwrap
import tomllib
from pathlib import Path

import pytest

from eidolon import locked_environment
from eidolon.locked_environment import (
    DISTRIBUTION,
    DRIFT_OVERRIDE_ENV,
    EnvironmentDriftError,
    find_lock,
    read_report,
    require_locked_environment,
    verify_locked_environment,
)

_REPOSITORY = Path(__file__).resolve().parents[3]


def _write_lock(
    directory: Path,
    *,
    root: str = "test-root",
    dependencies: tuple[tuple[str, str | None], ...] = (),
    pins: tuple[tuple[str, str], ...] = (),
    dev: tuple[str, ...] = (),
) -> Path:
    """A minimal ``uv.lock`` in the shape uv writes, with only what is read."""

    declared = "\n".join(
        f'    {{ name = "{name}", specifier = "{specifier}" }},'
        if specifier
        else f'    {{ name = "{name}" }},'
        for name, specifier in dependencies
    )
    declared += "\n" + "\n".join(
        f'    {{ name = "{name}", marker = "extra == \'dev\'", specifier = ">=1" }},'
        for name in dev
    )
    runtime = "\n".join(f'    {{ name = "{name}" }},' for name, _ in dependencies)
    optional = "\n".join(f'    {{ name = "{name}" }},' for name in dev)
    pinned = "\n".join(
        textwrap.dedent(
            f"""
            [[package]]
            name = "{name}"
            version = "{version}"
            source = {{ registry = "https://pypi.org/simple" }}
            """
        )
        for name, version in pins
    )
    lock = directory / "uv.lock"
    lock.write_text(
        textwrap.dedent(
            f"""
            version = 1
            revision = 3
            requires-python = "==3.13.*"

            [[package]]
            name = "{root}"
            version = "0.1.0"
            source = {{ editable = "." }}
            dependencies = [
            {runtime}
            ]

            [package.optional-dependencies]
            dev = [
            {optional}
            ]

            [package.metadata]
            requires-dist = [
            {declared}
            ]
            """
        )
        + pinned,
        encoding="utf-8",
    )
    return lock


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.locked_environment")


# --- the real repository ----------------------------------------------------


def test_the_real_environment_can_run_the_code_this_repository_locked(logger) -> None:
    report = verify_locked_environment(logger=logger, start=_REPOSITORY)

    assert report is not None
    assert report.lock == _REPOSITORY / "uv.lock"
    assert not report.blocking, report.refusal()


def test_the_checked_packages_are_the_declared_ones_and_not_a_list_kept_here() -> None:
    # The judgement "these are the packages whose API we call" rots on the first
    # dependency added. Deriving it from the lock means `uv lock` maintains it.
    pyproject = tomllib.loads((_REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
        for requirement in pyproject["project"]["dependencies"]
    }

    report = read_report(_REPOSITORY / "uv.lock")

    assert {verdict.name for verdict in report.verdicts} == {
        name.replace("_", "-").lower() for name in declared
    }


def test_the_dev_extra_is_not_required_on_a_host_that_has_no_pytest() -> None:
    report = read_report(_REPOSITORY / "uv.lock")

    assert {"pytest", "ruff", "mypy"} & {verdict.name for verdict in report.verdicts} == set()


# --- what gets refused ------------------------------------------------------


def test_the_2026_09_07_environment_refuses_to_start(monkeypatch, tmp_path, logger) -> None:
    # Exactly what was installed that day, against exactly what was declared.
    # It was not only the package that raised: all three LiveKit distributions
    # were below their floors, because the declaration moved on 8-31 and the
    # venv, built 8-29, never followed. Reporting one and stopping would send
    # someone to fix a third of it.
    installed = {
        "livekit-agents": "1.6.4",
        "livekit-api": "1.1.0",
        "livekit-plugins-openai": "1.6.4",
    }
    monkeypatch.setattr(locked_environment, "installed_version", lambda name: installed[name])
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(
            ("livekit-agents", ">=1.7.1,<1.8"),
            ("livekit-api", ">=1.2.1,<1.3"),
            ("livekit-plugins-openai", ">=1.7.1,<1.8"),
        ),
        pins=(
            ("livekit-agents", "1.7.1"),
            ("livekit-api", "1.2.1"),
            ("livekit-plugins-openai", "1.7.1"),
        ),
    )

    with pytest.raises(EnvironmentDriftError) as refusal:
        verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)

    report = refusal.value.report
    assert [verdict.status for verdict in report.blocking] == ["outside_declared_range"] * 3
    assert {verdict.installed for verdict in report.blocking} == {"1.6.4", "1.1.0"}
    # The one that actually raised the TypeError, named with both claims.
    agents = next(verdict for verdict in report.blocking if verdict.name == "livekit-agents")
    assert agents.locked == ("1.7.1",)
    assert agents.declared == ">=1.7.1,<1.8"
    # Every low package is in the refusal, not just the first one found.
    for name in installed:
        assert name in str(refusal.value)
    assert "uv sync" in str(refusal.value)


def test_a_version_above_the_declared_ceiling_refuses_too(monkeypatch, tmp_path, logger) -> None:
    # `<1.8` is as much a claim about the API as `>=1.7.1` is.
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.8.0")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("livekit-agents", ">=1.7.1,<1.8"),),
        pins=(("livekit-agents", "1.7.1"),),
    )

    with pytest.raises(EnvironmentDriftError):
        verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)


def test_a_declared_dependency_that_is_not_installed_refuses(tmp_path, logger) -> None:
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("no-such-distribution-anywhere", ">=1"),),
        pins=(("no-such-distribution-anywhere", "1.0.0"),),
    )

    with pytest.raises(EnvironmentDriftError) as refusal:
        verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)

    verdict = refusal.value.report.blocking[0]
    assert verdict.status == "not_installed"
    assert verdict.installed is None


def test_the_entry_point_wrapper_exits_instead_of_raising(monkeypatch, tmp_path, caplog) -> None:
    # A wrong environment is not a bug in the code, and a traceback reads like
    # the opposite — so a refusal is one CRITICAL line and a non-zero exit.
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.6.4")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("livekit-agents", ">=1.7.1,<1.8"),),
        pins=(("livekit-agents", "1.7.1"),),
    )
    monkeypatch.delenv(DRIFT_OVERRIDE_ENV, raising=False)

    with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit) as exit_:
        require_locked_environment(
            logger=logging.getLogger("test.exit"), start=tmp_path
        )

    assert exit_.value.code == 1
    assert "uv sync" in caplog.text


# --- what only warns -------------------------------------------------------


def test_drift_inside_the_declared_range_warns_and_starts(monkeypatch, tmp_path, caplog) -> None:
    # The API this code calls is still there; a silent voice channel would be a
    # worse outcome than an environment that is merely not the tested one.
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.7.5")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("livekit-agents", ">=1.7.1,<1.8"),),
        pins=(("livekit-agents", "1.7.1"),),
    )

    with caplog.at_level(logging.WARNING):
        report = verify_locked_environment(
            logger=logging.getLogger("test.drift"), start=tmp_path, allow_drift=False
        )

    assert report is not None
    assert not report.blocking
    assert [verdict.status for verdict in report.drifting] == ["lock_drift"]
    assert "uv sync" in caplog.text


def test_the_override_starts_a_refused_environment_and_names_itself(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.6.4")
    monkeypatch.setenv(DRIFT_OVERRIDE_ENV, "1")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("livekit-agents", ">=1.7.1,<1.8"),),
        pins=(("livekit-agents", "1.7.1"),),
    )

    with caplog.at_level(logging.ERROR):
        report = verify_locked_environment(
            logger=logging.getLogger("test.override"), start=tmp_path
        )

    assert report is not None and report.blocking
    assert DRIFT_OVERRIDE_ENV in caplog.text


def test_a_pin_forked_by_platform_marker_matches_either_entry(
    monkeypatch, tmp_path, logger
) -> None:
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.26.0")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("onnxruntime", ">=1.19"),),
        pins=(("onnxruntime", "1.25.0"), ("onnxruntime", "1.26.0")),
    )

    report = verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)

    assert report is not None
    assert [verdict.status for verdict in report.verdicts] == ["ok"]


def test_a_release_that_only_differs_in_trailing_zeros_is_not_drift(
    monkeypatch, tmp_path, logger
) -> None:
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "2.4.0")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("addict", ">=2.4"),),
        pins=(("addict", "2.4"),),
    )

    report = verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)

    assert report is not None and report.matches


# --- what cannot be checked at all ------------------------------------------


def test_a_lock_that_belongs_to_another_project_is_not_believed(tmp_path, caplog) -> None:
    # Walking up from an unexpected install location must not silently adopt
    # whatever lock it happens to reach.
    _write_lock(tmp_path, root="some-other-service", dependencies=(("addict", ">=2.4"),))

    with caplog.at_level(logging.ERROR):
        report = verify_locked_environment(
            logger=logging.getLogger("test.foreign"), start=tmp_path
        )

    assert report is None
    assert "some-other-service" in caplog.text


def test_a_missing_lock_is_reported_rather_than_assumed_fine(tmp_path, caplog) -> None:
    with caplog.at_level(logging.ERROR):
        report = verify_locked_environment(
            logger=logging.getLogger("test.nolock"), start=tmp_path / "nowhere"
        )

    assert report is None
    assert "uv.lock" in caplog.text


def test_a_specifier_the_lock_cannot_parse_is_neither_ok_nor_a_refusal(
    monkeypatch, tmp_path, logger
) -> None:
    monkeypatch.setattr(locked_environment, "installed_version", lambda _name: "1.7.1")
    _write_lock(
        tmp_path,
        root=DISTRIBUTION,
        dependencies=(("livekit-agents", "not-a-specifier"),),
        pins=(("livekit-agents", "1.7.1"),),
    )

    report = verify_locked_environment(logger=logger, start=tmp_path, allow_drift=False)

    assert report is not None
    assert [verdict.status for verdict in report.unverifiable] == ["unverifiable"]
    assert not report.blocking


def test_a_bug_in_the_check_does_not_take_the_service_down(monkeypatch, caplog) -> None:
    def _explode(_lock, _distribution=DISTRIBUTION):
        raise RuntimeError("the checker is broken")

    monkeypatch.setattr(locked_environment, "read_report", _explode)

    with caplog.at_level(logging.ERROR):
        report = verify_locked_environment(
            logger=logging.getLogger("test.bug"), start=_REPOSITORY
        )

    assert report is None
    assert "unverified" in caplog.text


def test_find_lock_walks_up_from_the_installed_package() -> None:
    assert find_lock(Path(locked_environment.__file__).parent) == _REPOSITORY / "uv.lock"


# --- both processes actually ask -------------------------------------------


class _Asked(Exception):
    """Raised by the stubbed check to prove nothing ran before it."""


def test_the_provider_asks_before_it_reads_its_configuration(monkeypatch) -> None:
    from eidolon.channel_provider import server as provider_server

    def _stub(**_kwargs):
        raise _Asked

    monkeypatch.setattr(provider_server, "require_locked_environment", _stub)

    # Reaching config load would raise a configuration error instead, so the
    # sentinel escaping is the proof that the check came first.
    with pytest.raises(_Asked):
        provider_server.main()


def test_the_worker_asks_before_it_registers_plugins(monkeypatch, tmp_path) -> None:
    from eidolon.livekit.agent import server as agent_server

    def _stub(**_kwargs):
        raise _Asked

    def _registered() -> None:  # pragma: no cover - must not be reached
        raise AssertionError("plugins were registered before the environment was checked")

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setattr(agent_server, "_configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_server, "require_locked_environment", _stub)
    monkeypatch.setattr(agent_server, "_register_plugins", _registered)

    with pytest.raises(_Asked):
        agent_server.main()
