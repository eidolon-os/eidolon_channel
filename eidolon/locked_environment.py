"""Say, at startup, whether this process is running what the repository locked.

On 2026-09-07 the agent worker joined every room and died 1.5 s later on
``TypeError: got an unexpected keyword argument 'transcription_timeout'``.  The
source was right the whole time: the keyword is a field of ``AgentSession`` in
livekit-agents 1.7.1, which is what ``pyproject.toml`` declares and what
``uv.lock`` pins.  The venv held 1.6.4.  Nothing in the running system could say
so — the Provider started, connected, dispatched and answered 200 throughout,
``device-channels/current`` kept reporting success, and the phone truthfully
showed 「正在聆听」.  The only symptom was a TypeError on a code path nobody
watches, and finding it cost two hours.

Both Channel processes now call :func:`verify_locked_environment` before they
serve anything.  It compares the versions actually installed against the two
statements this repository already makes about them:

* the **declared range** — ``pyproject.toml``'s specifier for a direct
  dependency, which ``uv.lock`` copies into ``[package.metadata]
  requires-dist``.  This is the code's own claim about which API it was written
  against.
* the **lock pin** — the exact version ``uv.lock`` resolved.  This is the
  version the tests ran against.

Fail closed, or log?  Graded, because those two statements do not carry the same
weight.  An install *outside the declared range* means this code is calling an
API that provably is not there; refusing is not a working deploy being bricked,
it is a broken deploy being named, so the process stops.  An install that
satisfies the range but differs from the pin means reproducibility is gone while
the API contract still holds — a silent voice channel would be worse than the
drift, so that logs at WARNING and continues.  A declared dependency that is not
installed at all also stops.  When a Host genuinely has to run outside the
declared range, ``EIDOLON_ALLOW_ENVIRONMENT_DRIFT=1`` downgrades the refusal to
an ERROR line that names the override: ignorable, but never silent.

Which packages?  The direct runtime dependencies, read out of the lock's own
entry for this project rather than listed here.  A hand-kept list of "the
packages whose API we call" is a judgement that rots on the first dependency
added, so deriving it means ``uv lock`` extends this check for free.  The ``dev``
extra is excluded — a production Host is right not to have pytest installed.

Where does this belong?  ``eidolon_ops``'s ``device_management_gate`` binds
per-repo commits into release artifacts and verifies them, which is the same
idea one layer up, at release time.  It cannot answer *this* question: only the
process holding an environment can say what is installed in it, and
``eidolon_ops`` is imported by neither Channel process.  So the call site has to
be here.  The mechanism, though, knows nothing about Channel — it takes a
distribution name and a directory — and belongs in ``eidolon_sdk`` as soon as a
second service wants it, which is a move of this file and no change to it.
Until then ``python -m eidolon.locked_environment`` lets a deploy step ask the
same question without starting a service.
"""

from __future__ import annotations

import logging
import os
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
from typing import Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

#: The distribution this repository builds.  Checked against the lock's root
#: project so that a ``uv.lock`` found by walking up from an unexpected install
#: location is reported as unverifiable rather than quietly believed.
DISTRIBUTION = "eidolon-channel"

#: Set to a truthy value to downgrade a refusal to start into an ERROR line.
DRIFT_OVERRIDE_ENV = "EIDOLON_ALLOW_ENVIRONMENT_DRIFT"

#: ``--frozen`` because the environment is what is wrong, not the lock.
#: ``--extra dev`` because ``dev`` is an optional-dependency extra here, and a
#: plain ``uv sync`` therefore uninstalls pytest, ruff and mypy.  A production
#: Host that wants none of those drops the flag.
REMEDY = "uv sync --frozen --extra dev"

_LOCK_FILENAME = "uv.lock"
_LOG_PREFIX = "[Env]"

Status = Literal[
    "ok",
    "lock_drift",
    "outside_declared_range",
    "not_installed",
    "unverifiable",
]

_BLOCKING: frozenset[Status] = frozenset({"outside_declared_range", "not_installed"})


class LockedEnvironmentError(Exception):
    """Base for everything this module raises."""


class EnvironmentUnverifiable(LockedEnvironmentError):
    """The lock could not be found, or does not describe this distribution."""


class EnvironmentDriftError(LockedEnvironmentError):
    """A direct dependency is installed at a version this code cannot run on."""

    def __init__(self, report: EnvironmentReport) -> None:
        self.report = report
        super().__init__(report.refusal())


@dataclass(frozen=True)
class PackageVerdict:
    """What one direct dependency's installed version is, against both claims."""

    name: str
    status: Status
    installed: str | None
    locked: tuple[str, ...]
    declared: str | None
    detail: str | None = None

    @property
    def blocks_startup(self) -> bool:
        return self.status in _BLOCKING

    def describe(self) -> str:
        locked = " or ".join(self.locked) if self.locked else "nothing"
        if self.status == "not_installed":
            return (
                f"{self.name} is a declared dependency but is not installed; "
                f"uv.lock pins {locked}"
            )
        if self.status == "outside_declared_range":
            return (
                f"{self.name} {self.installed} is installed, but this repository declares "
                f"{self.declared} and uv.lock pins {locked}"
            )
        if self.status == "lock_drift":
            return (
                f"{self.name} {self.installed} is installed but uv.lock pins {locked}; "
                f"it satisfies the declared {self.declared or 'range'}, so the API this "
                f"code calls should be there, but this is not what the tests ran against"
            )
        if self.status == "unverifiable":
            return f"{self.name} could not be checked: {self.detail}"
        return f"{self.name} {self.installed} matches uv.lock"


@dataclass(frozen=True)
class EnvironmentReport:
    """Every direct runtime dependency of ``distribution``, judged."""

    lock: Path
    distribution: str
    verdicts: tuple[PackageVerdict, ...]

    def _with_status(self, *statuses: Status) -> tuple[PackageVerdict, ...]:
        return tuple(verdict for verdict in self.verdicts if verdict.status in statuses)

    @property
    def blocking(self) -> tuple[PackageVerdict, ...]:
        """Dependencies whose installed version this code provably cannot use."""

        return tuple(verdict for verdict in self.verdicts if verdict.blocks_startup)

    @property
    def drifting(self) -> tuple[PackageVerdict, ...]:
        """Dependencies that satisfy the declared range but are not the pin."""

        return self._with_status("lock_drift")

    @property
    def unverifiable(self) -> tuple[PackageVerdict, ...]:
        return self._with_status("unverifiable")

    @property
    def matches(self) -> bool:
        return not self.blocking and not self.drifting and not self.unverifiable

    def refusal(self) -> str:
        problems = "; ".join(verdict.describe() for verdict in self.blocking)
        return (
            f"{self.distribution} is not running what {self.lock} locked: {problems}. "
            f"Run `{REMEDY}`, or set {DRIFT_OVERRIDE_ENV}=1 to start anyway."
        )


def find_lock(start: Path) -> Path:
    """The nearest ``uv.lock`` at or above ``start``.

    Both Channel units run from the repository checkout — ``ops/component.toml``
    execs ``.venv/bin/...`` inside it — so the lock sits above the installed
    package either way, editable or not.
    """

    start = start.resolve()
    for directory in (start, *start.parents):
        candidate = directory / _LOCK_FILENAME
        if candidate.is_file():
            return candidate
    raise EnvironmentUnverifiable(f"no {_LOCK_FILENAME} at or above {start}")


def _root_project(packages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    roots = [
        package
        for package in packages
        if any(
            package.get("source", {}).get(key) == "."
            for key in ("editable", "virtual", "directory")
        )
    ]
    if len(roots) != 1:
        raise EnvironmentUnverifiable(
            f"expected one root project in the lock, found {len(roots)}"
        )
    return roots[0]


def _pins(packages: Sequence[Mapping[str, Any]]) -> Mapping[str, tuple[str, ...]]:
    """Name to every version the lock pins for it.

    A lock that forks by platform marker can carry the same package twice at
    different versions; an install matching either of them is not drift.
    """

    pins: dict[str, tuple[str, ...]] = {}
    for package in packages:
        pinned = package.get("version")
        if not pinned:
            continue
        name = canonicalize_name(package["name"])
        if pinned not in pins.get(name, ()):
            pins[name] = (*pins.get(name, ()), pinned)
    return pins


def _declared_ranges(root: Mapping[str, Any]) -> Mapping[str, str]:
    """Name to the specifier ``pyproject.toml`` declares for it.

    ``extra ==`` markers are skipped: those requirements belong to an extra, and
    the runtime dependency list this walks never contains them.
    """

    declared: dict[str, str] = {}
    for requirement in root.get("metadata", {}).get("requires-dist", ()):
        if "extra == " in requirement.get("marker", ""):
            continue
        specifier = requirement.get("specifier")
        if specifier:
            declared[canonicalize_name(requirement["name"])] = specifier
    return declared


def _same_version(installed: Version, pinned: str) -> bool:
    try:
        return Version(pinned) == installed
    except InvalidVersion:
        return pinned == str(installed)


def _verdict(
    name: str,
    *,
    declared: Mapping[str, str],
    pins: Mapping[str, tuple[str, ...]],
) -> PackageVerdict:
    key = canonicalize_name(name)
    locked = pins.get(key, ())
    specifier = declared.get(key)
    try:
        found = installed_version(key)
    except PackageNotFoundError:
        return PackageVerdict(key, "not_installed", None, locked, specifier)
    try:
        parsed = Version(found)
    except InvalidVersion:
        return PackageVerdict(
            key,
            "unverifiable",
            found,
            locked,
            specifier,
            f"the installed version {found!r} is not a PEP 440 version",
        )
    if specifier is not None:
        try:
            allowed = SpecifierSet(specifier)
        except InvalidSpecifier:
            return PackageVerdict(
                key,
                "unverifiable",
                found,
                locked,
                specifier,
                f"the lock declares an unparseable specifier {specifier!r}",
            )
        # A prerelease is admitted by the range on purpose: it is not what the
        # tests ran against, which is drift, but it is not a missing API.
        if not allowed.contains(parsed, prereleases=True):
            return PackageVerdict(key, "outside_declared_range", found, locked, specifier)
    if locked and not any(_same_version(parsed, pinned) for pinned in locked):
        return PackageVerdict(key, "lock_drift", found, locked, specifier)
    return PackageVerdict(key, "ok", found, locked, specifier)


def read_report(lock: Path, distribution: str = DISTRIBUTION) -> EnvironmentReport:
    """Judge the installed environment against ``lock``, reading nothing else."""

    document = tomllib.loads(lock.read_text(encoding="utf-8"))
    packages = document.get("package", ())
    root = _root_project(packages)
    if canonicalize_name(root["name"]) != canonicalize_name(distribution):
        raise EnvironmentUnverifiable(
            f"{lock} locks {root['name']!r}, not {distribution!r}"
        )
    declared = _declared_ranges(root)
    pins = _pins(packages)
    verdicts = tuple(
        _verdict(dependency["name"], declared=declared, pins=pins)
        for dependency in root.get("dependencies", ())
    )
    return EnvironmentReport(
        lock=lock,
        distribution=canonicalize_name(distribution),
        verdicts=verdicts,
    )


def _drift_allowed(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.getenv(DRIFT_OVERRIDE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def verify_locked_environment(
    *,
    logger: logging.Logger,
    distribution: str = DISTRIBUTION,
    start: Path | None = None,
    allow_drift: bool | None = None,
) -> EnvironmentReport | None:
    """Log what the environment is, and refuse to continue if it cannot work.

    Returns the report, or ``None`` when the environment could not be checked at
    all.  Raises :class:`EnvironmentDriftError` when a direct dependency is
    installed at a version this code provably cannot call and the override is
    not set.  A failure *inside this check* is logged and swallowed: a
    diagnostic that takes the product down over its own bug is worse than no
    diagnostic.
    """

    try:
        report = read_report(find_lock(start or Path(__file__).parent), distribution)
    except EnvironmentUnverifiable as error:
        logger.error("%s cannot check the environment against uv.lock: %s", _LOG_PREFIX, error)
        return None
    except Exception:
        logger.exception(
            "%s the uv.lock check itself failed; the environment is unverified", _LOG_PREFIX
        )
        return None

    for verdict in report.unverifiable:
        logger.warning("%s %s", _LOG_PREFIX, verdict.describe())
    for verdict in report.drifting:
        logger.warning("%s %s — run `%s`", _LOG_PREFIX, verdict.describe(), REMEDY)
    for verdict in report.blocking:
        logger.error("%s %s", _LOG_PREFIX, verdict.describe())

    if not report.blocking:
        logger.info(
            "%s %d direct dependencies checked against %s: %d match the pin",
            _LOG_PREFIX,
            len(report.verdicts),
            report.lock,
            len(report.verdicts) - len(report.drifting) - len(report.unverifiable),
        )
        return report

    if _drift_allowed(allow_drift):
        logger.error(
            "%s starting anyway because %s is set — %s",
            _LOG_PREFIX,
            DRIFT_OVERRIDE_ENV,
            report.refusal(),
        )
        return report

    raise EnvironmentDriftError(report)


def require_locked_environment(
    *,
    logger: logging.Logger,
    distribution: str = DISTRIBUTION,
    start: Path | None = None,
) -> EnvironmentReport | None:
    """:func:`verify_locked_environment` for a process entry point.

    A refusal leaves one CRITICAL line naming the remedy and exits non-zero
    rather than raising through ``main``.  This is a wrong environment, not a
    bug in the code, and a traceback reads like the opposite.
    """

    try:
        return verify_locked_environment(
            logger=logger, distribution=distribution, start=start
        )
    except EnvironmentDriftError as error:
        logger.critical("%s %s", _LOG_PREFIX, error)
        raise SystemExit(1) from None


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m eidolon.locked_environment`` — the same check, as a command.

    Exits non-zero on a blocking mismatch so a deploy step can gate on it
    without importing either service.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("locked_environment")
    start = Path(argv[0]) if argv else None
    try:
        report = verify_locked_environment(logger=logger, start=start, allow_drift=False)
    except EnvironmentDriftError as error:
        logger.critical("%s %s", _LOG_PREFIX, error)
        return 1
    if report is None:
        return 2
    for verdict in report.verdicts:
        print(f"{verdict.status:24} {verdict.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
