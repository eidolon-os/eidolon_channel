#!/usr/bin/env python3
"""Will the release already on a Host still read the settings we are about to ship?

Ops renders ``config/settings.yaml`` into every product Host's
``/etc/eidolon/channel.yaml``. Under ``--cutover-mode reversible`` the staged
file is loaded by *two* interpreters before any live path is touched: the
candidate release's and the one already running. The second is the rollback
this mode is the promise of, and the loader rejects a field it does not know —
so a key added to this document in the same release that adds the field to the
schema strands the running release and the deploy stops at
``cross-release channel settings validation failed``.

That check lives on the Host (eidolon_ops ``host_application``), which is the
right place for it: only the Host knows which commit is actually installed. The
cost is that you learn on the Pi, mid-deploy. This rehearses the same check
here, against a baseline you name::

    scripts/check_settings_cross_release.py 1650c17d

One approximation: the baseline's loader runs under *this* checkout's
interpreter, not the venv that release was built with. The question is which
fields the schema accepts, and that is source, not environment — but a baseline
whose loader needs an import this venv lacks will report an error rather than a
verdict. No credentials are needed; ``validate_effective_config`` checks the
document's shape, not whether its secrets authenticate.

A release that must ship a key its predecessor cannot read has one honest way
out and it is not this script: ``--cutover-mode forward-only``, which does not
restore old interpreters and so is not held to rollback safety. It gives up the
rollback to buy the key.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]

#: What the Host runs, verbatim — this is the ``script`` eidolon_ops
#: ``_validate_product_settings_compatibility`` names for the channel component.
_LOAD = "from eidolon.livekit.common.config import load_effective_config; load_effective_config()"

#: The Host path contract the deployer exports. A settings value that resolves
#: through one of these expands the same way here as it will there.
_ROOTS = {
    "EIDOLON_RUNTIME_ROOT": "/run/eidolon",
    "EIDOLON_STATE_ROOT": "/var/lib/eidolon",
    "EIDOLON_CACHE_ROOT": "/var/cache/eidolon",
    "EIDOLON_LOG_ROOT": "/var/log/eidolon",
}


def _export(revision: str, destination: Path) -> None:
    """Lay out one revision's source tree, without touching the working copy."""
    archive = subprocess.run(
        ("git", "archive", "--format=tar", revision),
        cwd=_REPOSITORY,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r|") as tar:
        tar.extractall(destination, filter="data")


def _loads(tree: Path, settings: Path) -> str | None:
    """Return why ``tree``'s loader refuses ``settings``, or None if it accepts."""
    completed = subprocess.run(
        (sys.executable, "-c", _LOAD),
        # ``-c`` puts the working directory first on sys.path, ahead of
        # PYTHONPATH — and both trees hold a top-level ``eidolon``. Running in
        # the tree under test is what makes the answer be about that tree; with
        # PYTHONPATH alone this reports on the working copy twice and passes
        # everything.
        cwd=tree,
        env={
            **os.environ,
            **_ROOTS,
            "PYTHONPATH": str(tree),
            "EIDOLON_CHANNEL_SETTINGS_YAML": str(settings),
            # A Host reads its own channel.env; nothing here may read a
            # developer's, or the verdict would depend on whose laptop it is.
            "EIDOLON_CHANNEL_ENV_FILE": "",
            "EIDOLON_CHANNEL_LIVEKIT_ENV": "",
            "EIDOLON_CHANNEL_SETTINGS_OVERLAY_YAML": "",
        },
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return None
    return completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "no output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "baseline",
        help="the revision already installed on the Host (a commit, not a tag: "
        "Ops treats the commit as the release identity and a tag as a label on it)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=_REPOSITORY / "config" / "settings.yaml",
        help="the document Ops renders (default: config/settings.yaml)",
    )
    arguments = parser.parse_args()

    settings = arguments.settings.resolve()
    if not settings.is_file():
        print(f"no such settings document: {settings}", file=sys.stderr)
        return 2

    installed = f"baseline {arguments.baseline}"
    with tempfile.TemporaryDirectory() as workspace:
        baseline = Path(workspace) / "baseline"
        _export(arguments.baseline, baseline)
        refusals = {
            installed: _loads(baseline, settings),
            "candidate (working tree)": _loads(_REPOSITORY, settings),
        }

    for interpreter, refusal in refusals.items():
        print(f"{interpreter}: {'refuses' if refusal else 'loads'} {settings.name}")
        if refusal:
            print(f"    {refusal}")

    if refusals[installed] is not None:
        sys.stdout.flush()
        print(
            "\nA reversible cutover from this baseline will fail in host_application.\n"
            "Either hold the new keys back one release — the field and its default "
            "ship now, the key follows — or deploy --cutover-mode forward-only and "
            "give up the rollback.",
            file=sys.stderr,
        )
    return 1 if any(refusals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
