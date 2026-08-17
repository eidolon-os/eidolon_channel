"""Channel's operations contract against Channel's own configuration.

Channel is two processes on purpose. The Provider answers Hub while a device is
being admitted; the worker joins the room and speaks. If they shared a process,
a busy conversation would make new devices unadmittable — so the contract has
to keep saying they are separate, with separate ports and separate runtime
directories.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

_REPOSITORY = Path(__file__).resolve().parents[3]
_CONTRACT = _REPOSITORY / "ops/component.toml"
_STATE_ROOT = "/var/lib/eidolon"


@pytest.fixture(scope="module")
def contract() -> dict:
    return tomllib.loads(_CONTRACT.read_text(encoding="utf-8"))


def _settings(name: str) -> dict:
    return yaml.safe_load((_REPOSITORY / "config" / name).read_text(encoding="utf-8"))


def _expand(value: str) -> str:
    return value.replace("$EIDOLON_STATE_ROOT", _STATE_ROOT)


def test_the_declared_provider_port_is_the_one_it_binds(contract: dict) -> None:
    http = _settings("channel-provider.yaml")["http"]

    assert contract["ports"]["channel_provider"]["default"] == http["port"]
    assert http["host"] == "127.0.0.1"


def test_the_declared_provider_store_is_the_one_it_opens(contract: dict) -> None:
    declared = {entry["path"] for entry in contract["state"]["authority"]}
    storage = _settings("channel-provider.yaml")["storage"]["path"]

    assert _expand(storage) in declared


def test_the_voiceprint_root_is_declared_and_declared_uncarried(
    contract: dict,
) -> None:
    root = _expand(_settings("settings.yaml")["voiceprint"]["root"])
    entry = next(
        item for item in contract["state"]["authority"] if item["path"] == root
    )

    # Enrolled voices are the most personal thing this Host holds. Not carrying
    # them in a backup is defensible; carrying them silently, or omitting them
    # silently, is not — so the reason is required to be there.
    assert entry["backup"] == "none"
    assert "consent" in entry["uncovered_reason"]


def test_the_two_processes_stay_two(contract: dict) -> None:
    units = {unit["id"] for unit in contract["units"]}
    assert units == {"eidolon-channel", "eidolon-channel-provider"}

    served = [role for unit in contract["units"] for role in unit["serves"]]
    assert len(served) == len(set(served)) == 2
    # Separate runtime directories: one restarting must not take the other's
    # socket with it.
    assert len(set(contract["state"]["runtime"])) == 2


def test_both_processes_wait_for_the_media_server_they_need(contract: dict) -> None:
    for unit in contract["units"]:
        assert "eidolon-livekit" in unit["requires"], unit["id"]


def test_a_factory_reset_removes_everything_channel_holds(contract: dict) -> None:
    removed = [Path(item) for item in contract["reset"]["factory"]]

    for entry in contract["state"]["authority"]:
        assert any(Path(entry["path"]).is_relative_to(root) for root in removed), (
            f"{entry['path']} would survive a factory reset"
        )
