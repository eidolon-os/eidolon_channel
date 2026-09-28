"""What each trait's commands do to a virtual device, and that it is still so after a restart."""

from __future__ import annotations

import os
import sqlite3
import stat

import pytest
from eidolon_sdk.biz.smarthome import (
    DEVICE_TYPES,
    TRAIT_COMMANDS,
    Device,
    SmartHomeError,
    initial_state,
    validate_state,
)

from eidolon.capability_runtime.smarthome import VirtualProvider, apply_command

from .helpers import OTHER_OWNER, OWNER, cmd


def run(kind: str, state: dict | None, trait: str, command: str, **params):
    return apply_command(kind, state or initial_state(kind), cmd("d", trait, command, **params))


def refused(kind: str, trait: str, command: str, **params) -> str:
    with pytest.raises(SmartHomeError) as caught:
        run(kind, None, trait, command, **params)
    return caught.value.code


def test_on_off():
    assert run("switch", None, "on_off", "on")["on"] is True
    assert run("switch", {"on": True}, "on_off", "off")["on"] is False
    assert run("switch", {"on": False}, "on_off", "toggle")["on"] is True
    assert run("switch", {"on": True}, "on_off", "toggle")["on"] is False


def test_level_clamps_and_turns_an_off_light_on():
    off = {"on": False, "level": 60}
    assert run("light", off, "level", "set", value=40) == {"on": True, "level": 40}
    assert run("light", off, "level", "step", delta=10) == {"on": True, "level": 70}
    assert run("light", {"on": True, "level": 95}, "level", "step", delta=20)["level"] == 100
    assert run("light", {"on": True, "level": 5}, "level", "step", delta=-20)["level"] == 0
    assert refused("light", "level", "set", value=101) == "OUT_OF_RANGE"
    assert refused("light", "level", "set", value="50") == "INVALID_PARAMS"
    assert refused("light", "level", "set") == "INVALID_PARAMS"


def test_thermostat_stays_inside_the_type_bounds_and_turns_the_device_on():
    ac = initial_state("climate")
    assert run("climate", ac, "thermostat", "set_target", celsius=22) == ac | {
        "on": True,
        "target_c": 22,
    }
    assert run("climate", ac, "thermostat", "set_mode", mode="heat")["mode"] == "heat"
    assert run("climate", ac, "thermostat", "set_mode", mode="heat")["on"] is True
    assert run("climate", ac | {"target_c": 29}, "thermostat", "step", delta=5)["target_c"] == 30
    assert run("climate", ac | {"target_c": 17}, "thermostat", "step", delta=-5)["target_c"] == 16
    assert run("climate", ac, "thermostat", "step", delta=0.5)["target_c"] == 26.5
    whole = run("climate", ac, "thermostat", "set_target", celsius=24.0)["target_c"]
    assert whole == 24 and type(whole) is int
    heater = initial_state("water_heater")
    assert (
        run("water_heater", heater | {"target_c": 58}, "thermostat", "step", delta=5)["target_c"]
        == 60
    )
    assert refused("climate", "thermostat", "set_target", celsius=31) == "OUT_OF_RANGE"
    assert refused("water_heater", "thermostat", "set_target", celsius=30) == "OUT_OF_RANGE"
    assert refused("water_heater", "thermostat", "set_mode", mode="cool") == "OUT_OF_RANGE"


def test_fan_speed_position_lock_operational_and_volume():
    assert run("fan", None, "fan_speed", "set", value=80)["speed"] == 80
    cover = {"position": 40}
    assert run("cover", cover, "position", "open") == {"position": 100}
    assert run("cover", cover, "position", "close") == {"position": 0}
    assert run("cover", cover, "position", "stop") == cover
    assert run("cover", cover, "position", "set", value=65) == {"position": 65}
    assert run("lock", {"locked": True}, "lock", "unlock") == {"locked": False}
    assert run("lock", {"locked": False}, "lock", "lock") == {"locked": True}
    for verb, run_state in [
        ("start", "running"),
        ("pause", "paused"),
        ("stop", "idle"),
        ("dock", "docked"),
    ]:
        assert run("appliance", None, "operational", verb) == {"run_state": run_state}
    media = {"on": True, "volume": 90, "muted": False}
    assert run("media", media, "volume", "set", value=30)["volume"] == 30
    assert run("media", media, "volume", "step", delta=25)["volume"] == 100
    assert run("media", media | {"volume": 10}, "volume", "step", delta=-25)["volume"] == 0
    assert run("media", media, "volume", "mute", muted=True)["muted"] is True


def test_measure_and_foreign_traits_take_no_commands():
    assert refused("sensor", "measure", "set", value=1) == "UNSUPPORTED_COMMAND"
    assert refused("light", "thermostat", "set_target", celsius=22) == "UNSUPPORTED_COMMAND"
    assert refused("light", "on_off", "explode") == "UNSUPPORTED_COMMAND"


_SAMPLE_PARAMS = {
    "value": 50,
    "delta": 1,
    "muted": True,
    "celsius": None,
    "mode": None,
}


@pytest.mark.parametrize("kind", sorted(DEVICE_TYPES))
def test_every_command_a_type_takes_leaves_a_valid_state(kind):
    spec = DEVICE_TYPES[kind]
    for trait in spec.traits:
        for command, params in TRAIT_COMMANDS[trait].items():
            values = {name: _SAMPLE_PARAMS[name] for name in params}
            if "celsius" in values:
                values["celsius"] = spec.target_c[0]
            if "mode" in values:
                values["mode"] = spec.modes[-1]
            validate_state(kind, run(kind, None, trait, command, **values))


# -- the Provider ----------------------------------------------------------

AC = Device(device_id="living.ac", name="客厅空调", type="climate", area_id=None)
LAMP = Device(device_id="living.lamp", name="台灯", type="light")


@pytest.fixture
def provider(tmp_path):
    value = VirtualProvider(tmp_path / "state" / "virtual.sqlite3")
    value.initialize()
    return value


async def test_state_survives_a_new_provider_on_the_same_file(provider):
    await provider.reconcile(OWNER, [AC])
    after = await provider.execute(
        OWNER, AC, cmd("living.ac", "thermostat", "set_target", celsius=20)
    )
    assert after == initial_state("climate") | {"on": True, "target_c": 20}
    reopened = VirtualProvider(provider.path)
    reopened.initialize()
    assert await reopened.states(OWNER, [AC]) == {"living.ac": after}
    assert stat.S_IMODE(os.stat(provider.path).st_mode) == 0o600


async def test_reconcile_starts_new_devices_forgets_removed_ones_and_restarts_retyped_ones(
    provider,
):
    await provider.reconcile(OWNER, [AC, LAMP])
    await provider.execute(OWNER, AC, cmd("living.ac", "on_off", "on"))
    await provider.execute(OWNER, LAMP, cmd("living.lamp", "on_off", "on"))
    await provider.reconcile(OWNER, [AC])
    await provider.reconcile(OWNER, [AC, LAMP])
    assert (await provider.states(OWNER, [LAMP]))["living.lamp"] == initial_state("light")
    assert (await provider.states(OWNER, [AC]))["living.ac"]["on"] is True
    as_switch = LAMP.model_copy(update={"type": "switch"})
    await provider.execute(OWNER, LAMP, cmd("living.lamp", "on_off", "on"))
    await provider.reconcile(OWNER, [AC, as_switch])
    assert (await provider.states(OWNER, [as_switch]))["living.lamp"] == initial_state("switch")


async def test_owners_never_share_a_device(provider):
    await provider.reconcile(OWNER, [AC])
    await provider.reconcile(OTHER_OWNER, [AC])
    await provider.execute(OWNER, AC, cmd("living.ac", "on_off", "on"))
    assert (await provider.states(OTHER_OWNER, [AC]))["living.ac"]["on"] is False
    await provider.reconcile(OTHER_OWNER, [])
    assert (await provider.states(OWNER, [AC]))["living.ac"]["on"] is True


async def test_a_refused_command_changes_nothing(provider):
    await provider.reconcile(OWNER, [AC])
    with pytest.raises(SmartHomeError):
        await provider.execute(OWNER, AC, cmd("living.ac", "thermostat", "set_target", celsius=99))
    with pytest.raises(ValueError):
        await provider.execute(OWNER, AC, cmd("living.lamp", "on_off", "on"))
    assert (await provider.states(OWNER, [AC]))["living.ac"] == initial_state("climate")


def test_refuses_a_database_from_an_unknown_schema(tmp_path):
    path = tmp_path / "virtual.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 7")
    with pytest.raises(RuntimeError, match="schema version: 7"):
        VirtualProvider(path).initialize()
