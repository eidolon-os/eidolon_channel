"""Interrupt experience modes.

Modes describe product-facing interruption behaviour, not historical code
versions. Keep them centralized so future modes can be added without
scattering string checks across the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eidolon.livekit.common.config.schema import SUPPORTED_INTERRUPT_MODES

BALANCED_INTERRUPT_MODE = SUPPORTED_INTERRUPT_MODES[0]
RESPONSIVE_INTERRUPT_MODE = SUPPORTED_INTERRUPT_MODES[1]


@dataclass(frozen=True)
class InterruptModeSpec:
    name: str
    first_signal_cancel: bool = False
    fast_lexical_intents: bool = False
    stabilize_normal_interrupts: bool = True
    weak_signal_followup_hold: bool = True
    attention_enforce: bool | None = None


_MODES: dict[str, InterruptModeSpec] = {
    BALANCED_INTERRUPT_MODE: InterruptModeSpec(
        name=BALANCED_INTERRUPT_MODE,
    ),
    RESPONSIVE_INTERRUPT_MODE: InterruptModeSpec(
        name=RESPONSIVE_INTERRUPT_MODE,
        first_signal_cancel=True,
        fast_lexical_intents=True,
        stabilize_normal_interrupts=False,
        weak_signal_followup_hold=False,
        attention_enforce=False,
    ),
}


def interrupt_mode_spec(mode_or_config: str | Any | None) -> InterruptModeSpec:
    """Resolve a mode name or config object into an immutable mode spec."""

    if isinstance(mode_or_config, InterruptModeSpec):
        return mode_or_config

    mode = mode_or_config
    if mode is not None and not isinstance(mode, str):
        mode = getattr(mode, "interrupt_mode", BALANCED_INTERRUPT_MODE)
    name = str(mode or BALANCED_INTERRUPT_MODE).strip() or BALANCED_INTERRUPT_MODE
    try:
        return _MODES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODES))
        raise ValueError(
            f"unknown turn_policy.interrupt_mode {name!r}; expected one of: {supported}"
        ) from exc


def effective_attention_enforce(config: Any) -> bool:
    spec = interrupt_mode_spec(config)
    if spec.attention_enforce is not None:
        return spec.attention_enforce
    return bool(config.attention.enforce)


def supported_interrupt_modes() -> tuple[str, ...]:
    return tuple(sorted(_MODES))
