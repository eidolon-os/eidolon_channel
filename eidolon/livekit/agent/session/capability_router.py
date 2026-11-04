"""Session-owned bindings for local services and remote devices."""

import logging
from dataclasses import dataclass, replace

from eidolon_sdk.biz.capabilities import decode_arguments, installed_registry, validate

from ..eidolon_agent_rpc.channel_tools import ChannelToolResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Binding:
    definition: object
    execute: object
    cancel: object
    location: str


class CapabilityRouter:
    def __init__(self, registry=None):
        self.registry = registry or installed_registry()
        self.bindings = {}

    def register(self, names, execute, cancel, *, location):
        if execute is None:
            raise ValueError("capability has no executor")
        for name in names:
            definition = self.registry.get(name)
            if definition.execution_location != location or name in self.bindings:
                raise ValueError("conflicting capability binding")
        for name in names:
            self.bindings[name] = Binding(self.registry.get(name), execute, cancel, location)

    @property
    def names(self):
        return frozenset(self.bindings)

    async def execute(self, call):
        binding = self.bindings.get(call.name)
        if binding is None:
            return ChannelToolResult(
                call.turn_id,
                call.call_id,
                call.name,
                False,
                {},
                error_code="UNAVAILABLE",
                outcome_state="rejected",
            )
        try:
            call = replace(
                call,
                arguments=decode_arguments(binding.definition.input_schema, dict(call.arguments)),
            )
            validate(binding.definition.input_schema, dict(call.arguments))
        except ValueError:
            return ChannelToolResult(
                call.turn_id,
                call.call_id,
                call.name,
                False,
                {},
                error_code="INVALID_ARGUMENT",
                outcome_state="rejected",
            )
        log.info(
            "capability dispatch turn=%s call=%s name=%s digest=%s location=%s",
            call.turn_id,
            call.call_id,
            call.name,
            binding.definition.contract_digest,
            binding.location,
        )
        result = await binding.execute(call)
        if (result.turn_id, result.call_id, result.name) != (call.turn_id, call.call_id, call.name):
            raise ValueError("mismatched capability receipt")
        if result.ok:
            validate(binding.definition.result_schema, result.content)
        log.info(
            "capability receipt turn=%s call=%s name=%s state=%s error=%s",
            call.turn_id,
            call.call_id,
            call.name,
            result.outcome_state,
            result.error_code,
        )
        return result

    async def cancel(self, call):
        binding = self.bindings.get(call.name)
        if binding is not None and binding.cancel is not None:
            await binding.cancel(call)
