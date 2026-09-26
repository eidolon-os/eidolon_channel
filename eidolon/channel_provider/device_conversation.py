"""One directed conversation composed from reusable endpoint preparation."""
from eidolon_sdk.biz.control.device_conversation import DeviceConversationSelection
from .endpoint_preparation import EndpointPreparation


class DeviceConversation(EndpointPreparation):
    def __init__(self, selection: DeviceConversationSelection, owner_id: str,
                 adapter: object, handles: tuple[dict, dict]):
        self.selection = selection
        self.owner_id = owner_id
        super().__init__(adapter, handles, self._activate)

    async def _activate(self, sessions: dict[str, str]) -> None:
        source, target = self.handles
        await self.adapter.open_directed_session(source, target,
            source_session_id=sessions[source["device"]],
            target_session_id=sessions[target["device"]],
            target_companion_id=self.selection.target_companion_id)

    def snapshot(self) -> dict:
        return {"session_id": self.selection.session_id, "state": self.state,
                "error": self.error}
