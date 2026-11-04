"""Single-device session; shares endpoint preparation, not directed routing."""
from .endpoint_preparation import EndpointPreparation

class DeviceSession(EndpointPreparation):
    def __init__(self, selection, owner_id, adapter, handles):
        self.selection, self.owner_id = selection, owner_id
        super().__init__(adapter, handles, self._activate)

    async def _activate(self, sessions):
        handle, = self.handles
        await self.adapter.open_device_session(handle,
            conversation_id=sessions[handle["device"]],
            target_companion_id=self.selection.target_companion_id)

    def snapshot(self):
        return dict(session_id=self.selection.session_id, state=self.state, error=self.error)
