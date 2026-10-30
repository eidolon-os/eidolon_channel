"""IP team composition of the existing endpoint preparation capability."""
from .endpoint_preparation import EndpointPreparation


class TeamConversation(EndpointPreparation):
    def __init__(self, opened, adapter, handles):
        self.opened, self.selection = opened, opened.selection
        self.round_epoch, self.round_error = 0, ""
        super().__init__(adapter, handles, self._activate)

    def _round_state(self, status):
        if self.state not in {"preparing", "ready"} or status.epoch <= self.round_epoch:
            return
        self.round_epoch, self.round_error = status.epoch, status.error

    async def _activate(self, sessions):
        source = self.handles[0]
        self.remove_observers.append(self.adapter.observe_team_state(
            source, sessions[source["device"]], self.selection.session_id, self._round_state))
        await self.adapter.open_team_session(self.opened, self.handles, sessions)

    async def validate_endpoints(self):
        await super().validate_endpoints()
        await self.adapter.require_team_input(self.handles[0])

    def snapshot(self):
        return dict(session_id=self.selection.session_id, state=self.state, error=self.error,
                    scenario='ip_role_group', completion_basis='native_playout',
                    round_error=self.round_error, round_epoch=self.round_epoch)
