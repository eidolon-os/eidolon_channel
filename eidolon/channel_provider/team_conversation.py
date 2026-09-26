"""IP team composition of the existing endpoint preparation capability."""
from .endpoint_preparation import EndpointPreparation


class TeamConversation(EndpointPreparation):
    def __init__(self, opened, adapter, handles):
        self.opened, self.selection = opened, opened.selection
        super().__init__(adapter, handles, self._activate)

    async def _activate(self, sessions):
        await self.adapter.open_team_session(self.opened, self.handles, sessions)

    def snapshot(self):
        return dict(session_id=self.selection.session_id, state=self.state, error=self.error,
                    scenario='ip_role_group', completion_basis='native_playout')
