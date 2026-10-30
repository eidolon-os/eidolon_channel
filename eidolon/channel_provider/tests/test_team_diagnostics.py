"""Operator runtime diagnostics cannot bypass saved Team admission."""
from types import SimpleNamespace
import pytest
from eidolon.channel_provider import team_test

@pytest.mark.parametrize('action,method,suffix', [
    ('start', 'POST', ''), ('status', 'GET', '/run'), ('close', 'POST', '/run/close')])
async def test_runtime_commands_use_saved_team_management(monkeypatch, action, method, suffix):
    calls = []
    class Response:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def json(self):
            return dict(session_id='run', state='closed', scenario='ip_role_group',
                        completion_basis='native_playout')
    class Http:
        def __init__(self, **kwargs):
            assert kwargs['headers'] == {'Authorization': 'Bearer controller-test'}
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def request(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Response()
    monkeypatch.setenv('EIDOLON_CONTROLLER_ACCESS_TOKEN', 'controller-test')
    monkeypatch.setattr(team_test.aiohttp, 'ClientSession', Http)
    monkeypatch.setattr(team_test, 'load_provider_config', lambda: pytest.fail('runtime must not load service credentials'))
    args = SimpleNamespace(action=action, url='http://localhost:8080/', team='saved', revision=3, session='run')
    assert (await team_test.run(args))['session_id'] == 'run'
    assert calls == [((method, 'http://localhost:8080/api/management/v1/teams/saved/runs' + suffix),
                     dict(json={'expected_revision': 3, 'session_id': 'run'} if action == 'start' else None,
                          allow_redirects=False))]


def test_start_requires_saved_identity_and_stable_session():
    parser = team_test.parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['start', '--input', 'input', '--output', 'output'])
    with pytest.raises(SystemExit):
        parser.parse_args(['start', '--url', 'http://localhost', '--team', 'saved', '--revision', '1'])
