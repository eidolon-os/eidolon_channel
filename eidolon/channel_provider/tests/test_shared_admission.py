from eidolon.channel_provider.shared_admission import SharedAdmission


def admission():
    return SharedAdmission({"device-a": "invite-a", "device-b": "invite-b"}, deadline_ms=1000)


def test_delivery_ack_never_means_joined():
    state = admission()
    for device, command in [("device-a", "invite-a"), ("device-b", "invite-b")]:
        state.acknowledge(device, command, "accepted")
    assert not state.ready(now_ms=10)
    state.observe_members({"device-a", "device-b"})
    assert state.ready(now_ms=10)


def test_presence_can_arrive_before_ack_and_disconnect_withdraws_readiness():
    state = admission()
    state.observe_members({"device-a", "device-b", "agent"})
    assert not state.ready(now_ms=10)
    state.acknowledge("device-a", "invite-a", "completed")
    state.acknowledge("device-b", "invite-b", "accepted")
    assert state.ready(now_ms=10)
    state.observe_members({"device-a", "agent"})
    assert not state.ready(now_ms=10)


def test_wrong_device_or_old_command_cannot_satisfy_admission():
    state = admission()
    state.observe_members({"device-a", "device-b"})
    state.acknowledge("device-a", "invite-b", "accepted")
    state.acknowledge("device-b", "old-invite", "completed")
    assert not state.ready(now_ms=10)


def test_failure_is_terminal_even_after_late_success():
    state = admission()
    state.acknowledge("device-a", "invite-a", "failed")
    for device, command in [("device-a", "invite-a"), ("device-b", "invite-b")]:
        state.acknowledge(device, command, "completed")
    state.observe_members({"device-a", "device-b"})
    assert not state.ready(now_ms=10)
    assert state.failed_devices == frozenset({"device-a"})


def test_deadline_and_close_never_grant_permission():
    state = admission()
    state.observe_members({"device-a", "device-b"})
    state.acknowledge("device-a", "invite-a", "accepted")
    state.acknowledge("device-b", "invite-b", "accepted")
    assert not state.ready(now_ms=1000)
    assert not state.ready(now_ms=999)  # Clock adjustment cannot revive expiry.
    state.close()
    assert not state.ready(now_ms=10)
