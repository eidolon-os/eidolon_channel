import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from eidolon_sdk.biz.runtime import RuntimeTokenVerifier
from eidolon_sdk.biz.speaker_identity import SpeakerEvidenceSource, SpeakerIdentityObservation
from eidolon_sdk.biz.speaker_attestation import sign_observation, verify_observation
from eidolon.livekit.agent.runtime.automatic_memory import AutomaticMemorySession

SECRET = "test-secret-with-enough-entropy-32b"


def observation(person="a", *, confidence="validated", trace="t", status="identified"):
    now = time.time()
    return sign_observation(SpeakerIdentityObservation(
        owner_id="owner", session_id="transport", turn_id=trace,
        status=status, confidence=confidence, reason="voice_match",
        **(dict(person_id=person, person_revision=1, profile_id="profile", profile_revision=1)
           if status == "identified" else {}),
        model_id="model", weights_hash="a"*64, preprocessing_version="v1", scoring_version="v1",
        decision_version="v1", issued_at=now, expires_at=now+10,
        source=SpeakerEvidenceSource(request_id=trace, session_id="transport", input_id="mic",
            generation=1, sample_rate=16000, start_sample=0, end_sample=16000)), secret=SECRET, text="hello")


def coordinator():
    services = SimpleNamespace(personal_memory=None, resolve_room=AsyncMock(return_value=SimpleNamespace(
        owner_id="owner", companion_id="companion", device_id="device")),
        runtime=SimpleNamespace(resolve_person_memory=AsyncMock(return_value="private")))
    llm = SimpleNamespace(close_runtime=AsyncMock(), install_runtime=Mock())
    stop = AsyncMock()
    controller = AutomaticMemorySession(services=services, room=object(), source_session_id="transport",
        secret=SECRET, llm=llm, stop_outputs=stop)
    return controller, services, llm, stop


async def prepared(controller, **kwargs):
    """An observation whose background authority lookup already finished."""
    token = kwargs.get("observation_token")
    if token:
        try:
            value = verify_observation(token, secret=SECRET, owner_id="owner",
                session_id="transport", turn_id=kwargs.get("trace_id"), text=kwargs.get("text", ""))
        except ValueError:
            pass
        else:
            controller.observe(value)
            if controller._admission_task is not None:
                await controller._admission_task
    return await controller.prepare(**kwargs)


async def test_a_b_unknown_keep_room_history_without_default_memory():
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        c.consume(trace_id="t", text="hello")
        first = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert first.person_id == "a" and first.source_session_id == "transport"
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        assert stop.await_count == 0
        await prepared(c, observation_token=observation("b"), trace_id="t", text="hello")
        second = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert second.person_id == "b" and second.session_id == first.session_id
        await prepared(c, text="typed input")
        unknown = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert unknown.person_id is None and unknown.session_id == second.session_id
        assert unknown.source_session_id == "transport"
        assert stop.await_count == 0
    finally:
        await c.aclose()


@pytest.mark.parametrize("mode", ["forged", "wrong-turn", "unbound"])
async def test_ineligible_evidence_never_grants_memory(mode):
    c, services, llm, stop = coordinator()
    token = observation()
    if mode == "forged": token = "invalid"
    if mode == "unbound": services.runtime.resolve_person_memory.side_effect = LookupError("disabled")
    try:
        await prepared(c, observation_token=token, trace_id="other" if mode == "wrong-turn" else "t", text="hello")
        c.consume(trace_id="other" if mode == "wrong-turn" else "t", text="hello")
        assert c._person is None
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
    finally:
        await c.aclose()


async def test_failed_credential_transition_blocks_turn_until_retry():
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        llm.close_runtime.side_effect = TimeoutError("RPC close failed")
        with pytest.raises(TimeoutError):
            await prepared(c, observation_token=observation("b"), trace_id="t", text="hello")
        with pytest.raises(RuntimeError, match="admission"):
            c.consume(trace_id="t", text="hello")
        assert llm.install_runtime.call_count == 1
        llm.close_runtime.side_effect = None
        await prepared(c, observation_token=observation("b"), trace_id="t", text="hello")
        assert c._person == ("b", 1)
    finally:
        await c.aclose()


async def test_credential_refresh_preserves_identity_session_and_conversation():
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        original = llm.install_runtime.call_args.args
        c._renew_at = 0
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        refreshed = llm.install_runtime.call_args.args
        assert original[1] == refreshed[1]
        assert original[0] != refreshed[0]
        assert c._person == ("a", 1)
    finally:
        await c.aclose()


async def test_idle_person_revocation_stops_outputs_without_waiting_for_next_turn():
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        cleared = asyncio.Event()
        async def clear(**kwargs):
            cleared.set()
        stop.side_effect = clear
        services.runtime.resolve_person_memory.side_effect = LookupError("person inactive")
        await asyncio.wait_for(cleared.wait(), timeout=2.5)
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
        with pytest.raises(RuntimeError, match="admission"):
            c.consume(trace_id="t", text="hello")
    finally:
        await c.aclose()


@pytest.mark.parametrize("confidence", ["controlled_trial", "validated"])
async def test_recognition_policy_is_not_a_second_memory_gate(confidence):
    c, services, llm, stop = coordinator()
    try:
        changed = await prepared(c, observation_token=observation(confidence=confidence), trace_id="t", text="hello")
        assert not changed
        c.consume(trace_id="t", text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id == "a"
        services.runtime.resolve_person_memory.assert_awaited_once_with(owner_id="owner", person_id="a", person_revision=1)
        stop.assert_not_awaited()
    finally:
        await c.aclose()


@pytest.mark.parametrize("status", ["unknown", "uncertain", "unavailable"])
async def test_nonidentified_turn_drops_previous_memory_without_dropping_dialogue(status):
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        original_conversation = llm.install_runtime.call_args.args[1]
        await prepared(c, observation_token=observation(status=status), trace_id="t", text="hello")
        c.consume(trace_id="t", text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
        assert llm.install_runtime.call_args.args[1] == original_conversation
        stop.assert_not_awaited()
    finally:
        await c.aclose()


@pytest.mark.parametrize("failure", [TimeoutError("Data unavailable"), LookupError("inactive"), None])
async def test_failed_authority_never_keeps_previous_person(failure):
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        services.runtime.resolve_person_memory.side_effect = failure
        services.runtime.resolve_person_memory.return_value = None
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        c.consume(trace_id="t", text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
        stop.assert_not_awaited()
    finally:
        await c.aclose()


async def test_pending_authority_does_not_delay_turn_or_late_grant():
    c, services, llm, stop = coordinator()
    gate, started = asyncio.Event(), asyncio.Event()
    async def resolve(**kwargs):
        started.set()
        await gate.wait()
        return "private"
    services.runtime.resolve_person_memory.side_effect = resolve
    token = observation()
    value = verify_observation(token, secret=SECRET, owner_id="owner",
        session_id="transport", turn_id="t", text="hello")
    try:
        c.observe(value)
        await started.wait()
        await asyncio.wait_for(c.prepare(observation_token=token, trace_id="t", text="hello"), .2)
        c.consume(trace_id="t", text="hello")
        assert c._person is None
        gate.set()
        await c._admission_task
        assert c._person is None
        await c.prepare(observation_token=observation(trace="next"), trace_id="next", text="hello")
        assert c._person is None
    finally:
        await c.aclose()


async def test_close_cancels_background_admission():
    c, services, llm, stop = coordinator()
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def resolve(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    services.runtime.resolve_person_memory.side_effect = resolve
    c.observe(verify_observation(observation(), secret=SECRET, owner_id="owner",
        session_id="transport", turn_id="t", text="hello"))
    await started.wait()
    await c.aclose()
    assert cancelled.is_set()


async def test_slow_revocation_check_cannot_block_or_revoke_new_turn():
    c, services, llm, stop = coordinator()
    entered, release = asyncio.Event(), asyncio.Event()
    async def slow(**kwargs):
        entered.set()
        await release.wait()
        raise LookupError("old Person revoked")
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        c._watcher.cancel()
        await asyncio.gather(c._watcher, return_exceptions=True)
        c._watcher = None
        services.runtime.resolve_person_memory.side_effect = slow
        check = asyncio.create_task(c._check_authority())
        await entered.wait()
        # The old authority request must not serialize a new unknown speaker.
        await asyncio.wait_for(c.prepare(trace_id="new", text="new person"), .2)
        c.consume(trace_id="new", text="new person")
        release.set()
        await check
        assert c._person is None and not c._blocked
        stop.assert_not_awaited()
    finally:
        release.set()
        await c.aclose()
