import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from eidolon_sdk.biz.runtime import RuntimeTokenVerifier
from eidolon_sdk.biz.speaker_identity import (
    IdentityDecisions,
    PersonEvidence,
    SpeakerEvidenceSource,
    SpeakerIdentityObservation,
)
from eidolon_sdk.biz.speaker_attestation import sign_observation, verify_observation
from eidolon.livekit.agent.runtime.automatic_memory import AutomaticMemorySession

SECRET = "test-secret-with-enough-entropy-32b"
MEMORY_USES = ("personal_memory", "addressing", "speaker_label")
UNRESOLVED = {"uncertain": "speech_too_short", "unavailable": "dependency_unavailable"}


def observation(person="a", *, trace="t", status="compared", maturity="established",
                uses=MEMORY_USES, discovery=False):
    """A signed v2 observation; ``uses`` are the decisions naming ``person``."""
    now = time.time()
    compared = {}
    if status == "compared":
        posterior = 0.9999 if "personal_memory" in uses else (0.95 if uses else 0.0001)
        compared = dict(
            model_id="model", weights_hash="a" * 64, preprocessing_version="v1",
            backend_version="b1", policy_version="p1",
            evidence=[PersonEvidence(person_id=person, person_revision=1, maturity=maturity,
                is_named=maturity == "established", profile_id="profile", profile_revision=1,
                llr=10.0 if uses else -10.0, posterior=posterior)],
            posterior_unknown=1 - posterior,
            decisions=IdentityDecisions(**{use: person for use in uses}, discovery=discovery))
    return sign_observation(SpeakerIdentityObservation(
        owner_id="owner", session_id="transport", turn_id=trace,
        status=status, reason="compared" if compared else UNRESOLVED[status], **compared,
        issued_at=now, expires_at=now+10,
        source=SpeakerEvidenceSource(request_id=trace, session_id="transport", input_id="mic",
            generation=1, sample_rate=16000, start_sample=0, end_sample=16000, channel="mobile")),
        secret=SECRET, text="hello")


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
    if mode == "forged":
        token = "invalid"
    if mode == "unbound":
        services.runtime.resolve_person_memory.side_effect = LookupError("disabled")
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


@pytest.mark.parametrize("uses", [MEMORY_USES, ("personal_memory",)])
async def test_only_the_personal_memory_decision_gates_memory(uses):
    c, services, llm, stop = coordinator()
    try:
        changed = await prepared(c, observation_token=observation(uses=uses), trace_id="t", text="hello")
        assert not changed
        c.consume(trace_id="t", text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id == "a"
        services.runtime.resolve_person_memory.assert_awaited_once_with(owner_id="owner", person_id="a", person_revision=1)
        stop.assert_not_awaited()
    finally:
        await c.aclose()


@pytest.mark.parametrize("outcome", [
    dict(uses=(), discovery=True),             # someone not enrolled
    dict(uses=("addressing", "speaker_label")),  # addressed, below the memory threshold
    dict(uses=()),                             # compared, no use met
    dict(status="uncertain"),
    dict(status="unavailable"),
])
async def test_turn_without_memory_decision_drops_previous_memory_without_dropping_dialogue(outcome):
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        original_conversation = llm.install_runtime.call_args.args[1]
        services.runtime.resolve_person_memory.reset_mock()
        await prepared(c, observation_token=observation(**outcome), trace_id="t", text="hello")
        # Another use's decision never even prepares a memory lookup.
        services.runtime.resolve_person_memory.assert_not_awaited()
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


async def test_candidate_signs_scoped_v8_without_enabling_broad_memory_and_revokes():
    c, services, llm, stop = coordinator()
    services.runtime.resolve_provisional_memory = AsyncMock(return_value="candidate-realm")
    c.publish = AsyncMock()
    try:
        await prepared(c, observation_token=observation(maturity="candidate"), trace_id="t", text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET, allow_provisional=True).verify(
            llm.install_runtime.call_args.args[0])
        assert identity.memory_mode == "provisional" and identity.person_id == "a"
        assert identity.session_id == identity.source_session_id == "transport"
        services.runtime.resolve_person_memory.assert_not_awaited()
        services.runtime.resolve_provisional_memory.assert_awaited_once_with(
            owner_id="owner", person_id="a", person_revision=1, session_id="transport")
        await asyncio.gather(*tuple(c._notices))
        assert c.publish.call_args.args[0]["state"] == "disabled"
        services.runtime.resolve_provisional_memory.side_effect = ValueError("policy revoked")
        await c._check_authority()
        assert c._person is None
        stop.assert_awaited_once()
        cleared = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert cleared.person_id is None
    finally:
        await c.aclose()


@pytest.mark.parametrize("denied", [True, False])
async def test_candidate_pending_or_denied_policy_never_falls_back(denied):
    c, services, llm, _ = coordinator()
    release = asyncio.Event()
    async def resolve(**kwargs):
        if denied:
            raise ValueError("not permitted")
        await release.wait()
        return "candidate"
    services.runtime.resolve_provisional_memory = AsyncMock(side_effect=resolve)
    token = observation(maturity="candidate")
    try:
        c.observe(verify_observation(token, secret=SECRET, owner_id="owner", session_id="transport",
                                    turn_id="t", text="hello"))
        await asyncio.sleep(0)
        await c.prepare(observation_token=token, trace_id="t", text="hello")
        c.consume(trace_id="t", text="hello")
        assert c._person is None
        release.set()
        await c._admission_task
        assert c._person is None  # late completion cannot rewrite the consumed turn
        services.runtime.resolve_person_memory.assert_not_awaited()
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
    finally:
        await c.aclose()


async def test_a_turn_whose_recognition_is_still_running_keeps_the_binding_it_cannot_use():
    c, services, llm, stop = coordinator()
    try:
        await prepared(c, observation_token=observation(), trace_id="t", text="hello")
        installs = llm.install_runtime.call_count
        now = time.time()
        pending = sign_observation(SpeakerIdentityObservation(
            owner_id="owner", session_id="transport", turn_id="t2", status="unavailable",
            reason="recognition_pending", issued_at=now, expires_at=now + 10), secret=SECRET, text="hello")
        await prepared(c, observation_token=pending, trace_id="t2", text="hello")
        # No rebinding: the Agent opens memory only if the late result names this Person.
        assert llm.install_runtime.call_count == installs
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id == "a"
        await prepared(c, observation_token=observation(status="uncertain", trace="t3"), trace_id="t3",
                       text="hello")
        identity = await RuntimeTokenVerifier(secret=SECRET).verify(llm.install_runtime.call_args.args[0])
        assert identity.person_id is None
    finally:
        await c.aclose()
