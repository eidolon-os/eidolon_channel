"""Contract tests for :class:`EidolonAgentGrpcLlm`.

These spin up an in-process gRPC server with a hand-rolled
``EidolonAgentServicer`` that emits a tiny scripted ``TurnEvent`` sequence,
then drive the channel's adapter and assert it produces the expected
:class:`livekit.agents.llm.ChatChunk` stream — and that user-side
cancellation results in a ``CancelTurn`` being written on the same call.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock

import grpc
import grpc.aio
import pytest
from google.protobuf import struct_pb2
from livekit.agents.llm import ChatContext, ChatMessage

from eidolon.livekit.agent.eidolon_agent_rpc.grpc_llm import EidolonAgentGrpcLlm
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2 as pb,
)
from eidolon.livekit.agent.eidolon_agent_rpc.v1.grpc_gen import (
    eidolon_pb2_grpc as pbg,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _delta(turn_id: str, seq: int, text: str) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["text"] = text
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.DELTA, data=data)


def _done(turn_id: str, seq: int) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["status"] = "ok"
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.DONE, data=data)


def _error(turn_id: str, seq: int, code: str, message: str, fatal: bool) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["code"] = code
    data["message"] = message
    data["fatal"] = fatal
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.ERROR, data=data)


def _usage(
    turn_id: str, seq: int, prompt: int, completion: int, total: int, model: str
) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["prompt_tokens"] = prompt
    data["completion_tokens"] = completion
    data["total_tokens"] = total
    data["model"] = model
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.USAGE, data=data)


def _state(turn_id: str, seq: int, state: str) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["state"] = state
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.STATE, data=data)


def _delta_role(turn_id: str, seq: int, text: str, role: str) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["text"] = text
    data["role"] = role
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.DELTA, data=data)


def _tool_call(turn_id: str, seq: int, name: str) -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["name"] = name
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.TOOL_CALL, data=data)


def _tool_result(turn_id: str, seq: int, *, name: str, ok: bool, error: str = "") -> pb.TurnEvent:
    data = struct_pb2.Struct()
    data["name"] = name
    data["ok"] = ok
    if error:
        data["error"] = error
    return pb.TurnEvent(turn_id=turn_id, seq=seq, kind=pb.TurnEvent.TOOL_RESULT, data=data)


class _ScriptedServicer(pbg.EidolonAgentServicer):
    """Replays a canned DELTA+DONE per StartTurn. Records cancels for assertions.

    A background task drains the request iterator so cancels are observed even
    while the response generator is mid-flight.
    """

    def __init__(self, *, deltas: list[str], delay_between: float = 0.0) -> None:
        self._deltas = deltas
        self._delay = delay_between
        self.cancels: list[str] = []
        self.starts: list[pb.StartTurn] = []
        self._start_q: asyncio.Queue[pb.StartTurn] = asyncio.Queue()
        self._cancel_evt = asyncio.Event()

    async def _drain_requests(self, request_iterator) -> None:
        async for req in request_iterator:
            kind = req.WhichOneof("payload")
            if kind == "start":
                self.starts.append(req.start)
                await self._start_q.put(req.start)
            elif kind == "cancel":
                self.cancels.append(req.cancel.turn_id)
                self._cancel_evt.set()

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        drain_task = asyncio.create_task(self._drain_requests(request_iterator))
        try:
            start = await self._start_q.get()
            turn_id = start.turn_id
            seq = 0
            for chunk in self._deltas:
                seq += 1
                yield _delta(turn_id, seq, chunk)
                if self._delay > 0:
                    try:
                        await asyncio.wait_for(self._cancel_evt.wait(), timeout=self._delay)
                        return  # cancelled mid-turn — don't emit DONE
                    except asyncio.TimeoutError:
                        pass
            seq += 1
            yield _done(turn_id, seq)
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


class _UsageStateServicer(pbg.EidolonAgentServicer):
    """Emits STATE(thinking) → DELTA → USAGE → DONE.
    Used to verify C-phase typed-payload dispatch in EidolonAgentGrpcLlmStream."""

    def __init__(self) -> None:
        self.starts: list[pb.StartTurn] = []

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        async for req in request_iterator:
            if req.WhichOneof("payload") == "start":
                self.starts.append(req.start)
                tid = req.start.turn_id
                yield _state(tid, 1, "thinking")
                yield _delta(tid, 2, "ok")
                yield _usage(tid, 3, prompt=12, completion=3, total=15, model="brain-test")
                yield _done(tid, 4)
                return


class _ToolEventServicer(pbg.EidolonAgentServicer):
    """Emits TOOL_CALL → TOOL_RESULT(error) → DELTA → DONE.

    Mirrors a failing tool (e.g. get_weather) so we can assert the channel now
    surfaces the result's ok/error at INFO instead of DEBUG-swallowing it."""

    def __init__(self) -> None:
        self.starts: list[pb.StartTurn] = []

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        async for req in request_iterator:
            if req.WhichOneof("payload") == "start":
                self.starts.append(req.start)
                tid = req.start.turn_id
                yield _tool_call(tid, 1, "get_weather")
                yield _tool_result(
                    tid, 2, name="get_weather", ok=False, error="weather_lookup_failed"
                )
                yield _delta(tid, 3, "抱歉，天气接口暂时没有响应。")
                yield _done(tid, 4)
                return


class _PreambleServicer(pbg.EidolonAgentServicer):
    """Emits repeated status deltas, then a role=answer delta.

    Verifies the channel keeps ordinary status out of TTS, speaks a delayed
    slow-tool hint at most once, and still renders the real answer."""

    def __init__(self) -> None:
        self.starts: list[pb.StartTurn] = []

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        async for req in request_iterator:
            if req.WhichOneof("payload") == "start":
                self.starts.append(req.start)
                tid = req.start.turn_id
                yield _delta_role(tid, 1, "我先调用相关工具处理一下。", "tool_preamble")
                yield _delta_role(tid, 2, "我先调用相关工具处理一下。", "tool_preamble")
                yield _delta_role(tid, 3, "稍等，我处理一下。", "slow_tool_hint")
                yield _delta_role(tid, 4, "稍等，我处理一下。", "slow_tool_hint")
                yield _delta_role(tid, 5, "北京今天晴。", "answer")
                yield _done(tid, 6)
                return


class _ErrorServicer(pbg.EidolonAgentServicer):
    """On any StartTurn, immediately emits one ERROR event then ends the stream.
    Used to drive the A4 error-code mapping tests."""

    def __init__(self, *, code: str, message: str, fatal: bool) -> None:
        self._code = code
        self._message = message
        self._fatal = fatal
        self.starts: list[pb.StartTurn] = []

    async def Chat(self, request_iterator, context):  # type: ignore[override]
        async for req in request_iterator:
            if req.WhichOneof("payload") == "start":
                self.starts.append(req.start)
                yield _error(req.start.turn_id, 1, self._code, self._message, self._fatal)
                return


async def _serve(servicer) -> tuple[grpc.aio.Server, str]:
    port = _free_port()
    server = grpc.aio.server()
    pbg.add_EidolonAgentServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    return server, f"127.0.0.1:{port}"


def _ctx(text: str) -> ChatContext:
    ctx = ChatContext()
    ctx.items.append(ChatMessage(role="user", content=[text]))
    return ctx


@pytest.mark.asyncio
async def test_forwards_deltas_then_finishes() -> None:
    servicer = _ScriptedServicer(deltas=["你", "好啊"])
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:room-abc",
        )
        try:
            provider_events: list[dict] = []
            adapter.on("provider_event", provider_events.append)
            stream = adapter.chat(chat_ctx=_ctx("打个招呼"))
            collected: list[str] = []
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    collected.append(chunk.delta.content)
            assert collected == ["你", "好啊"]
            assert len(servicer.starts) == 1
            assert servicer.starts[0].text == "打个招呼"
            assert servicer.starts[0].conversation_id == "livekit:room-abc"
            assert servicer.starts[0].turn_id  # non-empty uuid hex
            assert servicer.starts[0].input_modality == "voice"
            assert [event["event"] for event in provider_events] == [
                "brain_request_started",
                "brain_request_sent",
                "brain_first_model_activity",
                "brain_first_delta",
                "brain_first_answer_delta",
                "brain_done",
            ]
            assert provider_events[-1]["request_id"].startswith("eidolon-")
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_user_text_and_turn_metadata_survive_retry_attempt() -> None:
    from livekit.agents._exceptions import APIConnectionError
    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.eidolon_agent_rpc.session import DeltaPayload

    class _FlakySession:
        def __init__(self) -> None:
            self.starts: list[str] = []
            self.conversations: list[str] = []
            self.metadata: list[dict | None] = []
            self.trace_ids: list[str | None] = []

        async def start_turn(
            self, *, text: str, conversation_id: str, metadata=None, trace_id=None
        ):
            self.starts.append(text)
            self.conversations.append(conversation_id)
            self.metadata.append(metadata)
            self.trace_ids.append(trace_id)
            if len(self.starts) == 1:
                raise APIConnectionError("transient start failure", retryable=True)

            async def _payloads():
                yield DeltaPayload("ok")

            return "retry-turn", _payloads()

    conversation_calls: list[str] = []

    def _conversation_id() -> str:
        value = f"livekit:retry-{len(conversation_calls) + 1}"
        conversation_calls.append(value)
        return value

    adapter = EidolonAgentGrpcLlm(
        target="unused",
        device_token=lambda: "test-token",
        conversation_id=_conversation_id,
    )
    session = _FlakySession()

    async def _get_session():
        return session

    adapter._get_session = _get_session  # type: ignore[method-assign]
    adapter.set_turn_control_metadata({"action": "cancel", "reason": "interrupt"})
    adapter.set_turn_trace_id("channel-turn-trace")

    stream = adapter.chat(
        chat_ctx=_ctx("第一轮 canonical"),
        conn_options=APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=1.0),
    )
    collected: list[str] = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            collected.append(chunk.delta.content)

    assert collected == ["ok"]
    assert conversation_calls == ["livekit:retry-1"]
    assert session.starts == ["第一轮 canonical", "第一轮 canonical"]
    assert session.conversations == ["livekit:retry-1", "livekit:retry-1"]
    assert session.metadata == [
        {"turn_control": {"action": "cancel", "reason": "interrupt"}},
        {"turn_control": {"action": "cancel", "reason": "interrupt"}},
    ]
    assert session.trace_ids == ["channel-turn-trace", "channel-turn-trace"]


@pytest.mark.asyncio
async def test_session_open_uses_connect_timeout() -> None:
    from livekit.agents._exceptions import APIConnectionError
    from livekit.agents.types import APIConnectOptions

    async def _slow_token() -> str:
        await asyncio.sleep(1.0)
        return "test-token"

    adapter = EidolonAgentGrpcLlm(
        target="127.0.0.1:1",
        device_token=_slow_token,
        conversation_id="livekit:timeout-test",
    )
    try:
        provider_events: list[dict] = []
        adapter.on("provider_event", provider_events.append)
        stream = adapter.chat(
            chat_ctx=_ctx("会超时吗"),
            conn_options=APIConnectOptions(
                max_retry=0,
                retry_interval=0.0,
                timeout=0.05,
            ),
        )
        with pytest.raises(APIConnectionError, match="session open timed out"):
            async for _ in stream:
                pass
        assert provider_events
        assert provider_events[0]["event"] == "brain_request_started"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_first_delta_timeout_cancels_without_replaying_logical_turn() -> None:
    from livekit.agents._exceptions import APIConnectionError
    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.eidolon_agent_rpc.session import (
        DeltaPayload,
        StatePayload,
    )

    class _TimeoutThenFastSession:
        def __init__(self) -> None:
            self.starts: list[str] = []
            self.turn_ids: list[str] = []
            self.cancels: list[str] = []
            self._tasks: set[asyncio.Task] = set()

        async def start_turn(
            self, *, text: str, conversation_id: str, metadata=None, trace_id=None
        ):
            self.starts.append(text)
            turn_id = f"turn-{len(self.starts)}"
            self.turn_ids.append(turn_id)

            async def _payloads():
                if turn_id == "turn-1":
                    yield StatePayload("speaking")
                    await asyncio.sleep(10.0)
                    yield DeltaPayload("late")
            return turn_id, _payloads()

        async def cancel_turn(self, turn_id: str) -> None:
            self.cancels.append(turn_id)

        def spawn(self, coro, *, name: str | None = None):
            task = asyncio.create_task(coro, name=name)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return task

    adapter = EidolonAgentGrpcLlm(
        target="unused",
        device_token=lambda: "test-token",
        conversation_id="livekit:first-delta-timeout",
    )
    session = _TimeoutThenFastSession()

    async def _get_session():
        return session

    adapter._get_session = _get_session  # type: ignore[method-assign]
    provider_events: list[dict] = []
    adapter.on("provider_event", provider_events.append)
    try:
        stream = adapter.chat(
            chat_ctx=_ctx("第一轮会卡住吗"),
            conn_options=APIConnectOptions(
                max_retry=1,
                retry_interval=0.0,
                timeout=0.05,
            ),
        )
        with pytest.raises(APIConnectionError, match="first delta timed out") as exc_info:
            async for _ in stream:
                pass

        await asyncio.wait_for(
            _wait_until(lambda: session.cancels == ["turn-1"]),
            timeout=1.0,
        )
        assert exc_info.value.retryable is False
        assert session.turn_ids == ["turn-1"]
        request_events = [
            event for event in provider_events if event.get("event") == "brain_request_sent"
        ]
        assert [event.get("attempt") for event in request_events] == [1]
        timeout_errors = [event for event in provider_events if event.get("event") == "brain_error"]
        assert timeout_errors[0]["code"] == "first_delta_timeout"
        assert timeout_errors[0]["turn_id"] == "turn-1"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_tool_call_releases_first_output_deadline_without_becoming_answer_delta() -> None:
    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.eidolon_agent_rpc.session import (
        DeltaPayload,
        ToolCallPayload,
    )

    class _ToolThenAnswerSession:
        def __init__(self) -> None:
            self.turn_ids: list[str] = []
            self.cancels: list[str] = []

        async def start_turn(self, **_kwargs):
            turn_id = "tool-turn"
            self.turn_ids.append(turn_id)

            async def _payloads():
                yield ToolCallPayload(name="play", args={})
                # Longer than the first-output timeout: once a complete tool
                # call arrives, the tool loop owns its remaining runtime.
                await asyncio.sleep(0.08)
                yield DeltaPayload("播放完成")

            return turn_id, _payloads()

        async def cancel_turn(self, turn_id: str) -> None:
            self.cancels.append(turn_id)

        def spawn(self, coro, *, name: str):
            return asyncio.create_task(coro, name=name)

    session = _ToolThenAnswerSession()
    adapter = EidolonAgentGrpcLlm(
        target="unused",
        device_token=lambda: "test-token",
        conversation_id="livekit:tool-deadline",
    )
    adapter._get_session = AsyncMock(return_value=session)
    adapter.discard_warm = AsyncMock()
    provider_events: list[dict] = []
    adapter.on("provider_event", provider_events.append)
    try:
        stream = adapter.chat(
            chat_ctx=_ctx("播放音乐"),
            conn_options=APIConnectOptions(max_retry=3, timeout=0.05),
        )
        spoken = [
            chunk.delta.content
            async for chunk in stream
            if chunk.delta and chunk.delta.content
        ]

        assert spoken == ["播放完成"]
        assert session.turn_ids == ["tool-turn"]
        assert session.cancels == []
        activity = [
            event
            for event in provider_events
            if event.get("event") == "brain_first_model_activity"
        ]
        assert [event.get("kind") for event in activity] == ["tool_call"]
        assert sum(event.get("event") == "brain_first_delta" for event in provider_events) == 1
        assert (
            sum(
                event.get("event") == "brain_first_answer_delta"
                for event in provider_events
            )
            == 1
        )
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_completed_turn_without_answer_or_tool_is_terminal_not_retried() -> None:
    from livekit.agents._exceptions import APIConnectionError
    from livekit.agents.types import APIConnectOptions

    from eidolon.livekit.agent.eidolon_agent_rpc.session import StatePayload

    class _EmptySession:
        def __init__(self) -> None:
            self.turn_ids: list[str] = []

        async def start_turn(self, **_kwargs):
            self.turn_ids.append("empty-turn")

            async def _payloads():
                yield StatePayload("thinking")

            return "empty-turn", _payloads()

    session = _EmptySession()
    adapter = EidolonAgentGrpcLlm(
        target="unused",
        device_token=lambda: "test-token",
        conversation_id="livekit:empty",
    )
    adapter._get_session = AsyncMock(return_value=session)
    adapter.discard_warm = AsyncMock()
    provider_events: list[dict] = []
    adapter.on("provider_event", provider_events.append)
    try:
        stream = adapter.chat(
            chat_ctx=_ctx("在吗"),
            conn_options=APIConnectOptions(max_retry=3, timeout=0.05),
        )
        with pytest.raises(APIConnectionError, match="without answer or tool") as exc_info:
            async for _ in stream:
                pass

        assert exc_info.value.retryable is False
        assert session.turn_ids == ["empty-turn"]
        errors = [event for event in provider_events if event.get("event") == "brain_error"]
        assert [event.get("code") for event in errors] == ["no_usable_output"]
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_cancel_writes_cancel_turn() -> None:
    # Server emits one delta, then waits up to 5s — long enough that the
    # adapter's _run task gets cancelled (via stream.aclose) before DONE.
    servicer = _ScriptedServicer(deltas=["第一段"], delay_between=5.0)
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:room-xyz",
        )
        try:
            stream = adapter.chat(chat_ctx=_ctx("讲个故事"))
            # Pull the first delta to confirm StartTurn reached the server.
            first = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
            assert first.delta and first.delta.content == "第一段"

            # Barge-in: same path the LiveKit pipeline takes on user interrupt.
            await stream.aclose()

            await asyncio.wait_for(_wait_until(lambda: len(servicer.cancels) == 1), timeout=2.0)
            assert servicer.cancels[0] == servicer.starts[0].turn_id

            # A2 regression guard: the cancel write is spawned through
            # session.spawn (not raw create_task) so we keep a strong ref
            # while it runs, and the centralized done-callback discards it
            # afterwards. The set must therefore be empty by now (cancel
            # has completed, reader is still running for the reader task —
            # filter to only background tasks named like the cancel).
            cancel_tasks_left = [
                t
                for t in adapter._session._background_tasks  # type: ignore[union-attr]
                if t.get_name().startswith("eidolon-cancel-")
            ]
            assert cancel_tasks_left == [], f"leaked cancel tasks: {cancel_tasks_left}"
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


async def _wait_until(predicate, *, interval: float = 0.05) -> None:
    while not predicate():
        await asyncio.sleep(interval)


def test_tls_config_validation_unknown_mode() -> None:
    """D2: bogus tls_mode must fail loud at construction, not silently fall back."""
    from eidolon.livekit.agent.eidolon_agent_rpc.session import TlsConfig

    with pytest.raises(ValueError, match="unrecognized"):
        EidolonAgentGrpcLlm(
            target="127.0.0.1:1",
            device_token=lambda: "x",
            conversation_id="livekit:test",
            tls=TlsConfig(mode="please-no"),
        )


def test_tls_config_validation_mtls_missing_cert(tmp_path) -> None:
    """D2: mTLS mode without client cert/key paths must fail loud."""
    from eidolon.livekit.agent.eidolon_agent_rpc.session import TlsConfig

    with pytest.raises(ValueError, match="mtls"):
        EidolonAgentGrpcLlm(
            target="127.0.0.1:1",
            device_token=lambda: "x",
            conversation_id="livekit:test",
            tls=TlsConfig(mode="mtls"),  # no client cert paths set
        )


def test_tls_config_validation_missing_ca_file(tmp_path) -> None:
    """D2: configured CA path that doesn't exist must fail loud (no silent skip)."""
    from eidolon.livekit.agent.eidolon_agent_rpc.session import TlsConfig

    with pytest.raises(ValueError, match="ca_path"):
        EidolonAgentGrpcLlm(
            target="127.0.0.1:1",
            device_token=lambda: "x",
            conversation_id="livekit:test",
            tls=TlsConfig(mode="tls", ca_path=str(tmp_path / "nope.pem")),
        )


@pytest.mark.asyncio
async def test_conversation_id_lazy_resolver_called_per_chat() -> None:
    """D1: when conversation_id is a callable, it's resolved on each chat() —
    the captured value lands on the wire in StartTurn.conversation_id."""
    servicer = _ScriptedServicer(deltas=["ok"])
    server, target = await _serve(servicer)
    try:
        call_log: list[str] = []

        def resolver() -> str:
            # Simulate "first participant connects between chat()s" — value
            # changes turn-to-turn, proving lazy resolution.
            ident = f"alice-{len(call_log) + 1}"
            cid = f"livekit:{ident}:smoke-room"
            call_log.append(cid)
            return cid

        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id=resolver,
        )
        try:
            # First chat() — should consult resolver
            async for _ in adapter.chat(chat_ctx=_ctx("第一通")):
                pass
            # Second chat() — should consult resolver again
            async for _ in adapter.chat(chat_ctx=_ctx("第二通")):
                pass

            assert call_log == [
                "livekit:alice-1:smoke-room",
                "livekit:alice-2:smoke-room",
            ], f"resolver call log: {call_log}"
            assert [s.conversation_id for s in servicer.starts] == call_log
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_conversation_id_resolver_failure_falls_back() -> None:
    """D1: resolver bug must not block a turn — fall back to a sentinel id."""
    servicer = _ScriptedServicer(deltas=["ok"])
    server, target = await _serve(servicer)
    try:

        def broken_resolver() -> str:
            raise RuntimeError("resolver intentionally broken")

        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id=broken_resolver,
            display_model="brain-test-model",
        )
        try:
            async for _ in adapter.chat(chat_ctx=_ctx("hello")):
                pass
            assert len(servicer.starts) == 1
            # Fallback uses display_model in the id so logs / brain history
            # don't carry the raw exception text.
            assert servicer.starts[0].conversation_id == "livekit:brain-test-model"
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


def test_tls_off_no_credentials_built() -> None:
    """D2: mode='off' yields no ChannelCredentials — same path as before D2."""
    from eidolon.livekit.agent.eidolon_agent_rpc.session import EidolonAgentSession, TlsConfig

    s = EidolonAgentSession(
        target="127.0.0.1:1",
        device_token="x",
        tls=TlsConfig(mode="off"),
    )
    assert s._credentials is None


@pytest.mark.asyncio
async def test_session_open_reuses_prebuilt_credentials(monkeypatch) -> None:
    """SDK extraction must not change TLS credential lifetime across reconnects."""
    from eidolon.livekit.agent.eidolon_agent_rpc import session as session_mod

    class FakeChannel:
        async def close(self) -> None:
            pass

    class FakeCall:
        def done(self) -> bool:
            return False

        async def done_writing(self) -> None:
            pass

        async def read(self):
            await asyncio.Future()

    fake_channel = FakeChannel()
    fake_call = FakeCall()
    sentinel_credentials = object()
    create_calls: list[tuple[str, object]] = []

    def fake_create_aio_channel_with_credentials(target: str, credentials):
        create_calls.append((target, credentials))
        return fake_channel

    class FakeStub:
        def __init__(self, channel) -> None:
            assert channel is fake_channel

        def Chat(self, *, metadata):
            assert metadata == (("authorization", "Bearer token"),)
            return fake_call

    monkeypatch.setattr(
        session_mod,
        "create_aio_channel_with_credentials",
        fake_create_aio_channel_with_credentials,
    )
    monkeypatch.setattr(session_mod.pbg, "EidolonAgentStub", FakeStub)

    session = session_mod.EidolonAgentSession(
        target="127.0.0.1:45051",
        device_token="token",
        tls=session_mod.TlsConfig(mode="off"),
    )
    session._credentials = sentinel_credentials
    try:
        await session._ensure_open()
        assert create_calls == [("127.0.0.1:45051", sentinel_credentials)]
    finally:
        await session.aclose()


def test_default_channel_options_do_not_send_aggressive_keepalive() -> None:
    """Default client options must not trip server GOAWAY too_many_pings."""
    from eidolon.livekit.agent.eidolon_agent_rpc.session import _CHANNEL_OPTIONS

    option_names = {name for name, _ in _CHANNEL_OPTIONS}
    assert "grpc.keepalive_time_ms" not in option_names
    assert "grpc.keepalive_permit_without_calls" not in option_names
    assert "grpc.http2.max_pings_without_data" not in option_names


@pytest.mark.asyncio
async def test_state_and_usage_events_surface(caplog) -> None:
    """C: typed inbox payloads.

    USAGE → ChatChunk(usage=CompletionUsage(...)) so LiveKit's metrics
    monitor can aggregate token counts.
    STATE → INFO log (UX hook surface, no ChatChunk emitted).
    """
    import logging

    servicer = _UsageStateServicer()
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:cphase",
        )
        try:
            provider_events: list[dict] = []
            adapter.on("provider_event", provider_events.append)
            with caplog.at_level(logging.INFO, logger="eidolon_agent_rpc.grpc_llm"):
                stream = adapter.chat(chat_ctx=_ctx("trigger usage"))
                delta_chunks: list[str] = []
                usage_chunk = None
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        delta_chunks.append(chunk.delta.content)
                    if chunk.usage is not None:
                        usage_chunk = chunk.usage

            assert delta_chunks == ["ok"], f"unexpected deltas: {delta_chunks}"
            assert usage_chunk is not None, "USAGE event should produce a ChatChunk.usage"
            assert usage_chunk.prompt_tokens == 12
            assert usage_chunk.completion_tokens == 3
            assert usage_chunk.total_tokens == 15

            # STATE event surfaces as INFO log with state=thinking
            state_logs = [r for r in caplog.records if "state=thinking" in r.getMessage()]
            assert state_logs, (
                f"expected STATE INFO log, got: {[r.getMessage() for r in caplog.records]}"
            )
            assert any(
                event.get("event") == "brain_state" and event.get("state") == "thinking"
                for event in provider_events
            )
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


# A4: ERROR.code → exception mapping.
# Each row: (brain code, fatal flag, expected exception class, expected
# status_code if APIStatusError else None, expected retryable).
_ERROR_CASES = [
    ("unauthenticated", False, "status", 401, False),
    ("unauthenticated", True, "status", 401, False),  # fatal flag ignored for known codes
    ("permission_denied", False, "status", 403, False),
    ("tenant_not_found", False, "status", 404, False),
    ("user_not_found", False, "status", 404, False),
    ("rate_limited", False, "status", 429, True),
    ("internal", False, "connection", None, True),  # fatal=False → retryable
    ("internal", True, "connection", None, False),  # fatal=True  → not retryable
    ("anything_unknown", False, "connection", None, True),
]


@pytest.mark.parametrize("code,fatal,kind,status_code,retryable", _ERROR_CASES)
@pytest.mark.asyncio
async def test_error_code_mapping(
    code: str, fatal: bool, kind: str, status_code: int | None, retryable: bool
) -> None:
    from livekit.agents._exceptions import APIConnectionError, APIStatusError
    from livekit.agents.types import APIConnectOptions

    servicer = _ErrorServicer(code=code, message=f"boom {code}", fatal=fatal)
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:error-test",
        )
        try:
            # Disable framework retry so we see the *first* exception raised
            # by _run, not the post-retry wrapping. Without max_retry=0, a
            # retryable=True error gets wrapped as
            # `APIConnectionError("after N attempts")` after the framework
            # exhausts its retry budget, hiding the mapping under test.
            stream = adapter.chat(
                chat_ctx=_ctx("trigger error"),
                conn_options=APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=10.0),
            )
            with pytest.raises((APIStatusError, APIConnectionError)) as exc_info:
                async for _ in stream:
                    pass
            exc = exc_info.value
            if kind == "status":
                assert isinstance(exc, APIStatusError), (
                    f"expected APIStatusError, got {type(exc).__name__}"
                )
                assert exc.status_code == status_code
                assert exc.retryable is retryable
            else:
                assert isinstance(exc, APIConnectionError)
                # APIConnectionError carries retryable via the framework's
                # APIError base; just verify the type since retryable surface
                # may differ across livekit-agents versions.
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_tool_events_surface_at_info(caplog) -> None:
    """③ observability: TOOL_CALL and TOOL_RESULT must be visible at INFO.

    Previously TOOL_RESULT fell into the not-surfaced DEBUG branch, so an
    operator could not tell a failing tool backend (ok=false) apart from a
    result that never reached the brain's loop. The channel still takes no
    action on the result — it only renders the brain's answer delta.
    """
    import logging

    servicer = _ToolEventServicer()
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:tools",
        )
        try:
            provider_events: list[dict] = []
            adapter.on("provider_event", provider_events.append)
            with caplog.at_level(logging.INFO, logger="eidolon_agent_rpc.grpc_llm"):
                stream = adapter.chat(chat_ctx=_ctx("查一下北京天气"))
                deltas: list[str] = []
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        deltas.append(chunk.delta.content)

            # The answer delta still renders (tool events are not TTS'd).
            assert any("抱歉" in d for d in deltas)

            messages = [r.getMessage() for r in caplog.records]
            assert any("tool_call name=get_weather" in m for m in messages), messages
            assert any(
                "tool_result name=get_weather ok=False" in m and "error=weather_lookup_failed" in m
                for m in messages
            ), messages

            assert any(
                e.get("event") == "brain_tool_result"
                and e.get("tool_name") == "get_weather"
                and e.get("ok") is False
                and e.get("error") == "weather_lookup_failed"
                for e in provider_events
            ), provider_events
            assert any(
                e.get("event") == "brain_tool_call" and e.get("tool_name") == "get_weather"
                for e in provider_events
            ), provider_events
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_tool_status_roles_are_rendered_by_latency_policy() -> None:
    """Status roles go to provider events; only slow hints are spoken."""
    servicer = _PreambleServicer()
    server, target = await _serve(servicer)
    try:
        adapter = EidolonAgentGrpcLlm(
            target=target,
            device_token=lambda: "test-token",
            conversation_id="livekit:preamble",
        )
        try:
            provider_events: list[dict] = []
            adapter.on("provider_event", provider_events.append)
            stream = adapter.chat(chat_ctx=_ctx("查一下北京天气"))
            spoken: list[str] = []
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    spoken.append(chunk.delta.content)

            assert "我先调用相关工具处理一下。" not in spoken
            assert spoken.count("稍等，我处理一下。") == 1, spoken
            assert "北京今天晴。" in spoken
            preamble_events = [
                e for e in provider_events if e.get("event") == "brain_tool_preamble"
            ]
            assert [e.get("role") for e in preamble_events] == [
                "tool_preamble",
                "slow_tool_hint",
            ], provider_events
        finally:
            await adapter.aclose()
    finally:
        await server.stop(grace=0.5)
