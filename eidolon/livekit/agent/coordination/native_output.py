"""Present already-generated Companion text through its existing AgentSession.

Room admission, output policy and the session's TTS/RoomIO belong to composition.
There is no media transport here. SDK playout and remote playback confirmation
remain distinct. By default sequence on native playout, without claiming a
remote hardware drain; an installation may supply an additional confirmation.
"""
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from livekit.agents.voice import AgentSession
from eidolon_sdk.biz.control.coordination_stream import ReplyStart


@dataclass(frozen=True)
class SpeechEndpoint:
    session: AgentSession
    # Optional additional device evidence. When absent the result means only
    # native SpeechHandle playout, explicitly labeled on the scene receipt.
    confirm_playback: Callable[[ReplyStart], Awaitable[bool]] | None = None


class NativeSpeechPresenter:
    def __init__(self, endpoints: dict[tuple[str, str], SpeechEndpoint]):
        self._endpoints = dict(endpoints)

    async def __call__(self, start: ReplyStart, text: AsyncIterator[str], speaking):
        endpoint = self._endpoints[(start.companion_id, start.device_id)]
        session = endpoint.session
        consumed = False
        failed = False
        nonempty = False

        async def words():
            nonlocal consumed, nonempty, failed
            try:
                async for part in text:
                    nonempty |= bool(part.strip())
                    yield part
                consumed = True
            except Exception:
                # Close the native speech on upstream failure. Letting an input
                # iterator error escape can leave the SDK's playout waiter open.
                failed = True
                handle.interrupt(force=True)

        # The Companion executor already owns model generation and history.
        # say() invokes the configured TTS, never the LLM or ASR a second time.
        handle = session.say(words(), allow_interruptions=True, add_to_chat_ctx=False)

        def state_changed(event):
            if session.current_speech is handle and event.new_state == "speaking":
                speaking()  # SDK output state, not a physical completion receipt.

        def on_error(event):
            nonlocal failed
            failed = True
            handle.interrupt(force=True)

        session.on("error", on_error)
        session.on("agent_state_changed", state_changed)
        completed = False
        try:
            await handle.wait_for_playout()
            if failed or handle.interrupted or not consumed or not nonempty:
                return False
            completed = (await endpoint.confirm_playback(start) is True
                         if endpoint.confirm_playback is not None else True)
            return completed and not failed and not handle.interrupted
        finally:
            session.off("error", on_error)
            session.off("agent_state_changed", state_changed)
            if not completed:
                handle.interrupt(force=True)
