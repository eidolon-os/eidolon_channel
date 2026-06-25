"""Half-duplex (PTT) manual turn-detection — plan §10 regression matrix.

The bug (real-device 2026-06-22): PTT user says "…今天的天气 <pause> 北京的" in one
hold; the framework's auto-EOT fired at the mid-sentence pause and committed before
release, so the agent replied to the partial and dropped "北京的" (it arrived while
the half-duplex reply was locked).

The fix:
  - half_duplex Agent uses ``turn_detection="manual"`` → NO automatic EOU.
  - the PTT release (ptt True→False edge) is the SOLE turn boundary → exactly one
    ``commit_user_turn`` (with transcript_timeout so an in-flight tail FINAL is
    included); never ``clear_user_turn``.
  - 守空: a no-speech press does not commit (manual would otherwise fire empty EOU).
  - the VAD speaking→listening transition no longer commits in half_duplex.
  - full_duplex is unchanged (EOT model turn_detection; VAD commits).

These tests drive the pipeline directly (no real LiveKit room) and assert the
turn-boundary behaviour for cases S1/S4 + the anti-regressions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from eidolon_sdk.biz.contracts import CLIENT_AUDIO_STATE_TOPIC

from eidolon.livekit.agent.streaming import StreamingPipeline


def _pipeline(*, half_duplex: bool) -> StreamingPipeline:
    """Minimally-initialised pipeline exercising only the PTT/turn slice."""
    p = StreamingPipeline.__new__(StreamingPipeline)
    p._is_half_duplex = half_duplex
    p._last_ptt_held = False
    p._ptt_turn_had_speech = False
    p._stt_commit_transcript_timeout = 5.0
    p._session = MagicMock()
    p._session.commit_user_turn = MagicMock()
    p._get_eot_model = lambda: MagicMock()
    return p


def _ptt_packet(held: bool) -> SimpleNamespace:
    return SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC,
        participant=SimpleNamespace(identity="dev-1"),
    )


def _send_ptt(p: StreamingPipeline, held: bool) -> None:
    # _handle_ptt_turn_edges reads ptt off the stored client audio state.
    p._latest_client_audio_state = lambda **_kw: SimpleNamespace(ptt=held)
    p._handle_ptt_turn_edges(_ptt_packet(held))


# ── turn_detection assembly (S1/S2 contrast) ────────────────────────────


def test_turn_detection_manual_for_half_duplex():
    p = _pipeline(half_duplex=True)
    assert p._turn_detection_for_mode() == "manual"


def test_turn_detection_eot_model_for_full_duplex():
    p = _pipeline(half_duplex=False)
    sentinel = object()
    p._get_eot_model = lambda: sentinel
    # full_duplex keeps the EOT model instance (auto VAD + semantic EOT).
    assert p._turn_detection_for_mode() is sentinel


# ── PTT edges: press arms, release commits once ─────────────────────────


def test_ptt_release_with_speech_commits_once():
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)          # press
    p._ptt_turn_had_speech = True  # speech arrived this hold
    _send_ptt(p, False)         # release → commit
    p._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
    # 守空 accounting reset for the next hold.
    assert p._ptt_turn_had_speech is False


def test_ptt_release_without_speech_is_guarded():
    """守空 (S4): an empty press must NOT commit (manual would fire empty EOU)."""
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)          # press, no speech
    _send_ptt(p, False)         # release
    p._session.commit_user_turn.assert_not_called()


def test_ptt_press_resets_had_speech():
    p = _pipeline(half_duplex=True)
    p._ptt_turn_had_speech = True  # stale from a previous turn
    _send_ptt(p, True)          # press → fresh accounting
    assert p._ptt_turn_had_speech is False


def test_repeated_held_packets_do_not_double_commit():
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)
    p._ptt_turn_had_speech = True
    _send_ptt(p, True)          # still held (heartbeat) — no edge
    p._session.commit_user_turn.assert_not_called()
    _send_ptt(p, False)         # release → exactly one commit
    p._session.commit_user_turn.assert_called_once()


def test_full_duplex_ptt_edges_are_noop():
    """full_duplex never commits on a ptt edge (its turn boundary is VAD/EOT)."""
    p = _pipeline(half_duplex=False)
    _send_ptt(p, True)
    p._ptt_turn_had_speech = True
    _send_ptt(p, False)
    p._session.commit_user_turn.assert_not_called()


# ── transcript arrival drives the 守空 flag (half_duplex only) ───────────


def _drive_transcript(p: StreamingPipeline, text: str) -> None:
    p._ensure_runtime_defaults = lambda: None
    p._suppress_transcripts_until_next_speech = False
    p._agent_output_active_for_interrupts = lambda **_k: False
    p._transcript_is_agent_echo = lambda _t: False
    p._mark_activity = lambda: None
    p._latest_asr_text = ""
    p._ensure_user_turn_coordinator = lambda: None
    p._user_turns = MagicMock()
    p._timeline = None
    p._interrupt_window_active = lambda: False
    # half_duplex never barges in; full_duplex may — but agent isn't speaking
    # here, so the semantic-EOT branch is skipped regardless.
    p._allow_interruptions = not p._is_half_duplex
    p._on_user_transcribed(SimpleNamespace(transcript=text, is_final=True, speaker_id=None))


def test_transcript_sets_had_speech_half_duplex():
    p = _pipeline(half_duplex=True)
    _drive_transcript(p, "北京的天气")
    assert p._ptt_turn_had_speech is True


def test_empty_transcript_does_not_set_had_speech():
    p = _pipeline(half_duplex=True)
    _drive_transcript(p, "   ")
    assert p._ptt_turn_had_speech is False


def test_full_duplex_transcript_does_not_touch_ptt_flag():
    p = _pipeline(half_duplex=False)
    _drive_transcript(p, "北京的天气")
    assert p._ptt_turn_had_speech is False


# ── VAD silence is not a turn boundary in half_duplex (core anti-regression) ──


def _drive_speaking_to_listening(p: StreamingPipeline) -> MagicMock:
    """Drive _on_user_state_changed(speaking→listening) with the heavy
    collaborators stubbed; returns the gated-commit scheduler mock."""
    p._ensure_runtime_defaults = lambda: None
    p._publish_companion_ui_state = lambda *a, **k: None
    p._voiceprint_turns = MagicMock()
    p._voiceprint_turns.finish_turn = MagicMock(return_value=None)
    p._soft_interrupt_active = False
    p._ducking = MagicMock()
    p._ducking.is_suspended = False
    p._callbacks = MagicMock()
    p._skip_commit_after_interrupt_cancel = False
    p._timeline = None
    p._latest_asr_text = "今天的天气"
    p._remember_candidate_voiceprint_task = lambda _t: None
    p._candidate_voiceprint_gate_task = lambda: None
    p._user_turns = MagicMock()
    p._user_turns.selected_text = "今天的天气"
    p._user_turns.active = SimpleNamespace(state="speaking")
    p._should_defer_low_eot_commit = lambda **_k: False
    p._user_turns.finish_speech = MagicMock(
        return_value=SimpleNamespace(action="commit", transcript="今天的天气", delay_sec=0.0)
    )
    p._schedule_voiceprint_gated_commit = MagicMock()
    p._schedule_deferred_low_eot_commit = MagicMock()
    p._clear_session_user_turn = MagicMock()
    p._reset_candidate_voiceprint_tasks = lambda: None
    p._completed_turn_voiceprint_task = None
    p._completed_turn_voiceprint_result = None
    p._completed_turn_voiceprint_timeline = None
    p._on_user_state_changed(
        SimpleNamespace(old_state="speaking", new_state="listening")
    )
    return p._schedule_voiceprint_gated_commit


def test_half_duplex_vad_silence_does_not_commit():
    """A mid-utterance pause (VAD speaking→listening) must NOT commit in
    half_duplex — only PTT release does. This is the core fix: otherwise the
    pause splits the turn and the continuation is dropped."""
    p = _pipeline(half_duplex=True)
    scheduler = _drive_speaking_to_listening(p)
    p._session.commit_user_turn.assert_not_called()
    scheduler.assert_not_called()


def test_full_duplex_vad_silence_still_commits():
    """full_duplex is unchanged: VAD silence drives the (voiceprint-gated) commit."""
    p = _pipeline(half_duplex=False)
    scheduler = _drive_speaking_to_listening(p)
    scheduler.assert_called_once()


# ── scenario S1: one hold, mid-pause two sentences, single release commit ──


def _send_full_packet(p: StreamingPipeline, *, ptt: bool, playback_active: bool = False) -> None:
    """Drive the production packet entry (_on_client_room_packet), which runs BOTH
    the explicit-interrupt handler and the PTT-edge handler — so we can assert
    they coexist without interference."""
    state = SimpleNamespace(
        ptt=ptt,
        playback_state="agent_speaking" if playback_active else "idle",
        participant_identity="dev-1",
        as_timeline_attr=lambda: {},
    )
    p._latest_client_audio_state = lambda **_k: state
    p._sync_room_data_compat_attrs = lambda: None
    p._agent_output_active_for_interrupts = lambda **_k: playback_active
    p._ensure_ducking_controller = lambda: None
    p._ducking = SimpleNamespace(is_cancelled=False)
    p._timeline = None
    packet = SimpleNamespace(
        topic=CLIENT_AUDIO_STATE_TOPIC, participant=SimpleNamespace(identity="dev-1")
    )
    p._on_client_room_packet(packet)


# ── multi-turn + interrupt coexistence (full packet path) ───────────────


def test_multi_turn_each_release_commits_once():
    """Two back-to-back PTT turns each commit exactly once; 守空 accounting
    resets across turns (no leakage)."""
    p = _pipeline(half_duplex=True)
    # turn 1
    _send_ptt(p, True)
    p._ptt_turn_had_speech = True
    _send_ptt(p, False)
    # turn 2
    _send_ptt(p, True)
    assert p._ptt_turn_had_speech is False  # reset on press
    p._ptt_turn_had_speech = True
    _send_ptt(p, False)
    assert p._session.commit_user_turn.call_count == 2


def test_tap_to_stop_during_playback_interrupts_without_spurious_commit():
    """Half-duplex tap-to-stop: PTT press during agent playback hard-cuts the
    agent (force interrupt) AND must NOT spuriously commit a turn on release
    when no speech was spoken (守空). Both handlers run on the same packet."""
    p = _pipeline(half_duplex=True)
    p._duck_cancel_and_interrupt = MagicMock()

    _send_full_packet(p, ptt=True, playback_active=True)   # tap during playback
    p._duck_cancel_and_interrupt.assert_called_once_with(force=True)  # interrupt fired
    assert p._ptt_turn_had_speech is False                 # turn armed, no speech

    _send_full_packet(p, ptt=False, playback_active=False)  # release (said nothing)
    p._session.commit_user_turn.assert_not_called()         # 守空: no spurious commit


def test_interrupt_then_speak_commits_on_release():
    """Tap-to-stop the agent, THEN speak a new turn in the same hold → exactly
    one commit on release."""
    p = _pipeline(half_duplex=True)
    p._duck_cancel_and_interrupt = MagicMock()

    _send_full_packet(p, ptt=True, playback_active=True)   # interrupt
    p._duck_cancel_and_interrupt.assert_called_once_with(force=True)
    p._ptt_turn_had_speech = True                          # user then speaks
    _send_full_packet(p, ptt=False, playback_active=False)  # release → commit
    p._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)


def test_full_packet_normal_turn_commits_once():
    """End-to-end via the production packet path: press → (speech) → release →
    one commit, with no interrupt (agent was idle)."""
    p = _pipeline(half_duplex=True)
    p._duck_cancel_and_interrupt = MagicMock()
    _send_full_packet(p, ptt=True, playback_active=False)
    p._ptt_turn_had_speech = True
    _send_full_packet(p, ptt=False, playback_active=False)
    p._duck_cancel_and_interrupt.assert_not_called()       # nothing to interrupt
    p._session.commit_user_turn.assert_called_once()


# ── packet robustness: duplicates, dropped edges, stray participants ────


def test_duplicate_press_packets_arm_once_no_commit():
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)
    _send_ptt(p, True)   # duplicate held heartbeat
    _send_ptt(p, True)
    p._session.commit_user_turn.assert_not_called()
    assert p._last_ptt_held is True


def test_release_via_heartbeat_commits_once():
    """A dropped release edge self-heals: the device's periodic ptt=False
    heartbeat still produces the falling edge → one commit."""
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)
    p._ptt_turn_had_speech = True
    # (release packet "dropped" — next thing we see is a heartbeat with ptt=False)
    _send_ptt(p, False)
    _send_ptt(p, False)   # further heartbeats — no second commit
    p._session.commit_user_turn.assert_called_once()


def test_stray_participant_packet_does_not_fake_release():
    """A packet from a participant with NO audio state (not the device) must not
    flip the held flag and fake a release mid-hold (the bug fixed in
    _handle_ptt_turn_edges)."""
    p = _pipeline(half_duplex=True)
    _send_ptt(p, True)              # device presses (held)
    p._ptt_turn_had_speech = True
    # stray packet: _latest_client_audio_state returns None for this identity
    p._latest_client_audio_state = lambda **_k: None
    p._handle_ptt_turn_edges(
        SimpleNamespace(
            topic=CLIENT_AUDIO_STATE_TOPIC,
            participant=SimpleNamespace(identity="agent-x"),
        )
    )
    p._session.commit_user_turn.assert_not_called()   # no spurious commit
    assert p._last_ptt_held is True                    # device still held


def test_noise_only_press_does_not_commit_full_path():
    """Press → only whitespace/noise transcript → release → no commit (守空)."""
    p = _pipeline(half_duplex=True)
    p._duck_cancel_and_interrupt = MagicMock()
    _send_full_packet(p, ptt=True, playback_active=False)
    _drive_transcript(p, "   ")   # noise → does not set had_speech
    _send_full_packet(p, ptt=False, playback_active=False)
    p._session.commit_user_turn.assert_not_called()


def test_wrong_topic_packet_ignored():
    p = _pipeline(half_duplex=True)
    p._last_ptt_held = True
    p._handle_ptt_turn_edges(
        SimpleNamespace(topic="eidolon.other", participant=SimpleNamespace(identity="dev-1"))
    )
    p._session.commit_user_turn.assert_not_called()
    assert p._last_ptt_held is True   # untouched


# ── framework contract: the LiveKit fact the manual fix depends on ──────


def test_framework_manual_turn_detection_contract():
    """The half-duplex fix rests on a LiveKit framework guarantee (handoff §5):
    ``_run_eou_detection`` early-returns on an empty transcript in non-manual
    modes, but ``turn_detection="manual"`` bypasses all automatic EOU — turns
    commit only via explicit ``commit_user_turn``. Pin that contract (+ the
    commit/clear API) so a ``livekit-agents`` upgrade that removes the manual
    special-case fails HERE (complements runtime ``check_framework_version``)."""
    import inspect

    from livekit.agents.voice.audio_recognition import AudioRecognition

    assert hasattr(AudioRecognition, "commit_user_turn"), (
        "framework dropped commit_user_turn — manual turn commit (plan §10) broken"
    )
    assert hasattr(AudioRecognition, "clear_user_turn")
    src = inspect.getsource(AudioRecognition._run_eou_detection)
    assert '"manual"' in src or "'manual'" in src, (
        "framework no longer special-cases manual turn_detection in "
        "_run_eou_detection — re-validate the half-duplex manual fix (plan §10)"
    )


def test_scenario_s1_pause_then_continuation_single_commit():
    """End-to-end of the real-device repro: press → "今天的天气" → (VAD pause,
    no commit) → "北京的天气" → release → exactly one commit. The aggregate is
    owned by the framework; we assert our side commits once and never early."""
    p = _pipeline(half_duplex=True)

    _send_ptt(p, True)                       # press
    _drive_transcript(p, "你帮我查询一下今天的天气")  # sentence 1
    _drive_speaking_to_listening(p)          # mid pause → must NOT commit
    p._session.commit_user_turn.assert_not_called()
    _drive_transcript(p, "北京的天气")        # continuation (the tail)
    _send_ptt(p, False)                      # release → single commit

    p._session.commit_user_turn.assert_called_once_with(transcript_timeout=5.0)
