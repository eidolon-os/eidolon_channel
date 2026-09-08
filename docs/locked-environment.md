# Does this process run what the lock pins?

## The incident this exists for

2026-09-07. The LiveKit agent worker joined every room and died 1.5 seconds
later:

```
TypeError: got an unexpected keyword argument 'transcription_timeout'
  at eidolon/livekit/agent/full_duplex/lifecycle.py:52
```

`lifecycle.py` was correct. `turn_handling`, `aec_warmup_duration` and
`transcription_timeout` are all fields of `AgentSession` in livekit-agents
1.7.1, which is what `pyproject.toml` declares and what `uv.lock` pins. The
installed environment was below that:

| | installed | pyproject declares | uv.lock pins |
|---|---|---|---|
| livekit-agents | 1.6.4 | `>=1.7.1,<1.8` | 1.7.1 |
| livekit-api | 1.1.0 | `>=1.2.1,<1.3` | 1.2.1 |
| livekit-plugins-openai | 1.6.4 | `>=1.7.1,<1.8` | 1.7.1 |

It was not only the package that raised. All three LiveKit distributions were
below their floors, because the declaration moved and the environment never
followed:

- 8-29 — the venv is created, resolving livekit-agents 1.6.4.
- 8-31 — `e71b7e4` raises the pin to 1.7.1. The lock and `pyproject.toml` now
  describe an environment that exists nowhere.
- 9-01 — `821d00d` writes the calls that use the new parameters. The code is now
  correct against the declaration and wrong against every running venv.
- 9-07 — a room is joined and the `TypeError` finally says so, nine days late.

That gap is the thing worth naming: a declaration change is a source change and
gets reviewed, while the environment following it is a command somebody has to
remember to run. Nothing was watching the distance between them.

`uv sync` fixed it. No source change was needed, and finding that out cost two
hours — because nothing in the running system could say the environment was
wrong. The Provider started, connected, dispatched and answered 200 throughout.
`device-channels/current` kept returning success. The phone truthfully showed
「正在聆听」. Every layer reported health, and the only evidence was a
`TypeError` on a code path nobody watches.

## What was added

[`eidolon/locked_environment.py`](../eidolon/locked_environment.py). Both
Channel processes call `require_locked_environment()` before they do anything
else:

- the Provider in [`server.py`](../eidolon/channel_provider/server.py), before
  it loads its configuration or opens its store;
- the worker in [`agent/server.py`](../eidolon/livekit/agent/server.py), right
  after logging is up and before plugins are registered or a room is joined.

It compares `importlib.metadata.version()` for each direct runtime dependency
against the two statements the repository already makes about that dependency,
both of which `uv.lock` carries:

- **the declared range** — `pyproject.toml`'s specifier, copied into the lock as
  `[package.metadata] requires-dist`. This is the code's own claim about which
  API it was written against.
- **the lock pin** — the exact version `uv.lock` resolved. This is the version
  the tests ran against.

An operator or a deploy step can ask the same question without starting a
service:

```bash
.venv/bin/python -m eidolon.locked_environment
```

Exit 0 means the environment can run the code, 1 means it cannot, 2 means the
question could not be answered.

## The three decisions

### Fail closed, or log and continue? Graded — because the two statements do not carry the same weight.

**Outside the declared range → refuse to start.** An install below `>=1.7.1` or
at or above `<1.8` means this code is calling an API that provably is not
there. Refusing is not a working deploy being bricked; it is a broken deploy
being named. On 2026-09-07 the worker was already useless — it just took a room
join and a `TypeError` to find out, instead of one line at boot. A declared
dependency that is not installed at all refuses for the same reason.

**Inside the range but not the pin → WARNING, continue.** numpy 2.4.4 where the
lock says 2.4.3 is real drift: reproducibility is gone and this is not what the
tests ran against. But the API contract holds, and a silent voice channel is a
worse outcome than an environment that is merely not the tested one. So it is
loud in the log and it starts.

**The escape hatch.** `EIDOLON_ALLOW_ENVIRONMENT_DRIFT=1` downgrades a refusal
to an ERROR line that names the override, so a Host that genuinely has to run
outside the declared range has a way forward and the log still says what
happened. It is ignorable. It is never silent.

A refusal exits non-zero after one CRITICAL line naming the remedy, not with a
traceback — this is a wrong environment, not a bug in the code, and a traceback
reads like the opposite. Under supervisord the unit hits `startsecs` and goes
FATAL, which is what the admin UI surfaces.

### Which packages? The ones the lock says are direct runtime dependencies.

Not a list kept in the module. "The packages whose API this code calls" is a
judgement that rots on the first dependency added, and the rot is invisible —
you find out the next time the un-listed package drifts. So the set is read out
of the lock's own entry for this project: `[[package]] name = "eidolon-channel"`
→ `dependencies`, which is exactly `[project].dependencies` after uv normalizes
it. `uv lock` maintains the check for free.

The `dev` extra is excluded. It lives in `[package.optional-dependencies]`, and
a production Host is right not to have pytest installed. Transitive
dependencies are excluded too: the lock pins hundreds of them, this code calls
none of them directly, and reporting all of them would produce exactly the kind
of noise that trains people to skip the log line.

Two consequences worth knowing:

- The `--extra dev` in the remedy is not decoration. `dev` is an
  optional-dependency extra here, not a dependency group, so a plain
  `uv sync` **uninstalls** 22 packages including pytest, ruff and mypy. Use
  `uv sync --frozen --extra dev` on a workstation; drop the flag on a Host that
  wants none of them.
- A drift in a transitive dependency will still surprise us. That is a real gap,
  accepted knowingly: closing it means either checking every pin in the lock, or
  keeping a hand-written list of the transitive packages that matter — the first
  is noise, the second is the judgement that rots.

### Where does it belong? Here, for now, and `eidolon_sdk` when a second service wants it.

[`eidolon_ops/device_management_gate.py`](../../eidolon_ops/src/eidolon_ops/device_management_gate.py)
binds per-repo commits into release artifacts and verifies them. That is the
same idea one layer up, at release time, and it cannot answer this question:
only the process that holds an environment can say what is installed in it, and
`eidolon_ops` is imported by neither Channel process. The call site has to be in
the service.

The mechanism, though, knows nothing about Channel. It takes a distribution name
and a directory to start walking up from. `eidolon-sdk` is already a runtime
dependency of channel, hub, kernel, agent, data, ops, admin and vision, so it is
the right shared home — but an eight-service API is better designed against a
second real caller than against seven hypothetical ones. Promotion is a move of
this one file and no change to it.

## Known limits

- **The lock has to be reachable.** `find_lock` walks up from the installed
  package. `ops/component.toml` execs `.venv/bin/...` inside the repository
  checkout, so the lock is above the package in every shape this repository is
  actually deployed in. If Channel is ever installed as a plain wheel into a venv
  outside a checkout, the check logs `cannot check the environment against
  uv.lock` and the service starts unverified — it degrades to saying so, and
  never to silently passing.
- **The lock's identity is checked.** A `uv.lock` found by walking up from an
  unexpected location is only believed if its root project is `eidolon-channel`.
  Otherwise the check reports it as unverifiable rather than judging against
  another service's pins.
- **An import nobody declared is invisible to this check.** It reads the
  declaration and asks whether the declaration was honoured. It cannot ask
  whether the code's imports correspond to anything declared at all. Measured on
  the same repository: a VAD fallback branch imported `livekit.agents.plugins`
  (which does not exist in livekit-agents 1.7.1) and then
  `livekit-plugins-silero` — and `silero` appears **zero** times in
  `pyproject.toml` and `uv.lock`. That branch was dead twice over while telling
  the reader that installing a package would fix it, and at the declaration
  layer it was in perfect health, because it named nothing the declaration layer
  knows about. So these are two different questions: *are the declared
  dependencies installed at the declared versions* (this check) and *does every
  import resolve, and is every fallback path reachable* (nothing watches this).
  A code path that has never executed is, on the evidence this check reads,
  indistinguishable from a healthy one. Closing that needs a different tool —
  import verification, or actually exercising the path — and this is not it.
- **An editable path dependency's version proves nothing about its source.**
  `eidolon-sdk` is pinned at 0.1.0 and stays there across commits, so a match
  means the distribution is installed, not that the checkout is current. Binding
  source is what the `eidolon_ops` gate does, and it stays that layer's job.
- **A pass means "matched at startup", not "is running the locked version".**
  This is the sharpest limit, and it was measured on 9-07 rather than imagined.
  The `uv sync` that day ran while the worker was still up (the stop step had
  been refused by a permission policy), which left 1.7.1 on disk and 1.6.4 in
  the live interpreter until the restart. A running interpreter keeps what it
  imported, so a mid-flight sync neither fixes it nor is visible to it. A
  startup check cannot see a window that opens after startup — by construction,
  not by oversight. Read a green line as "the environment on disk matched when
  this process started", and keep the operational order: **stop → sync →
  restart**. Syncing under a live worker is what produced the half-swapped venv
  in the first place.
- **A bug in the check does not take the service down.** An unexpected failure
  inside it is logged at ERROR and swallowed. A diagnostic that kills the product
  over its own bug is worse than no diagnostic. Verified refusals still refuse.

## Tests

[`eidolon/channel_provider/tests/test_locked_environment.py`](../eidolon/channel_provider/tests/test_locked_environment.py).
The ones that matter:

- `test_the_2026_09_07_environment_refuses_to_start` — livekit-agents 1.6.4
  against `>=1.7.1,<1.8` with the lock pinning 1.7.1, the exact shape of the
  incident.
- `test_the_checked_packages_are_the_declared_ones_and_not_a_list_kept_here` —
  fails if anyone replaces the derived set with a hand-written one.
- `test_the_real_environment_can_run_the_code_this_repository_locked` — runs the
  check against this repository's real lock and real venv, so a drifted
  workstation is told by `pytest` rather than by a `TypeError` in a room.
- `test_the_provider_asks_before_it_reads_its_configuration` and
  `test_the_worker_asks_before_it_registers_plugins` — the wiring, proved by a
  sentinel escaping before either process does its first real work.
