# Can the release we are replacing still read the settings we are shipping?

## The incident this exists for

2026-09-15, pi5. `eidolon-ops ... deploy --activate --cutover-mode reversible`
stopped in `host_application`:

```
cross-release channel settings validation failed:
  File ".../eidolon/livekit/common/config/loader.py", line 193, in _merge_dataclass
ValueError: unknown config field ObservabilityConfig.session_trace_path
```

Nothing was wrong with the candidate release. `session_trace_path` is a field
of `ObservabilityConfig`, `config/settings.yaml` set it to a real path, and the
candidate read the file exactly as written. The interpreter that refused it was
the *other* one — `1650c17d`, the release already running on the Pi, which was
built before the field existed.

## The mechanism

Ops renders **`config/settings.yaml`** — this repository's own settings
document, at the deployed revision — into every product Host's
`/etc/eidolon/channel.yaml` (`eidolon_ops` `product_settings.py`). Settings and
component symlinks then switch in two separate atomic operations, so for a
window the new settings file is on disk while the old code is still what runs.

`_validate_product_settings_compatibility` closes that window by refusing to
touch a live path until the staged file loads under **both** interpreters:

| | | |
|---|---|---|
| `/opt/eidolon/releases/<new>/.../python` | forward safety | always checked |
| `/opt/eidolon/current/.../python` | rollback safety | checked unless `forward-only` |

`--cutover-mode forward-only` drops the second, and says why in the source:
*"The restore this would protect is the one the mode gave up."* That mode
records starting the candidate as a durable barrier; when the post-activation
health gate fails it reports and stops rather than restoring old interpreters.

The refusal itself comes from `_merge_dataclass`, which raises on a field it
does not recognise. That strictness is deliberate and worth keeping: the tests
that pin it (`test_manual_config_sections_reject_unknown_fields`,
`test_behavior_pipeline_mode_is_rejected`, `test_behavior_agent_mode_is_rejected`,
`test_remote_agent_rejects_legacy_device_token_field`) are all about knobs that
were **removed or misspelled** — `pipeline_mode`, `agent_mode`, `typo_mode`,
`device_token`. An operator who writes one of those and is not told is running a
Host in a mode they did not choose. Loosening the loader to ignore unknown
fields would trade that guarantee away, in both directions, and would not have
helped here anyway: the interpreter that had to tolerate the new field was
already installed. Code shipped today cannot make yesterday's code lenient.

## The rule

> A settings field lands in two releases. The release that adds the field to
> the schema ships it with its default and **does not name it in
> `config/settings.yaml`**. Only the next release may write the key.

This is not a new policy. `eidolon_ops` `host_application` has stated it since
the cross-release check was written — *"under `reversible` the schema must
expand in two releases: code defaults first, and only a later release may
require new YAML fields"* — but it was stated only there, in a docstring in
another repository, where nobody editing `config/settings.yaml` would meet it.
That is the whole reason it was broken by an ordinary feature commit.

`config/settings.example.yaml` is **not** rendered by Ops and is not bound by
this. It documents the whole schema, and is where a new field stays visible
while its key waits a release.

### Which documents this binds, and why only one

Channel's two units read two different settings files. The unit files live in
**eidolon_kernel** — `deploy/systemd/eidolon-channel.service` and
`eidolon-channel-provider.service`, which `eidolon_ops` `release_matrix`
declares with `source_id = "eidolon_kernel"` and reads by exact Git object:

| unit | reads | bound by the rule |
|---|---|---|
| `eidolon-channel` (worker) | `EIDOLON_CHANNEL_SETTINGS_YAML=/etc/eidolon/channel.yaml` | **yes** |
| `eidolon-channel-provider` | `EIDOLON_CHANNEL_PROVIDER_SETTINGS_YAML=/opt/eidolon/current/eidolon_channel/config/channel-provider.yaml` | no |

The difference is not which file is stricter — `channel_provider/config.py`
rejects unknown fields exactly as the worker's loader does. It is *where the
file lives*. The worker's is rendered into `/etc`, which outlives a release, so
it is read by whichever release is running — including the one a rollback
restores. The Provider's sits under `/opt/eidolon/current`, a symlink that
flips in the same atomic step as the code that reads it, so the release that
reads it is always the release that shipped it. A settings document has
cross-release exposure only if it outlives the release that wrote it.

This is also why `ops/component.toml` names only the worker in the
`channel_settings` input's `required_by`: the Provider starts without
`/etc/eidolon/channel.yaml`, and nothing it imports reaches the worker's loader.

Adding a whole new top-level section is safe: the loader reads the sections it
knows by name and never rejects a document for containing others. It is a new
key inside a section that already exists which strands the old release.

## What this incident actually cost

`session_trace_path` was the first refusal, not the only problem. Rehearsing the
same check against the deployed revision found **seven** keys, from three
unrelated features, each of which would have surfaced one deploy at a time:

```
llm.extra_body
turn_policy.interrupt.intent_provider
turn_policy.interrupt.intent_timeout_ms
observability.session_trace_path
observability.session_trace_max_queue
observability.session_trace_max_file_bytes
observability.session_trace_retention_days
```

Two of them (`intent_provider`, `intent_timeout_ms`) were written at their own
schema defaults, so naming them bought nothing and cost the rollback.

## Checking before the Pi does

```bash
scripts/check_settings_cross_release.py <the revision installed on the Host>
```

It exports that revision, loads today's `config/settings.yaml` under its loader
and under the working tree's, and names whatever either refuses. Pass the
commit: Ops treats the commit as the release identity and a tag as a label on
it, so there is no tag this repository could check against by itself — which is
also why this is a script you run against a baseline you know, and not a test.
The authority stays the check on the Host, which knows what is installed.
