"""The one function that decides which voice a session speaks with.

Pure: no IO, no clock. channel-provider runs it for Admin's view and for the
worker, so what the Owner reads on the phone and what the session uses come
from the same decision. ``resolve_without_catalog`` is the worker's fallback
when channel-provider cannot be reached.
"""

from __future__ import annotations

from collections.abc import Mapping

from eidolon_sdk.biz.speech_voice import (
    CatalogVoiceSelection,
    CompanionSpeechVoice,
    HostVoiceCatalog,
    OwnerVoiceRuntime,
    OwnerVoiceSelection,
    ResolvedSpeechVoice,
    VoiceOption,
    VoiceRef,
    catalog_voice_key,
    owner_voice_key,
)

from .catalog_file import CatalogVoice

#: Every voice any registered adapter describes, by (provider, voice_id).
Overlay = Mapping[tuple[str, str], CatalogVoice]


def resolve(
    speech_voice: CompanionSpeechVoice | None, catalog: HostVoiceCatalog, overlay: Overlay
) -> ResolvedSpeechVoice:
    default = next((v for v in catalog.voices if v.is_host_default), None)

    def host(reason: str) -> ResolvedSpeechVoice:
        return ResolvedSpeechVoice(
            ref=default.ref if default else None, voice_key=default.voice_key if default else None,
            display_name=default.display_name if default else "主机默认", reason=reason,
            catalog_revision=catalog.catalog_revision,
        )

    def use(option: VoiceOption, reason: str) -> ResolvedSpeechVoice:
        return ResolvedSpeechVoice(
            ref=option.ref, voice_key=option.voice_key, display_name=option.display_name,
            reason=reason, catalog_revision=catalog.catalog_revision,
        )

    if catalog.not_selectable_reason == "no_adapter":
        return host("host_fixed_voice")
    if speech_voice is None:
        return host("unset")
    if speech_voice.record_state == "invalid":
        return host("invalid_record")
    selection = speech_voice.selection
    if selection is None:
        return host("unset")
    if isinstance(selection, OwnerVoiceSelection):
        return _owner_voice(speech_voice.owner_voice, default, host, catalog.catalog_revision)
    if not catalog.host_selectable:
        return host("host_fixed_voice")

    selectable = [v for v in catalog.voices if v.selectable]
    chosen = selection.ref
    exact = next((v for v in selectable if v.ref == chosen), None)
    if exact is not None:
        return use(exact, "selected")
    original = overlay.get((chosen.provider, chosen.voice_id))
    if original is None:
        return host("no_equivalent")
    same_speaker = [
        v for v in selectable
        if v.ref.provider == chosen.provider
        and (described := overlay.get((v.ref.provider, v.ref.voice_id))) is not None
        and described.speaker == original.speaker
    ]
    if same_speaker:
        return use(min(same_speaker, key=lambda v: (v.rank, v.voice_key)), "same_voice_other_model")
    mapped = _closest(original, selectable)
    return use(mapped, "mapped") if mapped is not None else host("no_equivalent")


def _owner_voice(owner_voice: OwnerVoiceRuntime | None, default, host, revision) -> ResolvedSpeechVoice:
    if owner_voice is None or owner_voice.state != "active" or default is None:
        return host("owner_voice_unavailable")
    current = default.ref
    binding = next(
        (b for b in owner_voice.bindings if (b.provider, b.model) == (current.provider, current.model)), None
    )
    if binding is None:
        return host("owner_voice_unavailable")
    if binding.status == "deploying":
        return host("owner_voice_pending")
    if binding.status != "ready":
        return host("owner_voice_rejected")
    return ResolvedSpeechVoice(
        ref=VoiceRef(provider=binding.provider, model=binding.model, voice_id=binding.provider_voice_id),
        voice_key=owner_voice_key(owner_voice.voice_asset_id), display_name=owner_voice.name,
        reason="owner_voice", catalog_revision=revision,
    )


def _closest(original: CatalogVoice, candidates: list[VoiceOption]) -> VoiceOption | None:
    """Same gender, same age band, speaks the original's first language; then the nearest styles."""
    if original.gender == "unspecified" or original.age_band == "unspecified":
        return None
    language = original.languages[0]
    eligible = [
        v for v in candidates
        if v.gender == original.gender and v.age_band == original.age_band and language in v.languages
    ]
    if not eligible:
        return None
    styles = set(original.styles)

    def score(option: VoiceOption) -> tuple[float, int, str]:
        union = styles | set(option.styles)
        jaccard = len(styles & set(option.styles)) / len(union) if union else 0.0
        return (-jaccard, option.rank, option.voice_key)

    return min(eligible, key=score)


def resolve_without_catalog(
    speech_voice: CompanionSpeechVoice | None, current: VoiceRef | None, *, reason: str
) -> ResolvedSpeechVoice:
    """channel-provider is unreachable: keep a choice that certainly fits this config, else the default.

    A stored choice was checked against the catalog when it was saved; a provider
    that is briefly unreachable is far likelier than a voice withdrawn meanwhile,
    and changing voice for a whole session is the worse failure.
    """
    selection = speech_voice.selection if speech_voice and speech_voice.record_state == "valid" else None
    if current is not None:
        if isinstance(selection, CatalogVoiceSelection) and (
            selection.provider, selection.model) == (current.provider, current.model):
            return ResolvedSpeechVoice(
                ref=selection.ref, voice_key=catalog_voice_key(selection.ref),
                display_name=selection.display_name, reason="selected_unverified",
            )
        owner_voice = speech_voice.owner_voice if speech_voice else None
        if isinstance(selection, OwnerVoiceSelection) and owner_voice is not None:
            binding = next((b for b in owner_voice.bindings if b.status == "ready"
                            and (b.provider, b.model) == (current.provider, current.model)), None)
            if binding is not None:
                return ResolvedSpeechVoice(
                    ref=VoiceRef(provider=binding.provider, model=binding.model,
                                 voice_id=binding.provider_voice_id),
                    voice_key=owner_voice_key(owner_voice.voice_asset_id),
                    display_name=owner_voice.name, reason="selected_unverified",
                )
    return ResolvedSpeechVoice(
        ref=current, voice_key=catalog_voice_key(current) if current else None,
        display_name="主机默认", reason=reason,
    )
