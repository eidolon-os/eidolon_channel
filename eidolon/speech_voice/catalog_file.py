"""The catalog file contract ``speech_voice_catalog/v1``, shared by every adapter.

The envelope and each voice's common fields are validated here, strictly: a
word outside the SDK vocabulary in a file we ship is our own mistake. ``ext`` is
the provider's own block; only its adapter knows what it means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from eidolon_sdk.biz.speech_voice import AGE_BANDS, GENDERS, LANGUAGES, STYLE_LABELS
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA = "speech_voice_catalog/v1"
#: A description may say it does not know the age; such a voice is never mapped.
CATALOG_AGE_BANDS = (*AGE_BANDS, "unspecified")


class CatalogSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    title: str
    url: str
    retrieved_on: str


class CatalogVoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    voice_id: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=64)
    summary: str | None = None
    gender: str
    age_band: str
    languages: list[str] = Field(min_length=1)
    styles: list[str] = Field(default_factory=list)
    lineage: str | None = None
    rank: int
    models: list[str] = Field(min_length=1)
    ext: dict[str, Any] = Field(default_factory=dict)

    @field_validator("gender")
    @classmethod
    def _gender(cls, value: str) -> str:
        if value not in GENDERS:
            raise ValueError(f"unknown gender {value!r}")
        return value

    @field_validator("age_band")
    @classmethod
    def _age(cls, value: str) -> str:
        if value not in CATALOG_AGE_BANDS:
            raise ValueError(f"unknown age band {value!r}")
        return value

    @field_validator("languages")
    @classmethod
    def _languages(cls, value: list[str]) -> list[str]:
        unknown = [v for v in value if v not in LANGUAGES]
        if unknown:
            raise ValueError(f"unknown languages {unknown}")
        return value

    @field_validator("styles")
    @classmethod
    def _styles(cls, value: list[str]) -> list[str]:
        unknown = [v for v in value if v not in STYLE_LABELS]
        if unknown:
            raise ValueError(f"unknown styles {unknown}")
        return value

    @property
    def speaker(self) -> str:
        return self.lineage or self.voice_id


class CatalogFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: str = Field(alias="schema")
    provider: str
    source: CatalogSource
    models: list[str] = Field(min_length=1)
    voices: list[CatalogVoice]

    @model_validator(mode="after")
    def _consistent(self) -> CatalogFile:
        if self.schema_ != SCHEMA:
            raise ValueError(f"unsupported catalog schema {self.schema_!r}")
        seen: set[str] = set()
        for voice in self.voices:
            if voice.voice_id in seen:
                raise ValueError(f"duplicate voice_id {voice.voice_id!r}")
            seen.add(voice.voice_id)
            outside = set(voice.models) - set(self.models)
            if outside:
                raise ValueError(f"{voice.voice_id}: models {sorted(outside)} are not in the file")
        return self


def load_catalog_file(path: Path, *, provider: str) -> CatalogFile:
    catalog = CatalogFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if catalog.provider != provider:
        raise ValueError(f"{path.name} describes {catalog.provider!r}, not {provider!r}")
    return catalog
