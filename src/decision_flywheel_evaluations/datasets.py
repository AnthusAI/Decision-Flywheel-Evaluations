"""Pinned, injectable dataset rows for offline study preparation."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping


_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    revision: str
    labels: tuple[str, ...]
    config: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.revision, str) or not _REVISION.fullmatch(self.revision):
            raise ValueError("dataset revision must be a 40-character lowercase hexadecimal SHA")
        if not isinstance(self.labels, tuple) or not self.labels or any(not isinstance(label, str) or not label for label in self.labels) \
                or len(set(self.labels)) != len(self.labels):
            raise ValueError("dataset labels must be nonempty, ordered, and unique")


AG_NEWS = DatasetSpec(
    "fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4",
    ("World", "Sports", "Business", "Sci/Tech"),
)
EMOTION = DatasetSpec(
    "dair-ai/emotion", "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
    ("sadness", "joy", "love", "anger", "fear", "surprise"), "split",
)
_SPECS = {"ag_news": AG_NEWS, "emotion": EMOTION}


@dataclass(frozen=True)
class DatasetRow:
    """One source row; text stays in memory and is never manifest serialized."""

    id: str
    source_split: str
    source_index: int
    label: str
    text: str


def dataset_spec(name: str, revision: str | None = None) -> DatasetSpec:
    """Return a supported pinned dataset, rejecting revision substitution."""
    try:
        spec = _SPECS[name]
    except KeyError as error:
        raise ValueError(f"unsupported dataset {name!r}") from error
    if revision is not None:
        if not _REVISION.fullmatch(revision):
            raise ValueError("dataset revision must be a 40-character lowercase hexadecimal SHA")
        if revision != spec.revision:
            raise ValueError(f"{name} must use recorded revision {spec.revision}")
    return spec


def normalized_text(text: str) -> str:
    """Match the core NFKC/casefold/whitespace identity rule."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def _label(spec: DatasetSpec, value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"label index {value} is not a canonical label for {spec.name}")
        try:
            return spec.labels[value]
        except IndexError as error:
            raise ValueError(f"label index {value} is not a canonical label for {spec.name}") from error
    if value in spec.labels:
        return str(value)
    raise ValueError(f"label {value!r} is not a canonical label for {spec.name}")


def rows_from_records(spec: DatasetSpec, source_split: str, records: Iterable[Mapping[str, object]]) -> tuple[DatasetRow, ...]:
    """Adapt injected dataset records without importing a network-capable client."""
    if not source_split:
        raise ValueError("source_split is required")
    output = []
    for index, record in enumerate(records):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{source_split} row {index} has no text")
        output.append(DatasetRow(f"{source_split}-{index}", source_split, index, _label(spec, record.get("label")), text))
    return tuple(output)


def load_huggingface_split(spec: DatasetSpec, source_split: str, *, cache_dir: str | None = None) -> tuple[DatasetRow, ...]:
    """Opt-in convenience loader; tests should use :func:`rows_from_records`."""
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional runtime integration
        raise RuntimeError("install the data extra to load upstream datasets") from error
    records = load_dataset(spec.name, name=spec.config, split=source_split, revision=spec.revision, cache_dir=cache_dir)
    return rows_from_records(spec, source_split, records)
