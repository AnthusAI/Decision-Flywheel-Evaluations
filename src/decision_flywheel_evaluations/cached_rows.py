"""Strict, download-free rehydration from the repository's pinned Arrow cache."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from .datasets import DatasetRow, DatasetSpec, dataset_spec, normalized_text_hash, rows_from_records
from .manifests import DatasetManifest


ArrowReader = Callable[[Path], Iterable[Mapping[str, object]]]


def cached_arrow_path(spec: DatasetSpec, source_split: str, cache_root: str | Path = ".data/huggingface") -> Path:
    """Return the one supported cache location; never search or choose a latest revision."""
    if not isinstance(source_split, str) or source_split not in {"train", "test", "validation"}:
        raise ValueError("cached source split is unsupported")
    dataset_name = spec.name.rsplit("/", 1)[-1]
    return Path(cache_root) / spec.name.replace("/", "___") / spec.config / "0.0.0" / spec.revision / f"{dataset_name}-{source_split}.arrow"


def load_cached_manifest_rows(
    manifest: DatasetManifest,
    *,
    cache_root: str | Path = ".data/huggingface",
    reader: ArrowReader | None = None,
    roles: tuple[str, ...] | None = None,
) -> tuple[DatasetRow, ...]:
    """Rehydrate exactly manifest-selected rows from explicit pinned Arrow files.

    This is intentionally not a dataset loader: it never imports or calls
    ``load_dataset``, and rejects an absent or incompatible cache rather than
    downloading, globbing revisions, or selecting a newest artifact.
    """
    manifest.validate()
    requested_roles = ("candidate", "development", "scoreboard") if roles is None else tuple(roles)
    allowed_roles = {"candidate", "development", "scoreboard"}
    if (not requested_roles or len(set(requested_roles)) != len(requested_roles)
            or any(role not in allowed_roles for role in requested_roles)):
        raise ValueError("roles must be unique manifest partition names")
    spec = dataset_spec(_short_name(manifest.dataset), manifest.revision)
    if spec.name != manifest.dataset:
        raise ValueError("manifest dataset is unsupported by the pinned cache reader")
    selected_records = tuple(record for record in manifest.records if record.role in requested_roles)
    source_splits = tuple(dict.fromkeys(record.source_split for record in selected_records))
    if not source_splits:
        raise ValueError("manifest has no rows to rehydrate")
    read = reader or _datasets_arrow_reader
    by_id: dict[str, DatasetRow] = {}
    for source_split in source_splits:
        path = cached_arrow_path(spec, source_split, cache_root)
        try:
            source_rows = tuple(rows_from_records(spec, source_split, read(path)))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError("pinned cached Arrow source is unavailable or unsupported") from error
        for row in source_rows:
            if row.id in by_id:
                raise ValueError("pinned cached Arrow source contains duplicate row identities")
            by_id[row.id] = row
    selected = []
    for record in selected_records:
        row = by_id.get(record.id)
        if row is None:
            raise ValueError("pinned cached Arrow source is missing a manifest row")
        if (row.source_split != record.source_split or row.source_index != record.source_index
                or row.label != record.label or normalized_text_hash(row.text) != record.normalized_text_sha256):
            raise ValueError("pinned cached Arrow row does not match manifest provenance")
        selected.append(row)
    if len({row.id for row in selected}) != len(selected):
        raise ValueError("manifest selected duplicate cached row identities")
    return tuple(selected)


def _datasets_arrow_reader(path: Path) -> Iterable[Mapping[str, object]]:
    if not path.is_file():
        raise OSError("missing pinned Arrow file")
    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - optional runtime integration
        raise ValueError("install the data extra to read the local Arrow cache") from error
    try:
        dataset = Dataset.from_file(str(path))
        return tuple(dataset[index] for index in range(len(dataset)))
    except Exception as error:  # pragma: no cover - pyarrow/datasets dependent
        raise ValueError("pinned Arrow file is unsupported") from error


def _short_name(dataset: str) -> str:
    if dataset == "fancyzhx/ag_news":
        return "ag_news"
    if dataset == "dair-ai/emotion":
        return "emotion"
    raise ValueError("manifest dataset is unsupported by the pinned cache reader")
