"""Explicit opt-in acquisition of the exact public dataset revisions used here.

This module is intentionally separate from preparation and cached-row reading:
those paths stay offline and will never cause a download.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .cached_rows import cached_arrow_path
from .datasets import AG_NEWS, EMOTION, DatasetSpec


DatasetLoader = Callable[..., Sequence[object]]


@dataclass(frozen=True)
class AcquiredSplit:
    """Text-free confirmation that one exact pinned Arrow split is present."""

    dataset: str
    config: str
    split: str
    revision: str
    rows: int
    arrow_path: str


_PINNED_SPLITS: tuple[tuple[DatasetSpec, tuple[str, ...]], ...] = (
    (AG_NEWS, ("train", "test")),
    (EMOTION, ("train", "test")),
)


def acquire_pinned_datasets(*, cache_root: str | Path = ".data/huggingface", confirmed: bool = False,
                            loader: DatasetLoader | None = None) -> tuple[AcquiredSplit, ...]:
    """Download only after explicit confirmation, pinning every request by SHA revision.

    No token, model, source row, or dataset payload is returned or printed.  The
    resulting Arrow files are local cache inputs, not study artifacts.
    """
    if confirmed is not True:
        raise ValueError("explicit confirmation is required before downloading pinned datasets")
    root = Path(cache_root)
    load = loader or _huggingface_loader
    acquired = []
    for spec, splits in _PINNED_SPLITS:
        for split in splits:
            dataset = load(spec.name, name_config=spec.config, split=split, revision=spec.revision, cache_dir=str(root))
            path = cached_arrow_path(spec, split, root)
            if not path.is_file():
                raise ValueError("dataset acquisition did not create the expected pinned Arrow file")
            try:
                rows = len(dataset)
            except TypeError as error:
                raise ValueError("dataset acquisition returned an unsupported split") from error
            if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
                raise ValueError("dataset acquisition returned an empty or invalid split")
            acquired.append(AcquiredSplit(spec.name, spec.config, split, spec.revision, rows, str(path)))
    return tuple(acquired)


def _huggingface_loader(dataset_name: str, *, name_config: str, split: str, revision: str,
                        cache_dir: str) -> Sequence[object]:
    """Lazy network-capable adapter used only by the explicit acquisition command."""
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - optional runtime integration
        raise RuntimeError("install the data extra to acquire the pinned dataset cache") from error
    return load_dataset(dataset_name, name=name_config, split=split, revision=revision, cache_dir=cache_dir)
