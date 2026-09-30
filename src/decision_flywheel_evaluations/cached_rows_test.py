import hashlib

import pytest

from .cached_rows import cached_arrow_path, load_cached_manifest_rows
from .datasets import AG_NEWS, normalized_text_hash
from .manifests import DatasetManifest, ExposureStatus, ManifestRecord


def _manifest():
    return DatasetManifest(
        AG_NEWS.name, AG_NEWS.revision, 0, {"candidate": 1, "development": 1, "scoreboard": 1},
        ExposureStatus.CONFIRMATORY_FRESH,
        (ManifestRecord("train-0", "train", 0, "World", normalized_text_hash("world candidate"), "candidate"),
         ManifestRecord("train-1", "train", 1, "Sports", normalized_text_hash("sports development"), "development"),
         ManifestRecord("test-0", "test", 0, "World", normalized_text_hash("world scoreboard"), "scoreboard")), (),
    )


def test_a_pinned_arrow_cache_rehydrates_exact_manifest_rows_without_a_dataset_download(tmp_path):
    manifest = _manifest()
    paths = []
    records = {
        "train": [{"text": "world candidate", "label": 0}, {"text": "sports development", "label": 1}],
        "test": [{"text": "world scoreboard", "label": 0}],
    }
    def reader(path):
        paths.append(path)
        return records[path.stem.rsplit("-", 1)[1]]

    rows = load_cached_manifest_rows(manifest, cache_root=tmp_path, reader=reader)

    assert tuple(row.id for row in rows) == ("train-0", "train-1", "test-0")
    assert paths == [cached_arrow_path(AG_NEWS, "train", tmp_path), cached_arrow_path(AG_NEWS, "test", tmp_path)]
    assert all(row.text not in repr(paths) for row in rows)


def test_cached_row_rehydration_rejects_missing_or_mismatched_manifest_provenance(tmp_path):
    manifest = _manifest()
    def missing(_path): return [{"text": "world candidate", "label": 0}]
    with pytest.raises(ValueError, match="missing"):
        load_cached_manifest_rows(manifest, cache_root=tmp_path, reader=missing)

    def changed(path):
        if path.stem.endswith("train"):
            return [{"text": "changed", "label": 0}, {"text": "sports development", "label": 1}]
        return [{"text": "world scoreboard", "label": 0}]
    with pytest.raises(ValueError, match="provenance"):
        load_cached_manifest_rows(manifest, cache_root=tmp_path, reader=changed)


def test_cached_arrow_paths_pin_the_exact_revision_without_globbing_or_latest_fallback(tmp_path):
    expected = tmp_path / "fancyzhx___ag_news" / "default" / "0.0.0" / AG_NEWS.revision / "ag_news-train.arrow"
    assert cached_arrow_path(AG_NEWS, "train", tmp_path) == expected
