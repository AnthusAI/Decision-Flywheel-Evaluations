from pathlib import Path

import pytest

from .datasets import AG_NEWS, EMOTION
from .download import acquire_pinned_datasets


def test_an_explicit_confirmed_acquisition_requests_only_the_pinned_revisions_configs_and_splits(tmp_path):
    calls = []
    def loader(name, *, name_config, split, revision, cache_dir):
        calls.append((name, name_config, split, revision, cache_dir))
        spec = AG_NEWS if name == AG_NEWS.name else EMOTION
        path = Path(cache_dir) / spec.name.replace("/", "___") / spec.config / "0.0.0" / spec.revision / f"{spec.name.rsplit('/', 1)[1]}-{split}.arrow"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return range(3)

    summary = acquire_pinned_datasets(cache_root=tmp_path, confirmed=True, loader=loader)

    assert [(item.dataset, item.config, item.split, item.revision, item.rows) for item in summary] == [
        (AG_NEWS.name, AG_NEWS.config, "train", AG_NEWS.revision, 3),
        (AG_NEWS.name, AG_NEWS.config, "test", AG_NEWS.revision, 3),
        (EMOTION.name, EMOTION.config, "train", EMOTION.revision, 3),
        (EMOTION.name, EMOTION.config, "test", EMOTION.revision, 3),
    ]
    assert all(call[3] in {AG_NEWS.revision, EMOTION.revision} for call in calls)
    assert all("text" not in repr(item) for item in summary)


def test_an_unconfirmed_or_incomplete_acquisition_never_claims_a_ready_pinned_cache(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="explicit confirmation"):
        acquire_pinned_datasets(cache_root=tmp_path, confirmed=False, loader=lambda *_args, **_kwargs: calls.append("called"))
    assert calls == []

    def missing(*_args, **_kwargs): return range(1)
    with pytest.raises(ValueError, match="did not create the expected pinned Arrow file"):
        acquire_pinned_datasets(cache_root=tmp_path, confirmed=True, loader=missing)
