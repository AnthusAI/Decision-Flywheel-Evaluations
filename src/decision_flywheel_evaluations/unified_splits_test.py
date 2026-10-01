import gzip
import json

import pytest

from . import unified_env
from .unified_splits import (
    LeakageError, assert_labels_from_pool, assert_retrieval_pool_clean, assert_target_excluded,
    dev_slice, label_order, load_splits, round_batches)


def _fixtures(tmp_path, *, pool=40, test=700, paper=600):
    rows = []
    for i in range(pool):
        rows.append({"id": f"p{i:04d}", "text": f"pool text {i}",
                     "metadata": {"split": "pool", "reference_label": "positive" if i % 2 else "negative"}})
    for i in range(test):
        rows.append({"id": f"t{i:04d}", "text": f"test text {i}",
                     "metadata": {"split": "test", "reference_label": "negative" if i % 3 else "positive"}})
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    extra = tmp_path / "recordings" / "simulated-labeler"
    extra.mkdir(parents=True)
    with gzip.open(extra / "extra_answers.jsonl.gz", "wt") as handle:
        for i in range(paper):
            handle.write(json.dumps({"item_id": f"t{i:04d}", "name": "x", "qhash": "h", "answer": {}}) + "\n")
        for i in range(5):  # recorded pool labels also carry extra answers; they are not test items
            handle.write(json.dumps({"item_id": f"p{i:04d}", "name": "x", "qhash": "h", "answer": {}}) + "\n")
    return tmp_path


def test_dev_slice_is_a_fixed_seeded_sample_of_test_items_outside_paper_600(tmp_path):
    splits = load_splits(_fixtures(tmp_path))
    assert len(splits.paper600) == 600 and len(splits.dev100) == 100
    assert not set(splits.dev100) & set(splits.paper600)
    assert set(splits.dev100) <= set(splits.test)
    assert splits.dev100 == load_splits(tmp_path).dev100


def test_paper_600_is_reachable_only_through_an_explicit_final_flag(tmp_path):
    splits = load_splits(_fixtures(tmp_path))
    assert splits.evaluation_slice(final=False) == ("dev-100", splits.dev100)
    assert splits.evaluation_slice(final=True) == ("paper-600", splits.paper600)


def test_the_label_order_is_a_seeded_permutation_of_the_pool_only(tmp_path):
    splits = load_splits(_fixtures(tmp_path))
    order = label_order(splits.pool, 1)
    assert sorted(order) == sorted(splits.pool)
    assert order == label_order(splits.pool, 1)
    assert order != label_order(splits.pool, 2)
    batches = round_batches(order, rounds=3, per_round=10)
    assert [len(b) for b in batches] == [10, 10, 10]
    assert len({i for b in batches for i in b}) == 30


def test_labeled_items_must_come_from_the_pool(tmp_path):
    splits = load_splits(_fixtures(tmp_path))
    assert_labels_from_pool(splits.pool[:5], splits)
    with pytest.raises(LeakageError):
        assert_labels_from_pool([splits.pool[0], splits.dev100[0]], splits)


def test_a_retrieval_pool_containing_any_held_out_item_is_refused(tmp_path):
    splits = load_splits(_fixtures(tmp_path))
    assert_retrieval_pool_clean(splits.pool, splits)
    for held_out in (splits.dev100[0], splits.paper600[0]):
        with pytest.raises(LeakageError):
            assert_retrieval_pool_clean([*splits.pool[:3], held_out], splits)


def test_a_target_in_its_own_context_is_refused():
    assert_target_excluded("a", ["b", "c"])
    with pytest.raises(LeakageError):
        assert_target_excluded("a", ["b", "a"])


def test_dev_slice_never_draws_from_paper_600_even_when_few_candidates_remain():
    test = [f"t{i}" for i in range(110)]
    assert set(dev_slice(test, test[:10], size=100)) == set(test[10:])


@pytest.mark.skipif(not unified_env.DEFAULT_CLONE.exists(), reason="pinned Jev-Flywheel clone not prepared")
def test_the_real_corpus_has_the_sizes_the_plan_states():
    splits = load_splits(unified_env.DEFAULT_CLONE / "fixtures")
    assert (len(splits.pool), len(splits.test), len(splits.paper600), len(splits.dev100)) == (5280, 3521, 600, 100)
