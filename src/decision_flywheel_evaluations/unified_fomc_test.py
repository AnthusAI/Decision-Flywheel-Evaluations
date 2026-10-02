from collections import Counter

import pytest

from .unified_fomc import (FOMC_LABELS, FOMC_REPO_ROOT, FOMC_TRAIN_CSV, build_fomc_splits, ceiling_score_config,
                           fomc_label_order, guideline_text, seed_score_config, starting_rubric)
from .unified_splits import LeakageError, assert_labels_from_pool


def _rows(prefix, n, shift=0):
    return [(f"fomc-{prefix}-{k}", FOMC_LABELS[(k + shift) % 3] if k % 2 else "neutral",
             f"{prefix} sentence number {k} about the policy rate") for k in range(n)]


def _splits(**kwargs):
    return build_fomc_splits(_rows("train", 400), _rows("test", 150, shift=1), dev_size=30, **kwargs)


def test_the_pool_holds_only_train_items_and_the_evaluation_slices_only_test_items():
    splits, _ = _splits()
    assert splits.pool and all(i.startswith("fomc-train-") for i in splits.pool)
    assert all(i.startswith("fomc-test-") for i in splits.dev100 + splits.paper600)
    assert all(splits.items[i].split == "pool" for i in splits.pool)


def test_dev_and_final_are_disjoint_and_final_is_every_remaining_test_item():
    splits, report = _splits()
    assert not set(splits.dev100) & set(splits.paper600)
    assert len(splits.dev100) == 30 and len(splits.paper600) == 150 - 30 == report.final_size
    assert set(splits.dev100) | set(splits.paper600) == set(splits.test)


def test_the_dev_slice_is_stable_for_its_seed_and_class_stratified():
    first, _ = _splits()
    again, _ = _splits()
    assert first.dev100 == again.dev100 and first.paper600 == again.paper600
    test_share = Counter(first.items[i].reference_label for i in first.test)
    dev = Counter(first.items[i].reference_label for i in first.dev100)
    for label in FOMC_LABELS:
        assert abs(dev[label] - 30 * test_share[label] / len(first.test)) < 1


def test_a_test_item_whose_text_repeats_a_pool_item_reaches_no_evaluation_slice():
    train = _rows("train", 400)
    test = _rows("test", 150, shift=1)
    test[5] = (test[5][0], test[5][1], "  " + train[7][2].upper())
    splits, report = build_fomc_splits(train, test, dev_size=30)
    assert report.duplicate_test_items_dropped == 1
    assert test[5][0] not in splits.dev100 + splits.paper600


def test_a_repeated_pool_text_is_kept_once_so_the_pool_has_no_duplicate_examples():
    train = _rows("train", 400)
    train[9] = (train[9][0], train[9][1], train[3][2] + " ")
    splits, report = build_fomc_splits(train, _rows("test", 150, shift=1), dev_size=30)
    assert report.pool_duplicates_dropped == 1 and train[3][0] in splits.pool and train[9][0] not in splits.items


def test_duplicate_item_ids_are_refused():
    rows = _rows("train", 10) + [_rows("train", 1)[0]]
    with pytest.raises(ValueError, match="duplicate item id"):
        build_fomc_splits(rows, _rows("test", 50), dev_size=10)


def test_the_label_order_is_class_stratified_so_a_round_of_100_is_34_33_33():
    splits, _ = _splits()
    order = fomc_label_order(splits, 1)
    assert set(order) == set(splits.pool)
    assert sorted(Counter(splits.items[i].reference_label for i in order[:99]).values()) == [33, 33, 33]
    assert fomc_label_order(splits, 1) == order != fomc_label_order(splits, 2)
    assert_labels_from_pool(order, splits)


def test_a_held_out_item_cannot_be_used_as_a_label():
    splits, _ = _splits()
    with pytest.raises(LeakageError):
        assert_labels_from_pool([splits.dev100[0]], splits)


def test_the_seed_question_is_a_three_option_choice_whose_instructions_are_exactly_the_one_line_rubric():
    config = seed_score_config()
    assert config["question_type"] == "choice" and list(config["criteria"]) == ["dovish", "hawkish", "neutral"]
    raw = (FOMC_REPO_ROOT / "studies/rubric_screen/prompts/fomc/S.txt").read_text(encoding="utf-8")
    assert config["instructions"] == starting_rubric() == raw.strip()
    assert "\n" not in config["instructions"]


def test_the_ceiling_question_differs_only_in_carrying_the_full_guideline():
    seed, ceiling = seed_score_config(), ceiling_score_config()
    assert ceiling["instructions"] == guideline_text() and ceiling["instructions"].startswith(seed["instructions"])
    assert {k: v for k, v in seed.items() if k != "instructions"} == {k: v for k, v in ceiling.items() if k != "instructions"}


@pytest.mark.skipif(not (FOMC_REPO_ROOT / FOMC_TRAIN_CSV).is_file(), reason="local FOMC CSVs absent")
def test_the_real_csvs_give_unique_ids_a_full_train_pool_and_a_dev_of_100():
    from .unified_fomc import load_fomc_splits

    splits, report = load_fomc_splits()
    assert len(splits.pool) == 1984 - report.pool_duplicates_dropped == 1941 and len(splits.test) == 496 and len(splits.dev100) == 100
    assert len(splits.paper600) == report.final_size
