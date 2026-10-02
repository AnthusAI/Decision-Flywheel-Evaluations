import json
from collections import Counter
from pathlib import Path

import pytest

from .datasets import EMOTION, normalized_text_hash
from .unified_corpus import CORPORA, PLANTED, get_corpus
from .unified_emotion import (
    EMOTION_DEV_SLICE_SEED, EMOTION_FINAL_SLICE_SEED, EMOTION_FINAL_SIZE, EMOTION_LABELS, EMOTION_REPO_ROOT,
    EmotionSplitReport, assert_no_duplicate_text, build_emotion_splits, emotion_label_order, load_emotion_splits)
from .unified_splits import LeakageError, assert_partitions_disjoint, label_order, round_batches


def _rows(*, per_label=30, scoreboard=400, shared_text=()):
    """Synthetic (id, label, text) rows: train-N candidates and test-N scoreboard items."""
    labels = EMOTION_LABELS
    candidates, scoreboard_rows = [], []
    for n in range(per_label * len(labels)):
        candidates.append((f"train-{n}", labels[n % len(labels)], f"candidate text number {n}"))
    for n in range(scoreboard):
        scoreboard_rows.append((f"test-{n}", labels[(n * 7) % len(labels)], f"scoreboard text number {n}"))
    for test_n, train_n in shared_text:
        scoreboard_rows[test_n] = (f"test-{test_n}", scoreboard_rows[test_n][1], f"  Candidate TEXT number {train_n}")
    return candidates, scoreboard_rows


def _build(**kw):
    candidates, scoreboard = _rows(**{k: v for k, v in kw.items() if k in ("per_label", "scoreboard", "shared_text")})
    return build_emotion_splits(candidates, scoreboard, dev_size=kw.get("dev_size", 20), final_size=kw.get("final_size", 60))


def test_item_ids_are_train_n_for_the_pool_and_test_n_for_the_scoreboard():
    splits, _ = _build()
    assert all(i.startswith("train-") for i in splits.pool) and len(splits.pool) == 180
    assert all(i.startswith("test-") for i in splits.test) and len(splits.test) == 400
    assert splits.items["train-3"].split == "pool" and splits.items["test-3"].split == "test"
    assert splits.items["train-3"].reference_label == EMOTION_LABELS[3 % 6]


def test_the_evaluation_slices_come_only_from_the_scoreboard_and_are_disjoint_and_sized():
    splits, report = _build()
    assert len(splits.dev100) == 20 and len(splits.paper600) == 60
    assert not set(splits.dev100) & set(splits.paper600)
    assert set(splits.dev100) | set(splits.paper600) <= set(splits.test)
    assert not set(splits.pool) & set(splits.test)
    assert_partitions_disjoint(splits)
    assert isinstance(report, EmotionSplitReport)


def test_the_final_slice_is_reachable_only_through_an_explicit_final_flag():
    splits, _ = _build()
    assert splits.evaluation_slice(final=False)[0] == "dev-100"
    assert splits.evaluation_slice(final=True) == ("paper-600", splits.paper600)


def test_the_slices_are_seed_stable_and_the_seeds_are_named_constants():
    first, _ = _build()
    second, _ = _build()
    assert first.dev100 == second.dev100 and first.paper600 == second.paper600
    assert EMOTION_DEV_SLICE_SEED != EMOTION_FINAL_SLICE_SEED
    assert EMOTION_FINAL_SIZE == 600


def test_exact_duplicate_text_is_dropped_from_the_evaluation_slices_and_counted():
    clean, clean_report = _build()
    assert clean_report.duplicate_scoreboard_items_dropped == 0
    shared = tuple((n, n) for n in range(0, 30))   # 30 scoreboard items repeat pool text (case and spacing differ)
    splits, report = _build(shared_text=shared)
    assert report.duplicate_scoreboard_items_dropped == 30
    banned = {f"test-{n}" for n, _ in shared}
    assert not banned & (set(splits.dev100) | set(splits.paper600))
    assert_no_duplicate_text(splits)


def test_a_scoreboard_item_repeating_an_earlier_scoreboard_text_is_dropped_too():
    candidates, scoreboard = _rows()
    scoreboard[5] = ("test-5", scoreboard[5][1], scoreboard[4][2].upper())
    splits, report = build_emotion_splits(candidates, scoreboard, dev_size=20, final_size=60)
    assert report.duplicate_scoreboard_items_dropped == 1
    assert not {"test-4", "test-5"} <= (set(splits.dev100) | set(splits.paper600))


def test_the_leakage_assert_fires_on_an_injected_text_overlap():
    splits, _ = _build()
    leaked_id = splits.dev100[0]
    pool_id = splits.pool[0]
    items = dict(splits.items)
    items[leaked_id] = type(items[leaked_id])(leaked_id, "test", items[leaked_id].reference_label, items[pool_id].text)
    with pytest.raises(LeakageError, match="duplicate"):
        assert_no_duplicate_text(type(splits)(items, splits.pool, splits.test, splits.paper600, splits.dev100))


def test_too_few_scoreboard_items_for_the_requested_slices_is_refused():
    candidates, scoreboard = _rows(scoreboard=50)
    with pytest.raises(ValueError, match="scoreboard"):
        build_emotion_splits(candidates, scoreboard, dev_size=20, final_size=60)


def test_the_stratified_order_gives_every_150_label_round_exactly_25_per_class():
    candidates, scoreboard = _rows(per_label=256)
    splits, _ = build_emotion_splits(candidates, scoreboard, dev_size=20, final_size=60)
    order = emotion_label_order(splits, 1)
    assert sorted(order) == sorted(splits.pool)
    for batch in round_batches(order, rounds=3, per_round=150):
        assert Counter(splits.items[i].reference_label for i in batch) == {label: 25 for label in EMOTION_LABELS}


def test_any_prefix_of_the_stratified_order_is_as_balanced_as_possible():
    candidates, scoreboard = _rows(per_label=256)
    splits, _ = build_emotion_splits(candidates, scoreboard, dev_size=20, final_size=60)
    order = emotion_label_order(splits, 2)
    for n in (1, 7, 50, 151, 449):
        counts = Counter(splits.items[i].reference_label for i in order[:n])
        assert max(counts.values()) - min(counts.get(label, 0) for label in EMOTION_LABELS) <= 1


def test_the_stratified_order_is_seeded_and_differs_between_seeds():
    splits, _ = _build(per_label=40)
    assert emotion_label_order(splits, 1) == emotion_label_order(splits, 1)
    assert emotion_label_order(splits, 1) != emotion_label_order(splits, 2)


def test_the_emotion_corpus_is_registered_with_its_labels_and_the_stratified_order():
    corpus = get_corpus("emotion")
    assert CORPORA["emotion"] is corpus and corpus.labels == EMOTION.labels == EMOTION_LABELS
    assert corpus.label_order_fn is emotion_label_order
    assert corpus.score_name == "Emotion"
    assert corpus.instructions.startswith("Choose exactly one emotion label.")
    assert tuple(corpus.task().labels) == EMOTION_LABELS


def test_the_planted_corpus_keeps_the_original_label_order_function():
    splits, _ = _build()
    assert PLANTED.label_order_fn(splits, 3) == label_order(splits.pool, 3)
    assert PLANTED.unsupported_reason is None


def test_the_emotion_corpus_refuses_runs_that_need_the_multi_class_features_not_built_yet():
    corpus = get_corpus("emotion")
    assert corpus.unsupported_reason
    with pytest.raises(NotImplementedError, match="emotion"):
        corpus.fewshot_feature
    with pytest.raises(NotImplementedError, match="emotion"):
        corpus.knn_features


def test_the_task_wording_is_the_one_the_static_emotion_study_froze():
    study_setup = pytest.importorskip("decision_flywheel_evaluations.study_setup")
    assert get_corpus("emotion").instructions == study_setup._TASKS[EMOTION.name][2]


_REAL = Path(EMOTION_REPO_ROOT) / ".data" / "huggingface" / "dair-ai___emotion"


@pytest.mark.skipif(not _REAL.is_dir(), reason="the local Emotion cache is absent")
def test_the_real_cached_data_loads_with_the_manifest_shape_and_no_leakage():
    pytest.importorskip("pyarrow")
    splits, report = load_emotion_splits()
    assert len(splits.pool) == 1536 and len(splits.test) == 2000
    assert Counter(splits.items[i].reference_label for i in splits.pool) == {l: 256 for l in EMOTION_LABELS}
    assert len(splits.dev100) == 100 and len(splits.paper600) == 600
    assert all(i.startswith("train-") for i in splits.pool) and all(i.startswith("test-") for i in splits.test)
    manifest = json.loads((Path(EMOTION_REPO_ROOT) / "studies" / "manifests" / "emotion.json").read_text())
    dev_ids = {r["id"] for r in manifest["records"] if r["role"] == "development"}
    assert not dev_ids & set(splits.items)       # the manifest's 600 development items are not used
    hashes = {r["id"]: r["normalized_text_sha256"] for r in manifest["records"]}
    assert all(normalized_text_hash(splits.items[i].text) == hashes[i] for i in splits.dev100 + splits.paper600)
    assert_no_duplicate_text(splits)
    assert report.duplicate_scoreboard_items_dropped >= 0
