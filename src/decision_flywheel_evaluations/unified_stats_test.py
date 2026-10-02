import pytest

from .unified_stats import ItemResult, contrasts, ece, metric, paired_interval, summarize


def _rows(spec):
    return [ItemResult(f"i{n}", conf, ok) for n, (conf, ok) in enumerate(spec)]


def test_ece_uses_ten_equal_width_bins_weighted_by_mass():
    rows = _rows([(0.95, 1), (0.95, 0), (0.55, 1), (0.55, 1)])
    # bin 9: mean 0.95, acc 0.5 -> 0.45; bin 5: mean 0.55, acc 1.0 -> 0.45
    assert ece([r.confidence for r in rows], [r.correct for r in rows]) == pytest.approx(0.45)
    assert summarize(rows)["accuracy"] == pytest.approx(0.75)


def test_an_arm_compared_with_itself_has_a_zero_effect_and_a_zero_width_interval():
    rows = _rows([(0.9, 1), (0.6, 0), (0.7, 1)])
    out = paired_interval(rows, rows, "brier", resamples=50)
    assert out == {"effect": 0.0, "lower": 0.0, "upper": 0.0}


def test_paired_intervals_are_deterministic_for_a_seed_and_bracket_the_effect():
    base = _rows([(0.6, 1), (0.6, 0), (0.6, 1), (0.6, 0)] * 10)
    better = _rows([(0.9, 1), (0.6, 0), (0.9, 1), (0.6, 1)] * 10)
    first = paired_interval(base, better, "accuracy", resamples=200, seed=3)
    assert first == paired_interval(base, better, "accuracy", resamples=200, seed=3)
    assert first["lower"] <= first["effect"] <= first["upper"]
    assert first["effect"] == pytest.approx(0.25)


def test_arms_scored_on_different_items_cannot_be_paired():
    with pytest.raises(ValueError):
        paired_interval(_rows([(0.5, 1)]), [ItemResult("other", 0.5, 1)], "accuracy")


def test_the_combined_arm_is_compared_with_the_better_single_lever_per_metric():
    results = {
        "0": _rows([(0.6, 1), (0.6, 0)] * 5),
        "A": _rows([(0.9, 1), (0.6, 0)] * 5),
        "B": _rows([(0.6, 1), (0.6, 1)] * 5),
        "B-local": _rows([(0.6, 1), (0.6, 0)] * 5),
        "A+B": _rows([(0.9, 1), (0.9, 1)] * 5),
    }
    out = contrasts(results, resamples=20)
    assert set(out) == {"A-0", "B-0", "B-B-local", "A+B-max(A,B)"}
    assert out["A+B-max(A,B)"]["accuracy"]["baseline_arm"] == "B"
    assert metric("accuracy", results["B"]) == 1.0


def test_only_contrasts_whose_arms_ran_are_reported():
    rows = _rows([(0.6, 1), (0.7, 0)])
    assert set(contrasts({"A": rows, "B": rows}, resamples=5)) == set()
    assert set(contrasts({"0": rows, "B": rows}, resamples=5)) == {"B-0"}


# ---- multi-class metrics (Emotion): macro-F1, per-class recall/precision/F1, confusion matrix -------------------

from .unified_stats import (LOW_SUPPORT_THRESHOLD, confusion_matrix, macro_f1,  # noqa: E402
                            multiclass_contrasts, multiclass_summary, per_class)

LABELS = ("a", "b", "c")


def _labeled(pairs, labels=LABELS):
    """(gold, predicted) pairs as rows; confidence is fixed because the multi-class metrics ignore it."""
    return [ItemResult(f"i{n}", 0.7, int(gold == predicted), predicted=predicted, gold=gold)
            for n, (gold, predicted) in enumerate(pairs)]


# a: tp 2, fp 1, fn 1 (P=R=F1=2/3); b: tp 1, fp 1, fn 1 (P=R=F1=1/2); c: tp 1 (P=R=F1=1)
HAND = [("a", "a"), ("a", "a"), ("a", "b"), ("b", "b"), ("b", "a"), ("c", "c")]


def test_macro_f1_and_per_class_scores_match_a_hand_computed_case():
    rows = _labeled(HAND)
    scores = per_class(rows, LABELS)
    assert scores["a"]["precision"] == pytest.approx(2 / 3) and scores["a"]["recall"] == pytest.approx(2 / 3)
    assert scores["b"]["f1"] == pytest.approx(0.5) and scores["c"]["f1"] == pytest.approx(1.0)
    assert (scores["a"]["support"], scores["a"]["predicted"]) == (3, 3)
    assert macro_f1(rows, LABELS) == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert multiclass_summary(rows, LABELS)["accuracy"] == pytest.approx(4 / 6)


def test_the_confusion_matrix_is_n_by_n_in_label_order_with_gold_on_the_rows():
    matrix = confusion_matrix(_labeled(HAND), LABELS)
    assert matrix == {"labels": ["a", "b", "c"], "rows": "gold", "columns": "predicted",
                      "counts": [[2, 1, 0], [1, 1, 0], [0, 0, 1]]}
    reordered = confusion_matrix(_labeled(HAND), ("c", "b", "a"))
    assert reordered["counts"] == [[1, 0, 0], [0, 1, 1], [0, 1, 2]]


def test_a_class_absent_from_the_slice_is_undefined_and_left_out_of_the_macro_average():
    rows = _labeled(HAND)
    four = ("a", "b", "c", "d")
    scores = per_class(rows, four)
    assert scores["d"] == {"support": 0, "predicted": 0, "recall": None, "precision": None, "f1": None,
                           "low_support": True}
    assert macro_f1(rows, four) == pytest.approx(macro_f1(rows, LABELS))   # d neither helps nor hurts
    summary = multiclass_summary(rows, four)
    assert summary["absent_classes"] == ["d"]
    assert summary["confusion_matrix"]["counts"][3] == [0, 0, 0, 0]


def test_a_class_nobody_has_but_something_predicted_counts_as_a_zero_f1_class():
    rows = _labeled(HAND + [("a", "d")])
    four = ("a", "b", "c", "d")
    scores = per_class(rows, four)
    assert scores["d"]["recall"] is None and scores["d"]["precision"] == 0.0 and scores["d"]["f1"] == 0.0
    assert macro_f1(rows, four) < macro_f1(_labeled(HAND), four)
    assert multiclass_summary(rows, four)["absent_classes"] == []   # d is predicted, so it is not simply absent


def test_a_class_that_is_never_predicted_has_zero_recall_and_f1_and_an_undefined_precision():
    scores = per_class(_labeled([("a", "a"), ("b", "a"), ("b", "a")]), ("a", "b"))
    assert scores["b"] == {"support": 2, "predicted": 0, "recall": 0.0, "precision": None, "f1": 0.0,
                           "low_support": True}


def test_per_class_numbers_from_fewer_than_ten_items_are_flagged_not_hidden():
    assert LOW_SUPPORT_THRESHOLD == 10
    rows = _labeled([("a", "a")] * 10 + [("b", "b")] * 9)
    scores = per_class(rows, ("a", "b"))
    assert scores["a"]["low_support"] is False and scores["b"]["low_support"] is True
    assert scores["b"]["recall"] == 1.0                           # still reported
    summary = multiclass_summary(rows, ("a", "b"))
    assert summary["low_support_classes"] == ["b"] and summary["low_support_threshold"] == 10


def test_rows_without_a_predicted_or_gold_label_or_with_a_foreign_label_cannot_be_scored():
    with pytest.raises(ValueError, match="predicted"):
        macro_f1(_rows([(0.5, 1)]), LABELS)
    with pytest.raises(ValueError, match="not one of"):
        macro_f1(_labeled([("a", "zzz")]), LABELS)
    with pytest.raises(ValueError, match="labels"):
        metric("macro_f1", _labeled(HAND))


def test_the_macro_f1_bootstrap_is_seed_stable_brackets_the_effect_and_is_zero_for_an_arm_against_itself():
    base = _labeled(HAND * 8)
    better = _labeled([("a", "a"), ("a", "a"), ("a", "a"), ("b", "b"), ("b", "a"), ("c", "c")] * 8)
    first = paired_interval(base, better, "macro_f1", resamples=200, seed=4, labels=LABELS)
    assert first == paired_interval(base, better, "macro_f1", resamples=200, seed=4, labels=LABELS)
    assert first["lower"] <= first["effect"] <= first["upper"]
    assert first["effect"] == pytest.approx(macro_f1(better, LABELS) - macro_f1(base, LABELS), abs=1e-6)
    assert paired_interval(base, base, "macro_f1", resamples=50, labels=LABELS) == {"effect": 0.0, "lower": 0.0, "upper": 0.0}


def test_the_macro_f1_bootstrap_uses_the_same_percentile_convention_and_draws_as_the_other_metrics():
    import random

    base = _labeled(HAND * 5)
    treat = _labeled(([("a", "a")] + HAND[1:]) * 5)
    out = paired_interval(base, treat, "macro_f1", resamples=101, seed=9, labels=LABELS)
    rng, effects, ordered = random.Random(9), [], sorted(base, key=lambda r: r.item_id)
    other = {r.item_id: r for r in treat}
    paired = [other[r.item_id] for r in ordered]
    for _ in range(101):
        sample = [rng.randrange(len(ordered)) for _ in ordered]
        effects.append(macro_f1([paired[i] for i in sample], LABELS) - macro_f1([ordered[i] for i in sample], LABELS))
    effects.sort()
    assert out["lower"] == round(effects[int(0.025 * 100)], 6) and out["upper"] == round(effects[int(0.975 * 100)], 6)


def test_multiclass_contrasts_report_macro_f1_and_accuracy_for_whichever_arms_ran():
    zero = _labeled(HAND * 4)
    results = {"0": zero,
               "A-c": _labeled([("a", "a"), ("a", "a"), ("a", "a"), ("b", "b"), ("b", "a"), ("c", "c")] * 4),
               "F": _labeled([("a", "a"), ("a", "b"), ("a", "a"), ("b", "b"), ("b", "b"), ("c", "c")] * 4),
               "F-rand": zero,
               "A-c+F": _labeled([("a", "a"), ("a", "a"), ("a", "a"), ("b", "b"), ("b", "b"), ("c", "c")] * 4)}
    out = multiclass_contrasts(results, LABELS, resamples=30)
    assert set(out) == {"A-c-0", "F-0", "F-rand-0", "F-F-rand", "A-c+F-0", "A-c+F-max(A-c,F)"}
    assert all(set(entry) == {"macro_f1", "accuracy"} for entry in out.values())
    assert out["A-c+F-max(A-c,F)"]["macro_f1"]["baseline_arm"] in ("A-c", "F")
    assert out["F-0"]["accuracy"]["effect"] == pytest.approx(metric("accuracy", results["F"]) - metric("accuracy", zero), abs=1e-6)
    assert set(multiclass_contrasts({"A-c": zero, "F": zero}, LABELS, resamples=5)) == set()


def test_the_legacy_metrics_are_unchanged_top_label_brier_and_ece_on_any_number_of_classes():
    rows = _labeled(HAND)   # confidence 0.7 is the probability of the PREDICTED label, whatever the class count
    out = summarize(rows)
    assert set(out) == {"n", "accuracy", "brier", "ece"}
    assert out["brier"] == pytest.approx((4 * 0.09 + 2 * 0.49) / 6, abs=1e-6)
