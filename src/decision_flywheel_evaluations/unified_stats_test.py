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
