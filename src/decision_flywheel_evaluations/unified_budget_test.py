"""Specs for the feature-budget cap: pure planning, no Jev, no network."""
from .unified_budget import FeatureCapPlan, plan_feature_cap

HOLISTIC = ["self.holistic.clr.anger", "self.holistic.clr.fear"]
TWO = ["topic.clr.work", "topic.clr.home", "hedged.logit_p"]


def _plan(features, keys, budget, **kwargs):
    return plan_feature_cap(features, keys, budget=budget, protected=("self",), **kwargs)


def test_nothing_is_dropped_when_the_features_fit_the_budget():
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=5)
    assert plan == FeatureCapPlan(dropped=(), features_before=5, features_after=5, reserved=0, satisfied=True)
    assert not plan.dropped and plan.record(n_labeled=30, budget=5) is None


def test_the_lowest_ranked_newest_element_is_dropped_first_until_the_head_fits():
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=4)
    assert [d.key for d in plan.dropped] == ["hedged"] and plan.features_after == 4 and plan.satisfied
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=2)
    assert [d.key for d in plan.dropped] == ["hedged", "topic"] and plan.features_after == 2


def test_the_holistic_features_are_never_dropped_even_when_they_alone_exceed_the_budget():
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=1)
    assert [d.key for d in plan.dropped] == ["hedged", "topic"]
    assert plan.features_after == 2 and not plan.satisfied


def test_features_reserved_for_the_arm_count_against_the_budget():
    # the few-shot arm will add 5 features after steering
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=9, reserved=5)
    assert [d.key for d in plan.dropped] == ["hedged"] and plan.features_after == 4 and plan.reserved == 5


def test_an_element_is_dropped_whole_with_its_feature_count_and_new_flag():
    plan = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=3, new_keys=("hedged",))
    assert [(d.key, d.n_features, d.new_this_round) for d in plan.dropped] == [
        ("hedged", 1, True), ("topic", 2, False)]


def test_the_plan_is_deterministic_and_the_record_is_text_free_keys_and_counts_only():
    first = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=4, new_keys=("hedged",))
    assert first == _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=4, new_keys=("hedged",))
    assert first.record(n_labeled=30, budget=4) == {
        "n_labeled": 30, "budget": 4, "reserved_features": 0, "features_before": 5, "features_after": 4,
        "reason": "over the capability-ladder feature budget",
        "dropped": [{"key": "hedged", "features": 1, "new_this_round": True}]}


def test_an_unresolvable_overrun_is_flagged_in_the_record():
    record = _plan(HOLISTIC, [], budget=1).record(n_labeled=30, budget=1)
    assert record is None  # nothing droppable and nothing dropped: behave as today (the fit refuses)
    record = _plan(HOLISTIC + TWO, ["topic", "hedged"], budget=1).record(n_labeled=30, budget=1)
    assert record["still_over_budget"] is True
