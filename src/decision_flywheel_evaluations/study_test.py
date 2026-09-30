import pytest

from .study import Engine, SplitManifest, StudyPlan


def test_overlapping_splits_are_rejected_before_a_model_can_be_called():
    manifest = SplitManifest("demo", "abc", frozenset({"a"}), frozenset({"b"}), frozenset({"a", "c"}))
    with pytest.raises(ValueError, match="disjoint"):
        manifest.validate()


def test_a_complete_plan_accepts_the_initial_engine_matrix():
    manifest = SplitManifest("demo", "abc", frozenset({"a"}), frozenset({"b"}), frozenset({"c"}))
    plan = StudyPlan("matrix", manifest, (Engine.JEV, Engine.KEV, Engine.LAYA), ("random", "retrieval"), 96, "macro_f1", 1000, "task-sha", {Engine.LAYA: "context contract pending"})
    plan.validate()
