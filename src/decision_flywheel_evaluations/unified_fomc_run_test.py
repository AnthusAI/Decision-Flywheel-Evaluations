"""Specs for an offline FOMC run of the unified loop: synthetic FOMC-shaped rows, a fake Jev, a fake stakeholder.

Like ``unified_emotion_run_test`` they need the pinned clone, scikit-learn and Tactus. They cover what the
rubric experiment adds around the loop: explanation arms receive the in-loop stakeholder's comments for
their own errors, the labels-only arm receives none, the shuffled control never keeps a comment on its
source item, arm 0 asks the seed question itself, and CEIL is a fixed reference that is never refit.
"""
import json
from dataclasses import replace

import pytest

from . import unified_env

if not unified_env.jev_dependencies_available():
    pytest.skip("needs scikit-learn and Tactus (Jev-Flywheel's interpreter)", allow_module_level=True)
try:
    unified_env.ensure_decision_flywheel()
    CLONE = unified_env.put_clone_first()
except unified_env.EnvironmentProblem as problem:
    pytest.skip(f"pinned Jev-Flywheel clone unavailable: {problem}", allow_module_level=True)

unified_env.install_network_guard()

from jev_flywheel.workspace import Workspace  # noqa: E402

from .unified_corpus import FOMC_CORPUS  # noqa: E402
from .unified_fomc import FOMC_LABELS, build_fomc_splits, guideline_text  # noqa: E402
from .unified_labeler import (STAKEHOLDER_FAKE_MODEL, CommentCache, RubricStakeholder,  # noqa: E402
                              fake_stakeholder_completion)
from .unified_loop import HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402

ARMS = ("0", "A", "A-c", "A-c-shuffled", "A-c-noisy", "F", "A-c+F", "CEIL")
SMALL = dict(per_round=60, rounds=2, dev_size=12, bootstrap_resamples=40, max_concurrency=4)
WORDS = {"dovish": "inflation decreases and unemployment increases", "hawkish": "inflation increases and house prices increase",
         "neutral": "the committee reaffirmed a moderate and mixed view"}


def _splits():
    train = [(f"fomc-train-{n}", FOMC_LABELS[n % 3], f"pool sentence {n}: {WORDS[FOMC_LABELS[(n * 5) % 3]]}") for n in range(240)]
    test = [(f"fomc-test-{n}", FOMC_LABELS[n % 3], f"test sentence {n}: {WORDS[FOMC_LABELS[(n * 2) % 3]]}") for n in range(40)]
    return build_fomc_splits(train, test, dev_size=12)[0]


def tagged_completion(prompt):
    """The fake stakeholder, with the source item's id appended to the rule (still valid), to trace shuffling."""
    answer = fake_stakeholder_completion(prompt)
    if not answer.startswith("{"):
        return answer
    raw = json.loads(answer)
    raw["explanation"] += f" (source {prompt.case.item_id})"
    return json.dumps(raw)


def _run(root, **overrides):
    splits = _splits()
    corpus = replace(FOMC_CORPUS, load_splits=lambda fixtures, dev_size: splits, final_size=len(splits.paper600))
    stakeholder = RubricStakeholder(guideline_text(), FOMC_LABELS, tagged_completion, model=STAKEHOLDER_FAKE_MODEL,
                                    cache=CommentCache(root / "stakeholder-cache.jsonl"), seed=1)
    cfg = RunConfig(clone=CLONE.path, run_dir=root, corpus=corpus, stakeholder=stakeholder,
                    **{**SMALL, "arms": ARMS, **overrides})
    flywheel = UnifiedFlywheel(cfg)
    return flywheel, flywheel.run(), splits


@pytest.fixture(scope="module")
def fomc_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("fomc") / "run")


def _feedback_comments(flywheel, arm):
    workspace = Workspace(flywheel.run_dir / "workspaces" / arm.replace("+", "plus"))
    return {fb.item_id: fb.edit_comment_value for fb in workspace.feedback() if fb.edit_comment_value}


def test_an_offline_fomc_run_completes_every_arm_and_freezes_every_bundle(fomc_run):
    _, summary, _ = fomc_run
    assert set(summary["arms"]) == set(ARMS) and len(summary["per_round"]) == 2
    assert set(summary["bundles"]) == set(ARMS)
    assert all("macro_f1" in entry["multiclass"] for r in summary["per_round"] for entry in r["arms"].values())


def test_explanation_arms_receive_stakeholder_comments_and_the_labels_only_arm_none(fomc_run):
    flywheel, summary, _ = fomc_run
    assert _feedback_comments(flywheel, "A") == {}
    for arm in ("A-c", "A-c-shuffled", "A-c-noisy", "A-c+F"):
        assert _feedback_comments(flywheel, arm), arm
        assert summary["per_round"][0]["arms"][arm]["stakeholder"]["forwarded"] > 0
    assert "stakeholder" not in summary["per_round"][0]["arms"]["A"]


def test_comments_are_only_on_misclassified_pool_items_and_at_most_15_per_round(fomc_run):
    flywheel, summary, splits = fomc_run
    for arm in ("A-c", "A-c+F"):
        assert set(_feedback_comments(flywheel, arm)) <= set(splits.pool)
        for round_ in summary["per_round"]:
            record = round_["arms"][arm]["stakeholder"]
            assert record["new_calls"] + record["cache_hits"] <= 15
    workspace = Workspace(flywheel.run_dir / "workspaces" / "A-c")
    assert all(not fb.is_agreement for fb in workspace.feedback() if fb.edit_comment_value)


def test_a_shuffled_comment_never_stays_on_its_source_item(fomc_run):
    flywheel, _, _ = fomc_run
    shuffled = _feedback_comments(flywheel, "A-c-shuffled")
    assert shuffled and all(f"(source {item_id})" not in comment for item_id, comment in shuffled.items())
    plain = _feedback_comments(flywheel, "A-c")
    assert all(f"(source {item_id})" in comment for item_id, comment in plain.items())


def test_the_noisy_control_replaces_about_a_fifth_of_its_comments(fomc_run):
    _, summary, _ = fomc_run
    record = summary["per_round"][0]["arms"]["A-c-noisy"]["stakeholder"]
    assert record["variant"] == "noisy" and record["noisy_replaced"] == round(0.2 * (record["accepted"] - record["repeat_capped"]))


def test_arm_0_asks_the_seed_question_and_ceil_is_a_fixed_reference_never_refit(fomc_run):
    flywheel, summary, _ = fomc_run
    steps = {(e["arm"], e["step"]) for e in flywheel.events}
    assert ("0", "fill-seed-question") in steps and ("CEIL", "fill-seed-question") in steps
    for round_ in summary["per_round"]:
        assert round_["arms"]["CEIL"]["gate"]["decision"] == "fixed reference (not refit)"
    assert summary["policies"]["comments"]["model"] == STAKEHOLDER_FAKE_MODEL


def test_the_summary_and_the_stakeholder_cache_hold_no_dataset_text(fomc_run):
    flywheel, summary, splits = fomc_run
    blobs = [json.dumps(summary), (flywheel.run_dir / "stakeholder-cache.jsonl").read_text()]
    assert not any(splits.items[i].text in blob for i in splits.items for blob in blobs)


def test_the_final_scores_every_frozen_bundle_on_the_final_slice(fomc_run):
    flywheel, _, splits = fomc_run
    corpus = flywheel.corpus
    cfg = RunConfig(clone=CLONE.path, run_dir=flywheel.run_dir, corpus=corpus, final=True, **{**SMALL, "arms": ARMS})
    summary = UnifiedFlywheel(cfg).run()
    assert summary["evaluation_slice"]["name"] == f"final-{len(splits.paper600)}"
    assert set(summary["per_round"][0]["arms"]) == set(ARMS)
    assert all(entry["unscored"] == 0 for entry in summary["per_round"][0]["arms"].values())


def test_the_explanation_controls_refuse_to_run_without_the_stakeholder(tmp_path):
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / "x", corpus=FOMC_CORPUS, arms=("A-c-shuffled",), **SMALL)
    with pytest.raises(HarnessError, match="stakeholder"):
        UnifiedFlywheel(cfg)
