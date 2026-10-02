"""Specs for frozen bundles, the final path through reloaded bundles, and labeler comments, offline.

They need the pinned Jev-Flywheel clone, scikit-learn and Tactus; run them with
``make unified-flywheel-test UF_PYTHON=/path/to/python``.
"""
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from . import unified_env

if not unified_env.jev_dependencies_available():
    pytest.skip("needs scikit-learn and Tactus (Jev-Flywheel's interpreter)", allow_module_level=True)
try:
    unified_env.ensure_decision_flywheel()
    CLONE = unified_env.put_clone_first()
except unified_env.EnvironmentProblem as problem:
    pytest.skip(f"pinned Jev-Flywheel clone unavailable: {problem}", allow_module_level=True)

unified_env.install_network_guard()

from decision_flywheel.bundle import EXAMPLES_FILE, MANIFEST_FILE  # noqa: E402
from decision_flywheel.models import Item as DFItem  # noqa: E402
from jev_flywheel.host import FlywheelHost  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

from .unified_bundles import BUNDLE_ARMS, ScoreRubric, bundle_dir  # noqa: E402
from .unified_fake_jev import FakeJevAsync, FakeJevCore, text_key  # noqa: E402
from .unified_labeler import (FAKE_MODEL, CommentCache, LabelerItem, fake_completion,  # noqa: E402
                              generate)
from .unified_loop import HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_spend import final_upper_bound  # noqa: E402
from .unified_splits import load_items  # noqa: E402

SMALL = dict(per_round=40, rounds=2, dev_size=30, bootstrap_resamples=50, max_concurrency=4)


class RecordingCore(FakeJevCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = []

    def answer(self, state, questions):
        self.targets.append(text_key(state["target"]["text"] if "target" in state else state["text"]))
        return super().answer(state, questions)


def _core():
    items = load_items(Path(CLONE.path) / "fixtures")
    return RecordingCore(planted={text_key(i.text): i.reference_label for i in items.values()}, strength=0.15)


def _labeler_comments(flywheel, root):
    labeled = [i for batch in flywheel.batches for i in batch]
    items = [LabelerItem(i, flywheel.splits.items[i].text, flywheel.labels[i]) for i in labeled]
    comments, _ = generate(items, fake_completion, model=FAKE_MODEL, cache=CommentCache(root / "labeler.jsonl"), seed=1)
    return comments


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    probe = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=root / "probe", **SMALL), fake_core=_core())
    comments = _labeler_comments(probe, root)
    cfg = RunConfig(clone=CLONE.path, run_dir=root / "run", arms=BUNDLE_ARMS, comments=comments, **SMALL)
    flywheel = UnifiedFlywheel(cfg, fake_core=_core())
    return flywheel, flywheel.run(), comments


@pytest.fixture(scope="module")
def final(frozen):
    flywheel, _, comments = frozen
    core = _core()
    cfg = RunConfig(clone=CLONE.path, run_dir=flywheel.run_dir, arms=BUNDLE_ARMS, comments=comments, final=True, **SMALL)
    final_run = UnifiedFlywheel(cfg, fake_core=core)
    return final_run, final_run.run(), core


def test_every_bundle_arm_is_frozen_at_the_end_of_the_last_round_with_a_text_free_manifest(frozen):
    flywheel, summary, _ = frozen
    assert set(summary["bundles"]) == set(BUNDLE_ARMS)
    for arm in BUNDLE_ARMS:
        path = bundle_dir(flywheel.run_dir, arm, 2)
        manifest = json.loads((path / MANIFEST_FILE).read_text())
        assert manifest["bundle_hash"] == summary["bundles"][arm]["bundle_hash"]
        assert manifest["lineage"]["round"] == 2 and manifest["lineage"]["arm"] == arm
        assert (path / EXAMPLES_FILE).exists() == (arm in ("F", "F-rand", "A-c+F"))
        blob = (path / MANIFEST_FILE).read_text()
        assert not any(len(item.text) >= 24 and item.text in blob for item in flywheel.splits.items.values())


def test_a_reloaded_bundle_predicts_exactly_like_the_in_memory_classifier(frozen):
    flywheel, _, _ = frozen
    for arm in BUNDLE_ARMS:
        bundle = flywheel.load_arm_bundle(arm)
        done = flywheel.classify_with_bundle(bundle, flywheel.eval_ids)
        assert done["requests"] == 0, "every dev-100 answer the in-memory classifier used is cached"
        in_memory = {r.item_id: (r.confidence, r.correct) for r in flywheel.results[2][arm]}
        reloaded = {i: (float(d.confidence), int(d.label == flywheel.splits.items[i].reference_label))
                    for i, d in done["results"].items()}
        assert reloaded == in_memory, arm


def test_a_reloaded_list_bundle_sends_the_same_request_as_the_in_memory_list(frozen):
    flywheel, _, _ = frozen
    for arm in ("F", "A-c+F"):
        bundle = flywheel.load_arm_bundle(arm)
        state = flywheel.results[2][arm]
        fresh = FakeJevAsync(_core())   # no cache at all: the answers come from the bundle's own request
        for item_id in flywheel.eval_ids[:10]:
            target = DFItem(item_id, {"text": flywheel.splits.items[item_id].text})
            decision = asyncio.run(bundle.classify(target, fresh))
            expected = next(r for r in state if r.item_id == item_id)
            assert float(decision.confidence) == expected.confidence


def test_the_final_run_scores_paper_600_only_through_the_reloaded_bundles(final, frozen):
    flywheel, _, _ = frozen
    final_run, summary, core = final
    paper = set(final_run.splits.paper600)
    assert summary["evaluation_slice"]["name"] == "paper-600" and summary["final"] is True
    assert json.loads((final_run.run_dir / "summary.json").read_text())["evaluation_slice"]["name"] == "dev-100"
    (entry,) = summary["per_round"]
    assert entry["scored_through"] == "reloaded bundles"
    for arm in BUNDLE_ARMS:
        assert entry["arms"][arm]["bundle"]["bundle_hash"] == flywheel.bundles[arm]["bundle_hash"]
        assert entry["arms"][arm]["metrics"]["n"] == 600 and entry["arms"][arm]["unscored"] == 0
    assert {text_key(final_run.splits.items[i].text) for i in paper} >= set(core.targets)
    assert "F-F-rand" in entry["contrasts"] and "A-c+F-max(A-c,F)" in entry["contrasts"]


def test_the_final_run_stays_within_its_upper_bound_and_arm_0_is_free(final):
    _, summary, core = final
    requests = summary["requests"]
    assert requests["new_this_invocation"] == core.calls
    assert core.calls <= final_upper_bound(BUNDLE_ARMS, eval_n=600)["total"]
    assert summary["per_round"][0]["arms"]["0"]["requests"] == 0   # Jev's paper-600 answers ship with the fixtures


def test_a_second_final_run_is_free_and_identical(final):
    final_run, first, _ = final
    core = _core()
    again = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=final_run.run_dir, arms=BUNDLE_ARMS, final=True,
                                      **SMALL), fake_core=core).run()
    assert core.calls == 0 and again["requests"]["new_this_invocation"] == 0
    def scored(summary):
        return {arm: (v["bundle"], v["metrics"]) for arm, v in summary["per_round"][0]["arms"].items()}
    assert scored(again) == scored(first)
    assert again["per_round"][0]["contrasts"] == first["per_round"][0]["contrasts"]


def test_the_final_run_refuses_arms_it_cannot_bundle_and_missing_bundles(tmp_path):
    with pytest.raises(HarnessError):
        RunConfig(clone=CLONE.path, run_dir=tmp_path / "x", arms=("0", "B"), final=True, **SMALL).validate()
    run = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "empty", arms=("F",), final=True, **SMALL),
                          fake_core=_core())
    with pytest.raises(HarnessError):
        run.run()


def test_labeler_comments_reach_the_analyst_briefing_of_the_comment_arms_only(frozen):
    flywheel, summary, comments = frozen
    for arm, expect in (("A", False), ("A-c", True), ("A-c+F", True)):
        workspace = Workspace(flywheel.run_dir / "workspaces" / arm.replace("+", "plus"))
        briefing = FlywheelHost(workspace, "Sentiment").briefing()
        seen = [m["human_comment"] for m in briefing["mismatches"] + briefing["commented_agreements"]
                if m.get("human_comment")]
        assert bool(seen) == expect, arm
        assert set(seen) <= set(comments.values())
    rounds = summary["per_round"]
    assert all(r["arms"]["A"]["steering"]["comments_in_feedback"] == 0 for r in rounds)
    assert rounds[-1]["arms"]["A-c"]["steering"]["comments_in_feedback"] == sum(
        i in comments for batch in flywheel.batches for i in batch)


def test_no_output_carries_dataset_or_comment_text(frozen, final):
    flywheel, _, comments = frozen
    final_run, _, _ = final
    names = ("summary.json", "run-log.json", "predictions.jsonl")
    for name in names + tuple("final-" + n for n in names):
        blob = (flywheel.run_dir / name).read_text()
        assert not any(c in blob for c in comments.values() if len(c) >= 12)
        for item_id in list(flywheel.batches[0])[:20] + list(final_run.eval_ids)[:20]:
            assert flywheel.splits.items[item_id].text not in blob


def test_a_rerun_that_changes_a_bundle_keeps_the_old_one_beside_it(frozen, tmp_path):
    flywheel, summary, _ = frozen
    path = bundle_dir(flywheel.run_dir, "F", 2)
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    copy = tmp_path / "F"
    import shutil
    shutil.copytree(path, copy)
    stale = {**manifest, "bundle_hash": "0" * 64}
    (path / MANIFEST_FILE).write_text(json.dumps(stale))
    try:
        cfg = RunConfig(clone=CLONE.path, run_dir=flywheel.run_dir, arms=("F",), **SMALL)
        UnifiedFlywheel(cfg, fake_core=_core()).run()
        assert (path.parent / "round-2.superseded-000000000000" / MANIFEST_FILE).exists()
        assert json.loads((path / MANIFEST_FILE).read_text())["bundle_hash"] == manifest["bundle_hash"]
    finally:
        shutil.rmtree(path.parent / "round-2.superseded-000000000000", ignore_errors=True)


# ---- ScoreRubric.predict: real per-class probabilities for N > 2, the exact binary numbers for N = 2 ----

SIX = ("anger", "fear", "joy", "love", "sadness", "surprise")


def _rubric(labels, *, calibration=None):
    """A frozen head over the few-shot ``fewshot.clr.<label>`` features of every label but the last."""
    kept = labels[:-1]
    weights = {label: {"intercept": 0.1 * i, **{f"fewshot.clr.{k}": (1.5 if k == label else -0.2) for k in kept}}
               for i, label in enumerate(labels) if label != labels[-1]} if len(labels) > 2 else \
        {labels[1]: {"intercept": 0.2, f"fewshot.clr.{labels[0]}": -1.0}}
    decision = {"model": "multinomial_logistic", "classes": list(labels), "features": [f"fewshot.clr.{k}" for k in kept],
                "parameters": {"weights": weights}}
    if calibration:
        decision["calibration"] = calibration
    config = {"name": "Feeling", "elements": [{"key": "fewshot", "question_type": "choice", "instructions": "Which?",
                                               "criteria": {label: None for label in labels}}], "decision": decision}
    return ScoreRubric.parse(yaml.safe_dump(config))


def _answer(labels, favoured, p=0.6):
    rest = (1 - p) / (len(labels) - 1)
    probabilities = {label: (p if label == favoured else rest) for label in labels}
    return {"feeling.fewshot": {"type": "choice", "choice": favoured, "confidence": p, "probabilities": probabilities}}


def test_for_six_labels_predict_returns_the_heads_real_probabilities_which_sum_to_one():
    from jev_flywheel.scoring import predict as head_predict
    rubric = _rubric(SIX)
    answers = _answer(SIX, "joy")
    result = rubric.predict(answers)
    expected = head_predict(rubric.score, {}, features=rubric.score.feature_vector(answers))
    assert result.label == expected.value
    assert set(result.probabilities) == set(SIX)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities == pytest.approx(expected.metadata["decision"]["probabilities"])
    assert result.confidence == pytest.approx(result.probabilities[result.label])
    assert len({round(v, 6) for k, v in result.probabilities.items() if k != result.label}) > 1, \
        "the non-top classes are not spread evenly"


def test_for_six_labels_a_calibrated_confidence_rescales_the_rest_so_the_total_stays_one():
    calibration = {"method": "temperature", "raw_confidence": [0.0, 1.0], "calibrated_confidence": [0.0, 0.5]}
    result = _rubric(SIX, calibration=calibration).predict(_answer(SIX, "fear"))
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities[result.label] == pytest.approx(result.confidence)
    assert max(result.probabilities, key=result.probabilities.get) == result.label


def test_for_two_labels_predict_keeps_the_exact_even_split_of_the_remaining_confidence():
    labels = ("positive", "negative")
    rubric = _rubric(labels)
    result = rubric.predict(_answer(labels, "positive"))
    confidence = min(max(float(result.confidence), 0.0), 1.0)
    other = [c for c in labels if c != result.label][0]
    assert result.probabilities == {result.label: confidence, other: (1.0 - confidence) / 1}
