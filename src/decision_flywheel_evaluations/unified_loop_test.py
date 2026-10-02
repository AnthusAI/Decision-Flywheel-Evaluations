"""Specs for the unified loop against a fake Jev, entirely offline.

They need the pinned Jev-Flywheel clone, scikit-learn and Tactus, so they skip in an
interpreter without them (this repository's ``.venv`` has neither). Run them with
``make unified-flywheel-test UF_PYTHON=/path/to/python``.
"""
import json
import subprocess
from pathlib import Path

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

from jev_flywheel.answers import AnswerCache  # noqa: E402
from jev_flywheel.evaluate import summarize as jev_summarize  # noqa: E402
from jev_flywheel.fit import build_training_set, fit_head  # noqa: E402
from jev_flywheel.items import FeedbackItem  # noqa: E402
from jev_flywheel.scorecard import Scorecard  # noqa: E402

from .unified_fake_jev import FakeJevCore, text_key  # noqa: E402
from .unified_loop import (  # noqa: E402
    FEWSHOT_FEATURE, FEWSHOT_WIRE, HarnessError, RunConfig, UnifiedFlywheel, candidate_template,
    feature_row, training_set, uses_fewshot)
from .unified_knn import KNN_FEATURES  # noqa: E402
from .unified_splits import load_items  # noqa: E402
from .unified_stats import ItemResult, summarize  # noqa: E402

SMALL = dict(per_round=40, rounds=2, dev_size=30, bootstrap_resamples=50, max_concurrency=4)


class RecordingCore(FakeJevCore):
    """The fake Jev, remembering the hash of every text it was shown and in what role."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets, self.examples, self.pairs = [], [], []

    def answer(self, state, questions):
        target = state["target"]["text"] if "target" in state else state["text"]
        shown = [text_key(e["text"]) for e in state.get("labeled_examples") or []]
        self.targets.append(text_key(target))
        self.examples.extend(shown)
        self.pairs.append((text_key(target), tuple(shown)))
        return super().answer(state, questions)


def _run(tmp_path, name, **overrides):
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / name, **{**SMALL, **overrides})
    items = load_items(Path(CLONE.path) / "fixtures")
    core = RecordingCore(planted={text_key(i.text): i.reference_label for i in items.values()}, strength=0.15)
    flywheel = UnifiedFlywheel(cfg, fake_core=core)
    return flywheel, flywheel.run(), core


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("unified"), "run")


def test_injected_knn_and_fewshot_features_pass_fit_head_unchanged(small_run):
    flywheel, _, _ = small_run
    labeled = list(flywheel.batches[0]) + list(flywheel.batches[1])
    base = flywheel.v1.score("Sentiment")
    template = candidate_template(base, fewshot=True, knn=True)
    assert uses_fewshot(template)
    rows = flywheel._rows(template, labeled, labeled)
    training = training_set(template, labeled, flywheel.labels, rows, "spec")
    assert training.n == len(labeled) and not training.needs_answers
    result = fit_head(training, template, seed=0)
    assert result.fitted
    weights = next(iter(result.head["weights"].values()))
    for name in (FEWSHOT_FEATURE, *KNN_FEATURES):
        assert name in weights and name in result.provenance["features"]
    # And nothing in the pinned clone was edited to make that work.
    status = subprocess.run(["git", "-C", CLONE.path, "status", "--porcelain", "--untracked-files=no"],
                            check=True, stdout=subprocess.PIPE).stdout.decode()
    assert status == ""


def test_the_harness_builds_the_same_rows_as_jev_flywheel_for_zero_shot_features(small_run):
    flywheel, _, _ = small_run
    labeled = list(flywheel.batches[0])
    reference = Scorecard.from_yaml((flywheel.fixtures / "scorecards" / "reference_full.yaml").read_text())
    score = flywheel.v1.score("Sentiment")
    feedback = [FeedbackItem(id=f"f{i}", item_id=i, score_name="Sentiment", final_answer_value=flywheel.labels[i],
                             metadata={"propensity": 1.0}) for i in labeled]
    theirs = build_training_set(score, reference.questions(), flywheel.cache, feedback)
    ours = training_set(score, labeled, flywheel.labels, flywheel._rows(score, labeled, labeled), "x")
    assert ours.rows == theirs.rows and ours.labels == theirs.labels and ours.weights == theirs.weights


def test_fewshot_answers_are_cached_under_a_context_fingerprint_of_policy_and_labeled_ids(small_run):
    flywheel, _, _ = small_run
    first = list(flywheel.batches[0])
    both = first + list(flywheel.batches[1])
    target = flywheel.eval_ids[0]
    small, large = flywheel._fewshot_key(flywheel._fewshot_context(first)), flywheel._fewshot_key(flywheel._fewshot_context(both))
    assert flywheel._fewshot_context(first) != flywheel._fewshot_context(both)
    assert flywheel._fewshot_context(both) == flywheel._fewshot_context(list(reversed(both)))
    assert flywheel.fewshot_cache.get(target, FEWSHOT_WIRE, small) is not None
    assert flywheel.fewshot_cache.get(target, FEWSHOT_WIRE, large) is not None
    reloaded = AnswerCache(flywheel.run_dir / "fewshot-answers.jsonl")
    assert reloaded.get(target, FEWSHOT_WIRE, small) == flywheel.fewshot_cache.get(target, FEWSHOT_WIRE, small)
    # A zero-shot lookup of the same element wording never sees a few-shot answer.
    plain = {k: v for k, v in small.items() if k != "context_fingerprint"}
    assert flywheel.fewshot_cache.get(target, FEWSHOT_WIRE, plain) is None


def test_leakage_rule_labeled_items_are_never_held_out_items(small_run):
    flywheel, summary, core = small_run
    labeled = {i for batch in flywheel.batches for i in batch}
    assert labeled <= set(flywheel.splits.pool)
    assert not labeled & set(flywheel.splits.test)
    for arm_dir in ("A", "Aplus" + "B"):
        assert {f.item_id for f in _feedback(flywheel.run_dir / "workspaces" / arm_dir)} == labeled


def _feedback(root):
    return [FeedbackItem(**json.loads(line)) for line in (root / "feedback.jsonl").read_text().splitlines()]


def test_leakage_rule_a_target_is_never_shown_as_its_own_example(small_run):
    _, _, core = small_run
    few_shot = [(target, shown) for target, shown in core.pairs if shown]
    assert few_shot, "the fake never received a few-shot request"
    assert all(target not in shown for target, shown in few_shot)


def test_leakage_rule_retrieval_pools_never_contain_held_out_items(small_run):
    flywheel, _, core = small_run
    held_out = {text_key(flywheel.splits.items[i].text) for i in flywheel.splits.test}
    labeled = {text_key(flywheel.splits.items[i].text) for batch in flywheel.batches for i in batch}
    assert set(core.examples) <= labeled
    assert not set(core.examples) & held_out


def test_without_final_paper_600_is_never_requested_or_scored(small_run):
    flywheel, summary, core = small_run
    paper = {text_key(flywheel.splits.items[i].text) for i in flywheel.splits.paper600}
    assert not paper & set(core.targets) and not paper & set(core.examples)
    scored = {json.loads(line)["item_id"] for line in (flywheel.run_dir / "predictions.jsonl").read_text().splitlines()}
    assert scored == set(flywheel.splits.dev100) == set(flywheel.eval_ids)
    assert summary["evaluation_slice"]["name"] == "dev-100"


def test_every_request_is_counted_per_arm_and_round_and_free_arms_spend_nothing(small_run):
    flywheel, summary, core = small_run
    requests = summary["requests"]
    assert requests["new_this_invocation"] == core.calls
    assert "0" not in requests["by_arm"] and "B-local" not in requests["by_arm"]
    assert requests["by_arm"]["B"]["few-shot"] == sum(len(flywheel.batches[0]) * r + len(flywheel.eval_ids) for r in (1, 2))
    # A+B runs after B with the same pool and policy, so its few-shot answers are cache hits.
    assert "few-shot" not in requests.get("by_arm", {}).get("A+B", {})


def test_outputs_contain_no_dataset_text(small_run):
    flywheel, _, _ = small_run
    for name in ("summary.json", "run-log.json", "predictions.jsonl"):
        blob = (flywheel.run_dir / name).read_text()
        for item_id in list(flywheel.batches[0])[:20] + list(flywheel.eval_ids)[:20]:
            assert flywheel.splits.items[item_id].text not in blob


def test_the_summary_reports_metrics_per_arm_per_round_and_the_plans_contrasts(small_run):
    _, summary, _ = small_run
    assert [r["round"] for r in summary["per_round"]] == [1, 2]
    for entry in summary["per_round"]:
        assert set(entry["arms"]) == {"0", "A", "B-local", "B", "A+B"}
        for arm in entry["arms"].values():
            assert set(arm["metrics"]) == {"n", "accuracy", "brier", "ece"}
        assert set(entry["contrasts"]) == {"A-0", "B-0", "B-B-local", "A+B-max(A,B)"}
    assert (summary["analyst"]["provider"], summary["analyst"]["model"]) == ("openai", "gpt-6-luna")


def test_two_offline_runs_produce_identical_outputs(small_run, tmp_path):
    first_flywheel, first, _ = small_run
    _, second, _ = _run(tmp_path, "again")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert (first_flywheel.run_dir / "predictions.jsonl").read_text() == (tmp_path / "again" / "predictions.jsonl").read_text()


def test_an_offline_run_refuses_to_start_without_the_network_guard(tmp_path):
    unified_env.remove_network_guard()
    try:
        with pytest.raises(HarnessError):
            UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "x", **SMALL))
    finally:
        unified_env.install_network_guard()


def test_a_run_directory_never_mixes_fake_and_real_answers(small_run):
    flywheel, _, _ = small_run
    manifest = flywheel.run_dir / "run-manifest.json"
    stored = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**stored, "mode": "live", "engine": "jev:jev-1.13.0"}))
    try:
        with pytest.raises(HarnessError):
            UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=flywheel.run_dir, **SMALL))
    finally:
        manifest.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")


def test_metric_definitions_match_jev_flywheel_summarize():
    rows = [ItemResult(str(i), c, k) for i, (c, k) in enumerate([(0.91, 1), (0.55, 0), (0.73, 1), (0.99, 0), (0.6, 1)])]
    ours = summarize(rows)
    theirs = jev_summarize([r.confidence for r in rows], [r.correct for r in rows])
    assert ours["accuracy"] == pytest.approx(theirs.accuracy)
    assert ours["brier"] == pytest.approx(theirs.brier)
    assert ours["ece"] == pytest.approx(theirs.ece)


def test_knn_feature_rows_are_leave_one_out_for_labeled_items(small_run):
    flywheel, _, _ = small_run
    labeled = list(flywheel.batches[0])
    template = candidate_template(flywheel.v1.score("Sentiment"), fewshot=False, knn=True)
    row = flywheel._rows(template, labeled[:1], labeled)[labeled[0]]
    without = flywheel._rows(template, labeled[:1], labeled[1:])[labeled[0]]
    assert row == without
    assert feature_row(template, {}, None) == {}


def test_the_run_config_defaults_to_the_planted_corpus():
    from .unified_corpus import PLANTED

    assert RunConfig(clone=".", run_dir=".").corpus is PLANTED
