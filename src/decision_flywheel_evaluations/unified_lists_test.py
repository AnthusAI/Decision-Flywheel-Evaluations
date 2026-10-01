"""Specs for the fixed-list arms (F, F-rand, A-c+F), the comment arm A-c and replay, offline.

Like ``unified_loop_test`` they need the pinned Jev-Flywheel clone, scikit-learn and Tactus;
run them with ``make unified-flywheel-test UF_PYTHON=/path/to/python``.
"""
import json
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

from decision_flywheel.context import FixedExampleList  # noqa: E402
from jev_flywheel.items import FeedbackItem  # noqa: E402

from .unified_fake_jev import FakeJevCore, text_key  # noqa: E402
from .unified_lists import LIST_KEY, ListAnswers  # noqa: E402
from .unified_loop import TASK, HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_spend import CircuitOpen  # noqa: E402
from .unified_splits import load_items  # noqa: E402

SMALL = dict(per_round=40, rounds=2, dev_size=30, bootstrap_resamples=50, max_concurrency=4)
LIST_ARMS = ("0", "A", "F", "F-rand", "A-c", "A-c+F")


class RecordingCore(FakeJevCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests = []

    def answer(self, state, questions):
        target = state["target"]["text"] if "target" in state else state["text"]
        shown = tuple(text_key(e["text"]) for e in state.get("labeled_examples") or [])
        self.requests.append({"target": text_key(target), "shown": shown, "questions": dict(questions)})
        return super().answer(state, questions)


def _core():
    items = load_items(Path(CLONE.path) / "fixtures")
    return RecordingCore(planted={text_key(i.text): i.reference_label for i in items.values()}, strength=0.15)


def _run(tmp_path, name, **overrides):
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / name, **{**SMALL, "arms": LIST_ARMS, **overrides})
    core = _core()
    flywheel = UnifiedFlywheel(cfg, fake_core=core)
    return flywheel, flywheel.run(), core


@pytest.fixture(scope="module")
def list_run(tmp_path_factory):
    labeled = None
    root = tmp_path_factory.mktemp("lists")
    # Comments for A-c / A-c+F: an uninformative stand-in for the Phase 2 explanation labeler.
    probe = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=root / "probe", **SMALL), fake_core=_core())
    labeled = [i for batch in probe.batches for i in batch]
    comments = {i: "stand-in reason" for i in labeled[::3]}
    return _run(root, "run", comments=comments) + (comments,)


def test_a_list_arm_sends_one_request_per_item_with_the_examples_and_every_rubric_question(list_run):
    flywheel, _, core, _ = list_run
    few_shot = [r for r in core.requests if r["shown"]]
    assert few_shot, "no request carried an example list"
    for request in few_shot:
        assert len(request["shown"]) == 2 * flywheel.cfg.list_per_label
        assert all(LIST_KEY not in q for q in request["questions"].values())
    # The first request for an item under a list carries the holistic question; a promoted
    # element rides in that same request, and an item already answered is topped up with
    # only the new question.
    first = {}
    for request in few_shot:
        first.setdefault((request["target"], request["shown"]), request)
    assert all("Sentiment" in r["questions"] for r in first.values())
    assert any("Sentiment" in r["questions"] and len(r["questions"]) > 1 for r in few_shot)


def test_a_target_never_sees_itself_and_lists_hold_only_labeled_pool_items(list_run):
    flywheel, summary, core, _ = list_run
    assert all(r["target"] not in r["shown"] for r in core.requests)
    labeled = {i for batch in flywheel.batches for i in batch}
    for entry in summary["per_round"]:
        for arm in ("F", "F-rand", "A-c+F"):
            chosen = entry["arms"][arm]["list"]
            assert set(chosen["example_ids"]) | set(chosen["reserve_ids"]) <= labeled
    held_out = {text_key(flywheel.splits.items[i].text) for i in flywheel.splits.test}
    assert not {h for r in core.requests for h in r["shown"]} & held_out


def test_the_cache_key_includes_the_list_so_a_new_list_is_asked_again_and_an_old_one_is_free(tmp_path):
    items = load_items(Path(CLONE.path) / "fixtures")
    pool = sorted(i for i, item in items.items() if item.split == "pool")
    from decision_flywheel.models import Item, LabeledItem
    rows = {i: LabeledItem(Item(i, {"text": items[i].text}), items[i].reference_label) for i in pool[:200]}
    by_label = {label: [i for i in rows if rows[i].label == label] for label in TASK.labels}
    first = FixedExampleList.from_items(TASK, [rows[i] for label in TASK.labels for i in by_label[label][:2]],
                                        [rows[by_label[label][2]] for label in TASK.labels])
    second = FixedExampleList.from_items(TASK, [rows[i] for label in TASK.labels for i in by_label[label][3:5]],
                                         [rows[by_label[label][5]] for label in TASK.labels])
    core = _core()
    from .unified_fake_jev import FakeJevAsync
    answers = ListAnswers(tmp_path / "a.jsonl", TASK, {i: items[i].text for i in items},
                          {i: items[i].reference_label for i in pool}, lambda: FakeJevAsync(core))
    question = {"Sentiment": {"type": "choice", "instructions": "x", "criteria": {"positive": None, "negative": None}}}
    targets = pool[300:305]
    assert answers.fill(targets, question, first)["requests"] == 5
    assert answers.fill(targets, question, first)["requests"] == 0
    assert answers.fill(targets, question, second)["requests"] == 5
    extra = {**question, "sentiment.extra": {"type": "noul", "instructions": "y", "criteria": None}}
    before = len(core.requests)
    assert answers.fill(targets, extra, first)["requests"] == 5
    assert all(set(r["questions"]) == {"sentiment.extra"} for r in core.requests[before:])
    member = by_label["positive"][0]
    shown = [row.item.id for row in answers.examples_for(first, member)]
    assert member not in shown and by_label["positive"][2] in shown


def test_f_records_its_optimizer_trials_and_f_rand_draws_an_independent_list(list_run):
    _, summary, _, _ = list_run
    for entry in summary["per_round"]:
        f, f_rand = entry["arms"]["F"]["list"], entry["arms"]["F-rand"]["list"]
        assert set(f["scores"]) == {"incumbent", "hard-swap", "random-control"} and f["n_development"] > 0
        assert {t["status"] for t in f["trial_status"].values()} == {"completed"}
        assert f["requests_not_prefetched"] == 0   # every trial answer was prefetched concurrently
        assert f["fingerprint"] != f_rand["fingerprint"]
        assert f_rand["fingerprint"] not in f["trial_fingerprints"].values()
    assert summary["per_round"][0]["arms"]["F"]["list"]["incumbent_source"] == "prototype-seed"
    assert summary["per_round"][1]["arms"]["F"]["list"]["incumbent_source"] == "given"


def test_list_arms_report_metrics_and_the_new_contrasts(list_run):
    _, summary, _, _ = list_run
    for entry in summary["per_round"]:
        assert set(entry["arms"]) == set(LIST_ARMS)
        assert {"F-0", "F-F-rand", "A-c-A", "A-c+F-max(A-c,F)"} <= set(entry["contrasts"])
        for arm in ("F", "F-rand", "A-c+F"):
            assert entry["arms"][arm]["metrics"]["n"] == 30
            assert entry["arms"][arm]["evaluation_rows_missing_features"] == 0
            assert entry["arms"][arm]["list_vs_zero_shot"]["n"] == entry["n_labeled"] + 30
    assert set(summary["requests"]["by_arm"]) >= {"F", "F-rand"}


def test_comments_reach_only_the_comment_arms_feedback(list_run):
    flywheel, _, _, comments = list_run
    def feedback(arm):
        root = flywheel.run_dir / "workspaces" / arm.replace("+", "plus")
        return [FeedbackItem(**json.loads(line)) for line in (root / "feedback.jsonl").read_text().splitlines()]
    assert all(f.edit_comment_value is None for f in feedback("A"))
    for arm in ("A-c", "A-c+F"):
        got = {f.item_id: f.edit_comment_value for f in feedback(arm) if f.edit_comment_value}
        assert got == comments


def test_without_comments_a_c_follows_a_exactly(tmp_path):
    _, summary, _ = _run(tmp_path, "nocomments", arms=("A", "A-c"))
    for entry in summary["per_round"]:
        assert entry["arms"]["A"]["metrics"] == entry["arms"]["A-c"]["metrics"]


def test_two_offline_list_runs_are_identical(list_run, tmp_path):
    _, first, _, comments = list_run
    _, second, _ = _run(tmp_path, "again", comments=comments)
    assert json.dumps(first["per_round"], sort_keys=True) == json.dumps(second["per_round"], sort_keys=True)


def test_a_replay_is_free_when_everything_is_cached_and_refuses_otherwise(tmp_path):
    cfg = dict(SMALL, rounds=1)
    flywheel = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "zero", arms=("0",), replay=True,
                                         provider_model="jev-1.13.0", **cfg))
    assert flywheel.run()["requests"]["new_this_invocation"] == 0
    flywheel = UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=tmp_path / "f", arms=("F",), replay=True,
                                         provider_model="jev-1.13.0", **cfg))
    with pytest.raises((HarnessError, CircuitOpen)):
        flywheel.run()
