"""Specs for arm D (optional dynamic retrieval, final only), offline with the fake Jev and the fake
hashing embedder. They need the pinned Jev-Flywheel clone, scikit-learn and Tactus; run them with
``make unified-flywheel-test UF_PYTHON=/path/to/python``.
"""
import json
import shutil
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

from decision_flywheel.models import Item as DFItem  # noqa: E402
from decision_flywheel.models import LabeledItem  # noqa: E402
from decision_flywheel.retrieval_config import build_retrieval_policy  # noqa: E402

from .unified_fake_jev import FakeJevCore, text_key  # noqa: E402
from .unified_loop import DISPLAY_ORDER, TASK, HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_retrieval import (RETRIEVERS, RetrievalAnswers, retrieval_config,  # noqa: E402
                                retrieval_context)
from .unified_spend import final_d_upper_bound  # noqa: E402
from .unified_splits import LeakageError, load_items  # noqa: E402

SMALL = dict(per_round=40, rounds=2, dev_size=30, bootstrap_resamples=50, max_concurrency=4)
ROUND_ARMS = ("0", "A-c", "F", "A-c+F")


class RecordingCore(FakeJevCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pairs = []

    def answer(self, state, questions):
        target = state["target"]["text"] if "target" in state else state["text"]
        self.pairs.append((text_key(target), tuple(text_key(e["text"]) for e in state.get("labeled_examples") or []),
                           tuple(sorted(questions))))
        return super().answer(state, questions)


def _core():
    items = load_items(Path(CLONE.path) / "fixtures")
    return RecordingCore(planted={text_key(i.text): i.reference_label for i in items.values()}, strength=0.15)


def _d(run_dir, retriever, core=None):
    cfg = RunConfig(clone=CLONE.path, run_dir=run_dir, arms=("D",), final=True, retriever=retriever, **SMALL)
    core = core or _core()
    flywheel = UnifiedFlywheel(cfg, fake_core=core)
    return flywheel, flywheel.run(), core


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """The rounds (A-c's bundle), then the bundle arms' final run (the comparison), as in the study."""
    run_dir = tmp_path_factory.mktemp("retrieval") / "run"
    UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=run_dir, arms=ROUND_ARMS, **SMALL), fake_core=_core()).run()
    UnifiedFlywheel(RunConfig(clone=CLONE.path, run_dir=run_dir, arms=ROUND_ARMS, final=True, **SMALL),
                    fake_core=_core()).run()
    return run_dir


@pytest.fixture(scope="module")
def variants(rounds):
    return {name: _d(rounds, name) for name in RETRIEVERS}


def test_every_variant_sends_one_request_per_labeled_and_per_paper_600_item(variants):
    for name, (flywheel, summary, core) in variants.items():
        labeled = [i for batch in flywheel.batches for i in batch]
        bound = final_d_upper_bound(n_labeled=len(labeled), eval_n=600)["total"]
        assert core.calls == summary["requests"]["new_this_invocation"] == bound == len(labeled) + 600, name
        assert summary["requests"]["by_arm"] == {"D": {"retrieval": bound, "total": bound}}, name
        assert summary["fills"]["labeled_leave_one_out"]["requests"] == len(labeled)
        assert summary["fills"]["evaluation"]["requests"] == 600
        assert summary["metrics"]["n"] == 600 and summary["unscored"] == 0


def test_the_studys_live_bound_is_900_requests_per_variant():
    assert final_d_upper_bound(n_labeled=300, eval_n=600) == {"D": 900, "total": 900}


def test_each_request_carries_retrieved_examples_and_every_a_c_rubric_question(variants):
    for name, (flywheel, summary, core) in variants.items():
        questions = tuple(summary["rubric"]["questions"])
        assert summary["rubric"]["source_arm"] == "A-c"
        assert all(shown and asked == questions for _, shown, asked in core.pairs), name
        assert all(len(shown) == 4 * len(TASK.labels) for _, shown, _ in core.pairs)


def test_leakage_rule_no_target_is_ever_shown_as_its_own_example(variants):
    for name, (_, _, core) in variants.items():
        assert all(target not in shown for target, shown, _ in core.pairs), name


def test_leakage_rule_examples_come_only_from_the_labeled_pool_and_never_from_held_out_items(variants):
    for name, (flywheel, _, core) in variants.items():
        labeled = {text_key(flywheel.splits.items[i].text) for batch in flywheel.batches for i in batch}
        held_out = {text_key(flywheel.splits.items[i].text) for i in flywheel.splits.test}
        shown = {h for _, examples, _ in core.pairs for h in examples}
        assert shown <= labeled and not shown & held_out, name
        targets = {t for t, _, _ in core.pairs}
        paper = {text_key(flywheel.splits.items[i].text) for i in flywheel.splits.paper600}
        assert targets == labeled | paper


def test_the_policy_protects_every_held_out_item_and_refuses_a_held_out_pool(variants):
    flywheel, _, _ = variants["bm25"]
    policy = build_retrieval_policy(retrieval_config("bm25"), protected_ids=flywheel.splits.test)
    assert set(flywheel.splits.test) <= policy.protected_ids
    leaky = [LabeledItem(DFItem(i, {"text": flywheel.splits.items[i].text}), flywheel.splits.items[i].reference_label)
             for i in list(flywheel.splits.pool)[:20] + list(flywheel.splits.paper600)[:1]]
    with pytest.raises(LeakageError):
        retrieval_context(policy, leaky, flywheel.splits, task=TASK, engine="fake", display_order=DISPLAY_ORDER)


def test_a_labeled_items_examples_are_leave_one_out_from_the_same_pool(variants):
    flywheel, _, _ = variants["lexical-v2"]
    labeled = [i for batch in flywheel.batches for i in batch]
    policy = build_retrieval_policy(retrieval_config("lexical-v2"), protected_ids=flywheel.splits.test)
    context = retrieval_context(policy, flywheel._labeled_rows(labeled), flywheel.splits, task=TASK, engine="fake",
                                display_order=DISPLAY_ORDER)
    answers = RetrievalAnswers(flywheel.run_dir / "unused.jsonl", TASK,
                               {i: it.text for i, it in flywheel.splits.items.items()}, flywheel.labels, None)
    for item_id in labeled[:15]:
        shown = [row.item.id for row in answers.examples_for(context, item_id)]
        assert item_id not in shown and set(shown) <= set(labeled)
    assert not (flywheel.run_dir / "unused.jsonl").exists()


def test_the_cache_key_changes_with_the_retriever_configuration(variants, rounds):
    contexts = {name: summary["retrieval"]["context_fingerprint"] for name, (_, summary, _) in variants.items()}
    assert len(set(contexts.values())) == len(RETRIEVERS)
    flywheel, summary, _ = variants["bm25"]
    labeled = [i for batch in flywheel.batches for i in batch]
    rows = flywheel._labeled_rows(labeled)

    def context(config):
        policy = build_retrieval_policy(config, protected_ids=flywheel.splits.test)
        return retrieval_context(policy, rows, flywheel.splits, task=TASK, engine=flywheel.engines.identity,
                                 display_order=DISPLAY_ORDER).fingerprint

    assert context(retrieval_config("bm25")) == summary["retrieval"]["context_fingerprint"]
    assert context(retrieval_config("bm25", per_label=3)) != summary["retrieval"]["context_fingerprint"]
    policy = build_retrieval_policy(retrieval_config("bm25"), protected_ids=flywheel.splits.test)
    fewer = retrieval_context(policy, rows[:-1], flywheel.splits, task=TASK, engine=flywheel.engines.identity,
                              display_order=DISPLAY_ORDER).fingerprint
    assert fewer != summary["retrieval"]["context_fingerprint"]
    # A rerun of the same variant is answered entirely from the cache.
    _, again, core = _d(rounds, "bm25")
    assert core.calls == 0 and again["requests"]["new_this_invocation"] == 0
    assert again["metrics"] == summary["metrics"]


def test_offline_embedding_uses_the_fake_embedder_and_caches_vectors_without_text(variants):
    flywheel, summary, _ = variants["embedding"]
    embedder = summary["retrieval"]["embedder"]
    assert embedder["identity"].startswith("fake-hashing-v1:")
    labeled = [i for batch in flywheel.batches for i in batch]
    assert embedder["texts_embedded_this_invocation"] == len(set(labeled) | set(flywheel.eval_ids))
    assert summary["retrieval"]["retriever"]["embedder"] == embedder["identity"]
    for name in ("bm25", "lexical-v2", "lexical-v1"):
        assert variants[name][1]["retrieval"]["embedder"] is None
    cache = (flywheel.run_dir / "embeddings.jsonl").read_text()
    assert cache and all(set(json.loads(line)) == {"embedder", "text_sha256", "vector"} for line in cache.splitlines())


def test_outputs_are_text_free_and_the_bundle_arms_final_summary_is_untouched(variants, rounds):
    final_summary = (rounds / "final-summary.json").read_text()
    for name, (flywheel, summary, _) in variants.items():
        out = rounds / "final-d" / name
        assert summary["bundle"] is None and "not bundled" in summary["bundle_note"]
        for path in [out / "final-d-summary.json", out / "final-d-run-log.json", out / "final-d-predictions.jsonl",
                     rounds / "retrieval-answers.jsonl", rounds / "embeddings.jsonl"]:
            blob = path.read_text()
            for item_id in [i for batch in flywheel.batches for i in batch][:30] + list(flywheel.eval_ids)[:30]:
                assert flywheel.splits.items[item_id].text not in blob, path
    assert json.loads(final_summary)["final"] is True and "final_d" not in json.loads(final_summary)
    assert (rounds / "final-summary.json").read_text() == final_summary


def test_the_summary_compares_d_with_the_default_a_c_and_f(variants):
    _, summary, _ = variants["embedding"]
    comparison = summary["comparison"]
    assert comparison["source"] == "final-predictions.jsonl"
    assert set(comparison["arms"]) == {"0", "A-c", "F", "A-c+F"}
    assert set(comparison["contrasts"]) == {"D-0", "D-A-c", "D-F", "D-A-c+F"}
    assert summary["head"]["n_train"] == 80 and summary["head"]["missing_features"] == 0


def test_two_offline_runs_of_a_variant_produce_identical_outputs(variants, rounds, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(rounds, copy, ignore=shutil.ignore_patterns("final-d", "retrieval-answers.jsonl",
                                                                "embeddings.jsonl"))
    for name in ("embedding", "bm25"):
        _, first, _ = variants[name]
        _, second, _ = _d(copy, name)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True), name
        assert ((rounds / "final-d" / name / "final-d-predictions.jsonl").read_text()
                == (copy / "final-d" / name / "final-d-predictions.jsonl").read_text())


def test_arm_d_is_final_only_runs_alone_and_needs_a_known_retriever(tmp_path):
    def cfg(**kw):
        return RunConfig(clone=CLONE.path, run_dir=tmp_path / "x", **{**SMALL, **kw})

    for bad in (dict(arms=("D",), retriever="bm25"), dict(arms=("D", "A-c"), final=True, retriever="bm25"),
                dict(arms=("D",), final=True), dict(arms=("D",), final=True, retriever="tfidf"),
                dict(arms=("A-c",), final=True, retriever="bm25")):
        with pytest.raises(HarnessError):
            cfg(**bad).validate()
    cfg(arms=("D",), final=True, retriever="embedding").validate()


def test_arm_d_refuses_to_run_without_the_a_c_bundle(tmp_path):
    with pytest.raises(HarnessError):
        _d(tmp_path / "empty", "bm25")
