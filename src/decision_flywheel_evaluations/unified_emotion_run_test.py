"""Specs for a labels-only Emotion run of the unified loop, offline (synthetic Emotion-shaped rows, a fake Jev).

Like ``unified_loop_test`` they need the pinned clone, scikit-learn and Tactus. They cover what the Emotion
corpus adds around the loop: it runs WITHOUT the explanation labeler when no comments are supplied, its
workspace holds only Emotion pool items (never the planted fixtures' items), and its summaries carry the
multi-class readout (macro-F1, per-class recall, confusion matrix) that planted summaries never do.
"""
import json
from dataclasses import replace
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

from jev_flywheel.host import FlywheelHost  # noqa: E402
from jev_flywheel.items import load_items as load_workspace_items  # noqa: E402
from jev_flywheel.workspace import Workspace  # noqa: E402

from .unified_corpus import EMOTION_CORPUS  # noqa: E402
from .unified_emotion import EMOTION_LABELS, build_emotion_splits  # noqa: E402
from .unified_loop import HarnessError, RunConfig, UnifiedFlywheel  # noqa: E402
from .unified_splits import load_items  # noqa: E402

ARMS = ("0", "A-c", "F", "F-rand", "A-c+F")
SMALL = dict(per_round=80, rounds=2, dev_size=10, bootstrap_resamples=40, max_concurrency=4)


def _splits():
    candidates = [(f"train-{n}", EMOTION_LABELS[n % 6], f"candidate text number {n} about feelings") for n in range(300)]
    scoreboard = [(f"test-{n}", EMOTION_LABELS[(n * 7) % 6], f"scoreboard text number {n} about moods") for n in range(120)]
    return build_emotion_splits(candidates, scoreboard, dev_size=10, final_size=20)[0]


def _corpus(splits):
    return replace(EMOTION_CORPUS, cached_zero_shot=None, load_splits=lambda fixtures, dev_size: splits)


def _run(tmp_path, name, **overrides):
    splits = _splits()
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / name, corpus=_corpus(splits),
                    **{**SMALL, "arms": ARMS, **overrides})
    flywheel = UnifiedFlywheel(cfg)
    return flywheel, flywheel.run(), splits


@pytest.fixture(scope="module")
def emotion_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("emotion"), "run")


def test_a_labels_only_emotion_run_needs_no_labeler_and_completes_for_every_arm(emotion_run):
    flywheel, summary, _ = emotion_run
    assert set(summary["arms"]) == set(ARMS) and len(summary["per_round"]) == 2
    assert summary["policies"]["comments"] == {"supplied": 0, "source": "none (A-c runs as A)"}
    assert all(entry["comments"] == 0 for round_ in summary["per_round"] for arm, entry in round_["arms"].items()
               if arm in ("A-c", "A-c+F") and "comments" in entry)
    assert flywheel.cfg.comments is None and set(summary["requests"]["by_arm"]) >= {"A-c", "F"}


def test_a_run_that_passes_comments_is_refused_while_the_labeler_does_not_exist(tmp_path):
    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / "x", corpus=EMOTION_CORPUS, comments={"train-1": "a reason"}, **SMALL)
    with pytest.raises(NotImplementedError, match="S5"):
        UnifiedFlywheel(cfg)


def test_each_arm_reports_the_multiclass_readout_beside_the_unchanged_legacy_metrics(emotion_run):
    _, summary, _ = emotion_run
    for round_ in summary["per_round"]:
        for arm, entry in round_["arms"].items():
            assert set(entry["metrics"]) == {"n", "accuracy", "brier", "ece"}   # top-label, as before
            readout = entry["multiclass"]
            assert readout["n"] == 10 and set(readout["per_class"]) == set(EMOTION_LABELS)
            assert readout["confusion_matrix"]["labels"] == list(EMOTION_LABELS)
            assert len(readout["confusion_matrix"]["counts"]) == 6 and all(len(r) == 6 for r in readout["confusion_matrix"]["counts"])
            assert readout["accuracy"] == entry["metrics"]["accuracy"]
            assert readout["low_support_classes"] == list(EMOTION_LABELS)   # 10 dev items cannot support any class
        contrasts = round_["multiclass_contrasts"]
        assert {"A-c-0", "F-0", "F-F-rand", "A-c+F-max(A-c,F)"} <= set(contrasts)
        assert set(contrasts["F-0"]) == {"macro_f1", "accuracy"}
        assert set(round_["contrasts"]) >= {"F-0"}   # the legacy contrasts stay too


def test_predictions_carry_the_predicted_and_gold_labels_for_emotion(emotion_run):
    flywheel, _, _ = emotion_run
    rows = [json.loads(line) for line in (flywheel.run_dir / "predictions.jsonl").read_text().splitlines()]
    assert rows and all(r["predicted"] in EMOTION_LABELS and r["gold"] in EMOTION_LABELS for r in rows)
    assert all(r["correct"] == int(r["predicted"] == r["gold"]) for r in rows)


# ---- the workspace holds Emotion items only ----------------------------------------------------------------------

def test_an_emotion_workspace_holds_only_emotion_pool_items_and_none_of_the_planted_corpus(emotion_run):
    flywheel, _, splits = emotion_run
    planted = load_items(Path(CLONE.path) / "fixtures")
    planted_texts = {item.text for item in planted.values()}
    for arm in ("A-c", "A-c+F"):
        root = flywheel.run_dir / "workspaces" / arm.replace("+", "plus")
        items = load_workspace_items(root / "items.jsonl")
        assert {i.id for i in items} == set(splits.pool)
        assert not {i.text for i in items} & planted_texts and not {i.id for i in items} & set(planted)
        assert all(i.text == splits.items[i.id].text for i in items)
        assert not set(splits.test) & {i.id for i in items}                 # held-out items are never in the workspace
        assert "reference_label" not in (root / "items.jsonl").read_text()   # gold labels reach it only as feedback


def test_the_analyst_briefing_can_only_show_emotion_pool_items(emotion_run):
    flywheel, _, splits = emotion_run
    workspace = Workspace(flywheel.run_dir / "workspaces" / "A-c")
    briefing = FlywheelHost(workspace, flywheel.corpus.score_name).briefing()
    blob = json.dumps(briefing)
    shown = {entry["text"] for key in ("mismatches", "agreements", "labeled_sample") for entry in briefing.get(key, [])
             if isinstance(entry, dict) and "text" in entry}
    pool_texts = {splits.items[i].text for i in splits.pool}
    assert shown and shown <= pool_texts
    planted = load_items(Path(CLONE.path) / "fixtures")
    assert not any(item.text in blob for item in planted.values())
    assert not any(splits.items[i].text in blob for i in splits.test)


def test_the_offline_fake_analyst_does_not_replay_the_planted_corpus_recording_for_emotion(emotion_run):
    flywheel, summary, _ = emotion_run
    steer = summary["per_round"][0]["arms"]["A-c"]["steering"]
    assert "topic_domain" not in steer["elements_after"]
    text = "".join(p.read_text() for p in (flywheel.run_dir / "workspaces" / "A-c").rglob("*.yaml"))
    assert "sports" not in text.lower() and "workplace" not in text.lower()


def test_the_planted_workspace_still_copies_the_fixture_items_byte_for_byte(tmp_path):
    from .unified_corpus import PLANTED

    cfg = RunConfig(clone=CLONE.path, run_dir=tmp_path / "planted", corpus=PLANTED, arms=("0", "A"), **{**SMALL, "rounds": 1})
    flywheel = UnifiedFlywheel(cfg)
    flywheel._workspace("A")
    assert (flywheel.run_dir / "workspaces" / "A" / "items.jsonl").read_bytes() == (Path(CLONE.path) / "fixtures" / "items.jsonl").read_bytes()
