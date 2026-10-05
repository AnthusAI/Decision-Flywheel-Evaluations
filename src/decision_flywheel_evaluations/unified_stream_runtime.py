"""Explicit fake-only actual-learning stream runtime for a synthetic corpus.

This module deliberately has no CLI, environment loading, or provider configuration.
Its only entry point is reached through ``unified_stream_cli runtime --synthetic``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any


def _synthetic_corpus(stream_size: int, heldout_size: int):
    """Build a tiny in-memory corpus with deterministic, non-private text."""
    from .unified_corpus import PLANTED
    from .unified_splits import CorpusItem, Splits, label_order

    labels = PLANTED.labels
    items = {}
    pool = []
    for index in range(stream_size):
        label = labels[index % len(labels)]
        cue = "sunny success" if label == labels[0] else "stormy failure"
        item_id = f"synthetic-stream-{index:04d}"
        items[item_id] = CorpusItem(item_id, "pool", label, f"{cue} synthetic item {index}")
        pool.append(item_id)
    development, heldout = [], []
    for index in range(heldout_size):
        label = labels[index % len(labels)]
        cue = "sunny success" if label == labels[0] else "stormy failure"
        development_id = f"synthetic-development-{index:04d}"
        scoreboard_id = f"synthetic-scoreboard-{index:04d}"
        items[development_id] = CorpusItem(development_id, "test", label, f"{cue} development item {index}")
        items[scoreboard_id] = CorpusItem(scoreboard_id, "test", label, f"{cue} scoreboard item {index}")
        development.append(development_id)
        heldout.append(scoreboard_id)

    def load(_fixtures, *, dev_size):
        del dev_size
        return Splits(items, tuple(pool), tuple((*development, *heldout)), tuple(heldout), tuple(development),
                      final_name="synthetic-heldout")

    return replace(PLANTED, name="synthetic-stream", load_splits=load,
                   label_order_fn=lambda splits, seed: label_order(splits.pool, seed),
                   workspace_items_from_pool=True)


def run_synthetic_runtime(run_dir: Path, *, seed: int, review_probability: float,
                          max_new_requests: int, stream_size: int, heldout_size: int) -> dict[str, Any]:
    """Run B/L/E/X through the pinned flywheel with fake Jev and actual learning seams.

    The network guard comes before every optional runtime import and remains active
    for construction, fitting, freezing, and serving.  The shared ledger enforces
    one invocation-wide cap across all arms.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(max_new_requests, bool) or not isinstance(max_new_requests, int) or max_new_requests < 0
            or isinstance(stream_size, bool) or not isinstance(stream_size, int) or stream_size < 31
            or isinstance(heldout_size, bool) or not isinstance(heldout_size, int) or heldout_size < 1
            or isinstance(review_probability, bool) or not isinstance(review_probability, (int, float))
            or not 0 <= review_probability <= 1):
        raise ValueError("synthetic runtime bounds must be valid")
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        # This synthetic exercise is not a resume protocol. Reusing feedback
        # from an earlier attempt would reveal future labels at the cold start.
        raise ValueError("the synthetic runtime requires a fresh empty run directory")

    from . import unified_env

    unified_env.install_network_guard()
    if not unified_env.jev_dependencies_available():
        raise RuntimeError("the synthetic runtime needs the pinned Jev-Flywheel interpreter")
    clone = unified_env.put_clone_first()
    unified_env.ensure_decision_flywheel()

    from .unified_fake_jev import FakeJevCore, text_key
    from .unified_loop import RunConfig, UnifiedFlywheel
    from .unified_spend import SpendLedger
    from .unified_stream import StreamConfig, StreamDriver, UnifiedFlywheelStreamArm

    corpus = _synthetic_corpus(stream_size, heldout_size)
    run_dir = Path(run_dir)
    stream = [(f"synthetic-stream-{index:04d}", corpus.labels[index % len(corpus.labels)])
              for index in range(stream_size)]
    # This fake's planted map is an offline oracle for its answer distribution,
    # never an externally contacted labeler or provider.
    planted = {text_key(f"{'sunny success' if label == corpus.labels[0] else 'stormy failure'} synthetic item {index}"): label
               for index, (_, label) in enumerate(stream)}
    ledger = SpendLedger(run_dir / "stream-ledger.json", ceiling=max_new_requests,
                         max_new=max_new_requests, run_label="synthetic-stream")
    driver = StreamDriver(StreamConfig(seed=seed, review_probability=review_probability,
                                       cold_start_reviews=30, refit_every_reviews=30,
                                       max_refits=4, max_steering_rounds=3,
                                       checkpoint_at=stream_size, list_per_label=4))
    selection_plan = driver.sample_plan(stream)
    outputs = {}
    for arm_name in ("B", "L", "E", "X"):
        core = FakeJevCore(planted=planted, strength=.3)
        cfg = RunConfig(clone=Path(clone.path), run_dir=run_dir / "arms" / arm_name,
                        seed=seed, rounds=1, per_round=1, arms=("0",), corpus=corpus,
                        dev_size=heldout_size, max_new_requests=max_new_requests,
                        request_ceiling=max_new_requests, bootstrap_resamples=1,
                        comments=None)
        flywheel = UnifiedFlywheel(cfg, ledger=ledger, fake_core=core)
        comments = {item_id: "synthetic offline explanation" for item_id, _ in stream} if arm_name in ("E", "X") else None
        arm = UnifiedFlywheelStreamArm.create(flywheel, arm_name, comments=comments)
        outputs[arm_name] = driver.run(arm_name, arm, stream, selection_plan=selection_plan,
                                       is_synthetic=True).to_dict()
    payload = {"schema": "decision-flywheel-evaluations/stream/v1", "is_synthetic": True,
               "labels": list(corpus.labels), "arms": outputs,
               "ledger": ledger.summary()}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stream-results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
