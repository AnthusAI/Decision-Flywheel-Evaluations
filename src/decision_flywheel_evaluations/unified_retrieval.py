"""Arm D: optional dynamic retrieval, final-only (studies/DECISION_FLYWHEEL_DESIGN.md, sections 4-5).

Dynamic retrieval is an optional feature, off by default and independent of the fixed-list
optimizer; the default product stays A-c+F. Arm D exists only to report how far that default is
from per-item retrieval. It is

    D = A-c's discovered rubric + each item's own retrieved examples + a refit head.

It runs after the rounds, behind ``--final`` (so it reads paper-600 only as a scored slice):

1. Load the A-c bundle the last round froze and take its rubric questions (the head is ignored).
2. The labeled pool is the run's labels (300 for the seed-1 study). Each item gets ``k`` examples
   per label from that pool through core's ``PerLabelRetrieval`` (``retrieval_config``), in the
   owner's request shape: **one Jev request per item**, ``state = {labeled_examples, target}``
   with every rubric question. Labeled items retrieve leave-one-out (the policy and the shared
   firewall exclude the target by ID and normalized text); held-out items are never in the pool,
   and every held-out ID is also passed to the policy as protected.
3. The head is refit on the labeled items' answers (with their leave-one-out examples) and then
   scores paper-600 from the paper items' answers (with their retrieved examples).

Variants (``--retriever``): ``embedding`` (OpenAI ``text-embedding-3-small``, 512 d, live only;
offline the fake ``HashingEmbedder``; vectors cached text-free on disk), ``bm25``, ``lexical-v2``
(stop words on, binary cosine) and ``lexical-v1`` (the v1-equivalent configuration used by
``scripts/retrieval_purity.py``: no stop words, ASCII tokens, binary cosine).

Answers are cached like the list arms' (``ListAnswers``), keyed by the question body plus a
*retrieval context fingerprint*: the policy fingerprint (retriever configuration, ``k``, the
protected set), the sorted labeled IDs, the display order, the task and the engine. A different
retriever, ``k`` or pool therefore never reuses an answer.

D is **not bundled**: core's ``ClassifierBundle`` records ``retrieval`` as ``null`` and
``load_bundle`` refuses retrieval bundles, and D is optional, so core is not widened for it.

Outputs go to ``run_dir/final-d/<variant>/`` (``final-d-summary.json``, ``final-d-run-log.json``,
``final-d-predictions.jsonl``) and never touch ``final-summary.json``. The comparison with A-c+F,
A-c, F and 0 reads the earlier final run's text-free ``final-predictions.jsonl`` when it exists.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from decision_flywheel.budget import ContextBudget, build_context_plan
from decision_flywheel.embedding_retrieval import HashingEmbedder
from decision_flywheel.models import Item as DFItem
from decision_flywheel.models import LabeledItem
from decision_flywheel.retrieval_config import (EmbeddingOptions, LexicalOptions, RetrievalConfig,
                                                build_retrieval_policy)

from .unified_knn import context_fingerprint
from .unified_lists import ListAnswers
from .unified_splits import (LeakageError, Splits, assert_labels_from_pool, assert_retrieval_pool_clean,
                             assert_target_excluded, fingerprint_ids)
from .unified_stats import METRICS, ItemResult, paired_interval, summarize

D_ARM = "D"
RETRIEVERS = ("embedding", "bm25", "lexical-v2", "lexical-v1")
RETRIEVAL_KEY = "retrieval_context"
SOURCE_ARM = "A-c"
D_PER_LABEL = 4
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
COMPARISON_ARMS = ("A-c+F", "A-c", "F", "0")
REQUEST_SHAPE = "one Jev request per item: retrieved examples in state + every rubric question"
NOT_BUNDLED = ("not bundled: core ClassifierBundle records retrieval as null and load_bundle refuses "
               "retrieval bundles; D is optional, so core is not widened")


class RetrievalReplayMiss(RuntimeError):
    """A replay needed an embedding that is not cached. Nothing was sent."""


# ---- configuration ----------------------------------------------------------------------------

def retrieval_config(variant: str, *, embedding_provider: str = "fake", embedding_cache: Optional[Path] = None,
                     per_label: int = D_PER_LABEL) -> RetrievalConfig:
    """Core's ``retrieval:`` block for one D variant (always ``enabled``: D is the opt-in arm)."""
    if variant not in RETRIEVERS:
        raise ValueError(f"unknown retriever {variant!r}; choose from {RETRIEVERS}")
    if variant == "embedding":
        return RetrievalConfig(enabled=True, per_label=per_label, backend="embedding", embedding=EmbeddingOptions(
            provider=embedding_provider, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS,
            cache=str(embedding_cache) if embedding_cache else None))
    lexical = {"bm25": LexicalOptions(weighting="bm25"), "lexical-v2": LexicalOptions(),
               "lexical-v1": LexicalOptions(stopwords="none", tokenizer="ascii-v1")}[variant]
    return RetrievalConfig(enabled=True, per_label=per_label, backend="lexical", lexical=lexical,
                           embedding=EmbeddingOptions(cache=None))


class CountingEmbedder:
    """Wraps an embedder (same ``identity``) and counts the texts it was asked to embed."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls = 0
        self.texts = 0

    @property
    def identity(self) -> str:
        return self.inner.identity

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return self.inner(texts)


class CacheOnlyEmbedder:
    """Replay: the live embedder's identity, but every vector must come from the cache."""

    identity = f"openai:{EMBEDDING_MODEL}:{EMBEDDING_DIMENSIONS}"

    def __call__(self, texts):
        raise RetrievalReplayMiss("replay needed an uncached embedding")


def embedder_for(*, live: bool, replay: bool) -> CountingEmbedder:
    """Offline: the fake hashing embedder. Live: OpenAI, built only here (after every gate)."""
    if live:  # pragma: no cover - never run offline
        from decision_flywheel.embedding_retrieval import OpenAIEmbedder
        return CountingEmbedder(OpenAIEmbedder.from_environment(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS))
    if replay:
        return CountingEmbedder(CacheOnlyEmbedder())
    return CountingEmbedder(HashingEmbedder(EMBEDDING_DIMENSIONS))


# ---- the per-item context and its answers -----------------------------------------------------

@dataclass(frozen=True, eq=False)
class RetrievalContext:
    """What a D answer depends on: the policy over one labeled pool, and its fingerprint."""

    policy: Any
    candidates: Tuple[LabeledItem, ...]
    fingerprint: str
    per_label: int
    pool_ids: FrozenSet[str]


def retrieval_context(policy: Any, rows: Sequence[LabeledItem], splits: Splits, *, task, engine: str,
                      display_order: str) -> RetrievalContext:
    """The context for a labeled pool, after the leakage rules (pool items only, never held out)."""
    ids = [row.item.id for row in rows]
    assert_labels_from_pool(ids, splits)
    assert_retrieval_pool_clean(ids, splits)
    per_label = policy.per_label
    fingerprint = context_fingerprint(policy.fingerprint, ids, per_label=per_label, display_order=display_order,
                                      task=task.fingerprint, engine=engine, request=REQUEST_SHAPE)
    return RetrievalContext(policy, tuple(sorted(rows, key=lambda row: row.item.id)), fingerprint, per_label,
                            frozenset(ids))


class RetrievalAnswers(ListAnswers):
    """``ListAnswers`` with each item's own retrieved examples instead of one fixed list."""

    key = RETRIEVAL_KEY

    def examples_for(self, context: RetrievalContext, item_id: str) -> Tuple[LabeledItem, ...]:
        key = (context.fingerprint, item_id)
        if key not in self._examples:
            target = DFItem(item_id, {self.task.input_field: self.texts[item_id]})
            plan = build_context_plan(self.task, target, context.candidates, context.policy,
                                      budget=ContextBudget(per_label=context.per_label), display_order="canonical",
                                      order_seed=0, presentation_label_order=self.task.labels)
            assert_target_excluded(item_id, plan.example_ids)
            if not set(plan.example_ids) <= context.pool_ids:
                raise LeakageError("a retrieved example is not in the labeled pool")
            self._examples[key] = plan.examples
        return self._examples[key]


# ---- the arm ------------------------------------------------------------------------------------

def _comparison(run_dir: Path, round_number: int, eval_ids: Sequence[str]) -> Tuple[Dict[str, List[ItemResult]], Optional[str]]:
    """The earlier final run's per-item results for the comparison arms (complete arms only)."""
    path = run_dir / "final-predictions.jsonl"
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    by_arm: Dict[str, List[ItemResult]] = {}
    for line in raw.decode().splitlines():
        row = json.loads(line)
        if row["arm"] in COMPARISON_ARMS and row["round"] == round_number:
            by_arm.setdefault(row["arm"], []).append(ItemResult(row["item_id"], row["confidence"], row["correct"]))
    wanted = set(eval_ids)
    complete = {arm: rows for arm, rows in by_arm.items()
                if {r.item_id for r in rows} == wanted and len(rows) == len(wanted)}
    return complete, hashlib.sha256(raw).hexdigest()


def run_final_d(flywheel) -> Dict[str, Any]:
    """Arm D on the evaluation slice; writes ``run_dir/final-d/<variant>/final-d-*``."""
    from jev_flywheel.fit import fit_head
    from jev_flywheel.items import agrees
    from jev_flywheel.ladder import LadderRefusal

    from . import unified_env
    from .unified_bundles import text_free
    from .unified_loop import (DISPLAY_ORDER, HARNESS_VERSION, SCORE_NAME, TASK, HarnessError, _assert_text_free,
                               _summary_dict, candidate_template, feature_row, head_from_fit, serve, training_set)

    cfg, splits, ledger = flywheel.cfg, flywheel.splits, flywheel.ledger
    variant = cfg.retriever
    labeled = [i for batch in flywheel.batches for i in batch]
    eval_ids = list(flywheel.eval_ids)

    bundle = flywheel.load_arm_bundle(SOURCE_ARM)
    template = candidate_template(bundle.rubric.score, fewshot=False, knn=False)
    questions = template.questions()

    embedder = embedder_for(live=cfg.live, replay=cfg.replay) if variant == "embedding" else None
    cache_path = Path(cfg.embedding_cache) if cfg.embedding_cache else flywheel.run_dir / "embeddings.jsonl"
    config = retrieval_config(variant, embedding_provider="openai" if (cfg.live or cfg.replay) else "fake",
                              embedding_cache=cache_path if embedder else None)
    policy = build_retrieval_policy(config, embedder=embedder, protected_ids=splits.test)
    context = retrieval_context(policy, flywheel._labeled_rows(labeled), splits, task=TASK,
                                engine=flywheel.engines.identity, display_order=DISPLAY_ORDER)
    answers = RetrievalAnswers(flywheel.run_dir / "retrieval-answers.jsonl", TASK,
                               {i: item.text for i, item in splits.items.items()}, flywheel.labels,
                               flywheel.engines.zero_shot_factory, concurrency=cfg.max_concurrency)
    if embedder is not None:   # embed every target in batches up front (one by one otherwise)
        flywheel._step(D_ARM, "final", "embedding", "embed-pool-and-targets", lambda: policy.retriever.warm(
            TASK, [DFItem(i, {TASK.input_field: splits.items[i].text}) for i in labeled + eval_ids]))

    def fill(ids: Sequence[str]) -> Dict[str, int]:
        out = answers.fill(ids, questions, context)
        ledger.raise_if_tripped()
        return out

    fills = {"labeled_leave_one_out": flywheel._step(D_ARM, "final", "retrieval", "fill-labeled-leave-one-out",
                                                      lambda: fill(labeled)),
             "evaluation": flywheel._step(D_ARM, "final", "retrieval", "fill-evaluation-slice",
                                          lambda: fill(eval_ids))}
    if cfg.replay and ledger.new_attempts:
        raise HarnessError(f"replay needed {ledger.new_attempts} uncached answers; it is not a pure replay")

    found = answers.answers(labeled, questions, context)
    rows = {i: feature_row(template, found[i], None) for i in labeled}
    training = training_set(template, labeled, flywheel.labels, rows, f"retrieval:{context.fingerprint}")
    try:
        result = fit_head(training, template, folds=cfg.folds, seed=cfg.fit_seed)
    except LadderRefusal as refusal:
        raise HarnessError(f"arm D's head could not be fit: {refusal}") from refusal
    if not result.fitted:
        raise HarnessError(f"arm D's head could not be fit: {result.reason}")
    head = head_from_fit(template, result)

    scored = answers.answers(eval_ids, questions, context)
    results: List[ItemResult] = []
    for item_id in eval_ids:
        if any(name not in scored[item_id] for name in questions):
            continue   # a failed request: reported as unscored, never guessed
        value, confidence = serve(head, feature_row(head, scored[item_id], None))
        results.append(ItemResult(item_id, confidence, int(agrees(value, splits.items[item_id].reference_label))))

    comparison, source_sha = _comparison(flywheel.run_dir, cfg.rounds, eval_ids)
    contrasts: Dict[str, Any] = {}
    if len(results) == len(eval_ids):
        for arm, rows_ in comparison.items():
            contrasts[f"D-{arm}"] = {name: paired_interval(rows_, results, name, resamples=cfg.bootstrap_resamples,
                                                           seed=cfg.bootstrap_seed) for name in METRICS}

    holistic = {SCORE_NAME: questions[SCORE_NAME]} if SCORE_NAME in questions else {}
    zero = flywheel.cache.bulk_partial_answers(labeled + eval_ids, holistic)
    every = {**found, **scored}
    pairs = [(every[i].get(SCORE_NAME), zero[i].get(SCORE_NAME)) for i in labeled + eval_ids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    changed = sum(a.get("choice") != b.get("choice") for a, b in pairs)

    decision = head.decision
    summary = {
        "harness": HARNESS_VERSION, "arm": D_ARM, "final": True, "final_d": True,
        "mode": flywheel.engines.mode, "replay": cfg.replay, "engine": flywheel.engines.identity,
        "jev_flywheel": {"clone": "var/" + Path(cfg.clone).name, "commit": unified_env.JEV_FLYWHEEL_COMMIT},
        "decision_flywheel": flywheel.decision_flywheel,
        "seed": cfg.seed, "rounds": cfg.rounds, "per_round": cfg.per_round,
        "evaluation_slice": {"name": flywheel.slice_name, "n": len(eval_ids), "fingerprint": fingerprint_ids(eval_ids)},
        "label_order_sha256": flywheel._order_sha(),
        "retrieval": {
            "variant": variant, "enabled": True, "default": "off (the product default is A-c+F)",
            "backend": config.backend, "per_label": config.per_label, "combine": "head refit on retrieved-example answers",
            "lexical": asdict(config.lexical) if config.backend == "lexical" else None,
            "retriever": dict(policy.retriever.configuration), "policy_fingerprint": policy.fingerprint,
            "context_fingerprint": context.fingerprint, "request": REQUEST_SHAPE,
            "pool": {"n": len(labeled), "fingerprint": fingerprint_ids(labeled), "labeled_items": "leave-one-out"},
            "protected": {"held_out_ids": len(splits.test)},
            "embedder": None if embedder is None else {"identity": embedder.identity,
                                                       "texts_embedded_this_invocation": embedder.texts,
                                                       "embedding_calls_this_invocation": embedder.calls},
        },
        "rubric": {"source_arm": SOURCE_ARM, "source_bundle": text_free(bundle.manifest()),
                   "questions": sorted(questions)},
        "head": {"features": list(decision.features), "fit_id": (decision.provenance or {}).get("fit_id"),
                 "calibration": (decision.calibration or {}).get("method"), "n_train": training.n,
                 "missing_features": len(training.needs_answers), "fit_head_oof": _summary_dict(result.metrics)},
        "bundle": None, "bundle_note": NOT_BUNDLED,
        "fills": fills, "upper_bound_requests": {"labeled": len(labeled), "evaluation": len(eval_ids),
                                                 "total": len(labeled) + len(eval_ids)},
        "metrics": summarize(results), "unscored": len(eval_ids) - len(results),
        "comparison": {"source": "final-predictions.jsonl" if source_sha else None, "source_sha256": source_sha,
                       "arms": {arm: summarize(rows_) for arm, rows_ in sorted(comparison.items())},
                       "contrasts": contrasts},
        "retrieved_vs_zero_shot": {"n": len(pairs),
                                   "choice_differs_rate": round(changed / len(pairs), 4) if pairs else None},
        "requests": ledger.summary(),
    }
    _assert_text_free(summary, splits)
    out_dir = flywheel.run_dir / "final-d" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final-d-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "final-d-run-log.json").write_text(json.dumps(
        {"events": flywheel.events, "requests": ledger.summary()}, indent=2, sort_keys=True) + "\n")
    with (out_dir / "final-d-predictions.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps({"round": "final", "arm": D_ARM, "variant": variant, "item_id": row.item_id,
                                     "confidence": round(row.confidence, 6), "correct": row.correct}) + "\n")
    return summary
