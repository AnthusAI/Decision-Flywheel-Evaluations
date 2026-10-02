"""Command line for the unified-flywheel harness (``scripts/unified_flywheel.py``).

Offline is the default and the only mode the specs exercise: a deterministic fake Jev
behind the same counting clients, the analyst replaced by fixed replies through
``jev_flywheel.steer``'s ``mock_replies``, and a process-wide guard that blocks every
non-local socket.

Live mode mirrors the repository's collector: it needs ``--live`` *and* ``--confirm``, a
request ceiling no greater than the plan's 9,500, a per-invocation cap at least as large as
the run's precomputed upper bound, a durable ``--spend-ledger``, and an explicit Jev model.
Every one of these is checked before any data is read or any client is constructed.

``run --final`` runs no rounds: it reloads the bundles the last round froze in ``--run-dir`` and
scores paper-600 through them; its upper bound is one request per item per bundle.

``run --final --arms D --retriever <variant>`` is the optional dynamic-retrieval arm
(``unified_retrieval``): it runs alone, after the rounds, and its upper bound is one request per
labeled item plus one per paper-600 item (300 + 600 = 900 for the seed-1 study). ``embedding``
calls OpenAI only with ``--live --confirm``; offline it uses a fake hashing embedder.

``comments`` writes the simulated reviewer's reasons for the labeled items (``unified_labeler``):
offline with a deterministic fake by default, ``--replay`` from the cache, or ``--live --confirm``
against OpenAI with a durable call ledger capped at 300 calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import unified_env
from .unified_corpus import CORPUS_CHOICES, DEFAULT_CORPUS, get_corpus
from .unified_spend import final_d_upper_bound, final_upper_bound, request_upper_bound

UNIFIED_ARMS = ("0", "A", "B-local", "B", "A+B")
ARMS = UNIFIED_ARMS + ("F", "F-rand", "A-c", "A-c+F")
BUNDLE_ARMS = ("0", "A", "A-c", "F", "F-rand", "A-c+F")
D_ARM = "D"                                     # final-only, optional (unified_retrieval)
RETRIEVERS = ("embedding", "bm25", "lexical-v2", "lexical-v1")
PLAN_REQUEST_CEILING = 9500
DEFAULT_RUN_ROOT = unified_env.REPO_ROOT / "var" / "unified-flywheel"


class UsageError(ValueError):
    pass


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="unified_flywheel.py", description=__doc__.split("\n")[0])
    commands = top.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-clone", help="clone Jev-Flywheel locally at the pinned commit")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--clone", type=Path, default=None)

    commands.add_parser("check-env", help="verify the clone, Decision-Flywheel and scikit-learn/Tactus; spends nothing")

    run = commands.add_parser("run", help="run the unified loop (offline with a fake Jev unless --live)")
    run.add_argument("--corpus", choices=CORPUS_CHOICES, default=DEFAULT_CORPUS,
                     help="the corpus to run on (labels, task wording, loader); default " + DEFAULT_CORPUS)
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--clone", type=Path, default=None)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--arms", default=",".join(UNIFIED_ARMS),
                     help="comma-separated subset of " + ",".join(ARMS) + "; or D alone with --final")
    run.add_argument("--retriever", choices=RETRIEVERS, default=None,
                     help="arm D only: the dynamic-retrieval variant (embedding is live-only OpenAI; offline a fake)")
    run.add_argument("--embedding-cache", type=Path, default=None,
                     help="arm D embedding only: text-free vector cache (default: <run-dir>/embeddings.jsonl)")
    run.add_argument("--comments", type=Path, default=None,
                     help="JSONL of {item_id, comment} from an explanation labeler, for arms A-c and A-c+F")
    run.add_argument("--final", action="store_true",
                     help="no rounds: score paper-600 through the bundles the last round froze in --run-dir")
    run.add_argument("--max-concurrency", type=int, default=8)
    run.add_argument("--request-ceiling", type=int, default=PLAN_REQUEST_CEILING)
    run.add_argument("--analyst-provider", default="openai")
    run.add_argument("--analyst-model", default="gpt-6-luna")
    run.add_argument("--analyst-replies", type=Path, default=None,
                     help="offline only: JSON object {round: reply} replacing the fixed fake replies")
    run.add_argument("--bootstrap-resamples", type=int, default=1000)
    run.add_argument("--replay", action="store_true",
                     help="offline, $0: rerun a copied live run directory from its cached answers and recorded "
                          "analyst replies; fails if anything is uncached (needs --provider-model)")
    live = run.add_argument_group("live mode (never default)")
    live.add_argument("--live", action="store_true")
    live.add_argument("--confirm", action="store_true")
    live.add_argument("--spend-ledger", type=Path, default=None)
    live.add_argument("--max-new-requests", type=int, default=None)
    live.add_argument("--provider-model", default=None)
    live.add_argument("--base-url", default=None)
    live.add_argument("--timeout-seconds", type=float, default=30.0)
    live.add_argument("--max-consecutive-failures", type=int, default=25)

    comments = commands.add_parser("comments", help="write the simulated reviewer's reasons for the labeled items")
    comments.add_argument("--out", type=Path, required=True, help="JSONL of {item_id, comment} for run --comments")
    comments.add_argument("--cache", type=Path, default=DEFAULT_RUN_ROOT / "labeler-cache.jsonl")
    comments.add_argument("--corpus", choices=CORPUS_CHOICES, default=DEFAULT_CORPUS,
                          help="the corpus whose labeled items get comments; default " + DEFAULT_CORPUS)
    comments.add_argument("--clone", type=Path, default=None)
    comments.add_argument("--seed", type=int, default=1)
    comments.add_argument("--rounds", type=int, default=3)
    comments.add_argument("--per-round", type=int, default=100)
    comments.add_argument("--model", default=None, help="the labeler model (live or replay); default gpt-6-luna")
    comments.add_argument("--replay", action="store_true", help="offline: every comment must come from the cache")
    comments.add_argument("--live", action="store_true")
    comments.add_argument("--confirm", action="store_true")
    comments.add_argument("--spend-ledger", type=Path, default=None)
    comments.add_argument("--max-calls", type=int, default=None)
    return top


def parse_arms(text: str) -> tuple:
    arms = tuple(a.strip() for a in text.split(",") if a.strip())
    unknown = set(arms) - set(ARMS) - {D_ARM}
    if not arms or unknown:
        raise UsageError(f"unknown arms {sorted(unknown)}; choose from {','.join(ARMS)} (or D alone with --final)")
    if D_ARM in arms:
        if arms != (D_ARM,):
            raise UsageError("arm D runs alone (--final --arms D --retriever ...)")
        return arms
    return tuple(a for a in ARMS if a in arms)


def check_live_gates(args: argparse.Namespace, arms: tuple, eval_n: int) -> Dict[str, int]:
    """Every live gate, checked before any data is loaded or any client is constructed."""
    if getattr(args, "replay", False):
        raise UsageError("--replay is offline and cannot be combined with --live")
    if args.confirm is not True:
        raise UsageError("--live needs --confirm; nothing was loaded and no client was constructed")
    if not 0 <= args.request_ceiling <= PLAN_REQUEST_CEILING:
        raise UsageError(f"--request-ceiling must be between 0 and the plan's {PLAN_REQUEST_CEILING}")
    if args.spend_ledger is None:
        raise UsageError("--live needs --spend-ledger, the durable cumulative request counter")
    if args.provider_model is None:
        raise UsageError("--live needs an explicit --provider-model (the Jev model the cached answers came from)")
    if args.max_new_requests is None or args.max_new_requests < 0 or args.max_new_requests > args.request_ceiling:
        raise UsageError("--live needs --max-new-requests between 0 and --request-ceiling")
    if args.analyst_replies is not None:
        raise UsageError("--analyst-replies is for offline runs; live runs replay their own recorded replies")
    bound = upper_bound(args, arms)
    if bound["total"] > args.max_new_requests:
        raise UsageError(f"this run may need up to {bound['total']} requests ({bound}); "
                         f"--max-new-requests {args.max_new_requests} is lower. Raise it deliberately or run fewer arms/rounds")
    return bound


def upper_bound(args: argparse.Namespace, arms: tuple) -> Dict[str, int]:
    if D_ARM in arms:
        if not args.final:
            raise UsageError("arm D is final-only: run the rounds first, then --final --arms D --retriever ...")
        if args.retriever is None:
            raise UsageError(f"arm D needs --retriever ({','.join(RETRIEVERS)})")
        return final_d_upper_bound(n_labeled=args.rounds * 100, eval_n=600)
    if args.retriever is not None or args.embedding_cache is not None:
        raise UsageError("--retriever and --embedding-cache apply only to arm D")
    if args.final:
        if set(arms) - set(BUNDLE_ARMS):
            raise UsageError(f"--final scores frozen bundles; choose arms from {','.join(BUNDLE_ARMS)}")
        return final_upper_bound(arms, eval_n=600)
    return request_upper_bound(arms, rounds=args.rounds, per_round=100, eval_n=100)


def _offline_replies(path: Optional[Path]) -> Optional[Dict[int, str]]:
    if path is None:
        return None
    raw = json.loads(path.read_text())
    return {int(k): v if isinstance(v, str) else json.dumps(v) for k, v in raw.items()}


def _comments(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["item_id"]): str(row["comment"]) for row in rows if row.get("comment")}


def _require_corpus_ready(name: str, what: str) -> None:
    try:
        get_corpus(name).require_ready(what)
    except NotImplementedError as error:
        raise UsageError(str(error)) from None


def run(args: argparse.Namespace) -> Dict:
    _require_corpus_ready(args.corpus, "a flywheel run")
    arms = parse_arms(args.arms)
    eval_n = 600 if args.final else 100
    bound = upper_bound(args, arms)
    if args.final and args.run_dir is None:
        raise UsageError("--final needs the --run-dir whose last round froze the bundles")
    if args.live:
        check_live_gates(args, arms, eval_n)
    else:
        if args.confirm or args.spend_ledger or args.max_new_requests is not None:
            raise UsageError("live-only options were given without --live; refusing to guess")
        if bool(args.provider_model) != bool(args.replay):
            raise UsageError("--provider-model without --live is only for --replay, and --replay needs it")
        if args.replay and args.run_dir is None:
            raise UsageError("--replay needs an explicit --run-dir (a copy of a live run directory)")
        unified_env.install_network_guard()
    core = unified_env.ensure_decision_flywheel()
    if args.live and core.get("dirty"):
        raise UsageError(f"the Decision-Flywheel working tree at {core.get('path')} has uncommitted changes in src; "
                         "commit them first so the run records an exact core commit")
    identity = unified_env.put_clone_first(args.clone)
    if not unified_env.jev_dependencies_available():
        raise unified_env.EnvironmentProblem(
            "this interpreter lacks scikit-learn or Tactus; use one that has both "
            "(Jev-Flywheel's environment does), see `make unified-flywheel-dry-run`")
    from .unified_loop import RunConfig, UnifiedFlywheel  # noqa: PLC0415 - after the path checks

    mode = "live" if args.live else "offline"
    run_dir = args.run_dir or DEFAULT_RUN_ROOT / f"{mode}-seed{args.seed}"
    cfg = RunConfig(
        corpus=get_corpus(args.corpus), clone=Path(identity.path), run_dir=Path(run_dir), seed=args.seed, rounds=args.rounds, arms=arms,
        final=args.final, live=args.live, replay=args.replay, request_ceiling=args.request_ceiling,
        max_new_requests=args.max_new_requests, spend_ledger=args.spend_ledger,
        max_concurrency=args.max_concurrency, max_consecutive_failures=args.max_consecutive_failures,
        analyst_provider=args.analyst_provider, analyst_model=args.analyst_model,
        analyst_replies=_offline_replies(args.analyst_replies), comments=_comments(args.comments),
        provider_model=args.provider_model or "fake-jev-0", base_url=args.base_url,
        timeout_seconds=args.timeout_seconds, bootstrap_resamples=args.bootstrap_resamples,
        retriever=args.retriever, embedding_cache=args.embedding_cache)
    summary = UnifiedFlywheel(cfg).run()
    summary["upper_bound_requests"] = bound
    return summary


def check_comment_gates(args: argparse.Namespace, needed: int) -> None:
    """The labeler's live gates: --confirm, a durable ledger, and a call cap that covers the run."""
    from .unified_labeler import LABELER_CALL_CEILING

    if args.replay:
        raise UsageError("--replay is offline and cannot be combined with --live")
    if args.confirm is not True:
        raise UsageError("--live needs --confirm; no client was constructed")
    if args.spend_ledger is None:
        raise UsageError("--live needs --spend-ledger, the durable call counter")
    if args.max_calls is None or not 0 <= args.max_calls <= LABELER_CALL_CEILING:
        raise UsageError(f"--live needs --max-calls between 0 and {LABELER_CALL_CEILING}")
    if needed > args.max_calls:
        raise UsageError(f"this run may need up to {needed} calls; --max-calls {args.max_calls} is lower")


def comments(args: argparse.Namespace) -> Dict:
    """Comments for the first rounds x per-round labels of the seeded order; prints a text-free report."""
    _require_corpus_ready(args.corpus, "comment generation")
    from . import unified_labeler as labeler
    from .unified_spend import SpendLedger
    from .unified_splits import DEV_SLICE_SIZE

    if not args.live and (args.confirm or args.spend_ledger or args.max_calls is not None):
        raise UsageError("live-only options were given without --live; refusing to guess")
    if args.model and not (args.live or args.replay):
        raise UsageError("--model names a real labeler; offline runs use the fake (add --replay or --live)")
    if args.live:
        check_comment_gates(args, 0)   # every static gate, before any data is read
    identity = unified_env.verify_clone(args.clone)
    splits = get_corpus(args.corpus).load(Path(identity.path) / "fixtures", dev_size=DEV_SLICE_SIZE)
    order = get_corpus(args.corpus).label_order_fn(splits, args.seed)[:args.rounds * args.per_round]
    items = [labeler.LabelerItem(i, splits.items[i].text, splits.items[i].reference_label) for i in order]
    model = args.model or (labeler.DEFAULT_MODEL if (args.live or args.replay) else labeler.FAKE_MODEL)
    cache = labeler.CommentCache(args.cache)
    ledger = None
    if args.live:
        needed = sum(labeler.wants_comment(i.item_id, args.seed)
                     and cache.get(labeler.cache_key(i, labeler.labeler_prompt(i), model)) is None for i in items)
        check_comment_gates(args, needed)
        ledger = SpendLedger(args.spend_ledger, labeler.LABELER_CALL_CEILING, max_new=args.max_calls,
                             max_consecutive_failures=5, run_label=f"labeler-seed{args.seed}")
        complete = labeler.openai_completion(model)
    else:
        unified_env.install_network_guard()
        complete = labeler.refusing_completion if args.replay else labeler.fake_completion
    found, report = labeler.generate(items, complete, model=model, cache=cache, seed=args.seed, ledger=ledger)
    labeler.write_comments(args.out, found)
    report["upper_bound_calls"] = labeler.upper_bound_calls(items, seed=args.seed)
    if ledger is not None:
        report["ledger"] = {k: v for k, v in ledger.summary().items() if k in ("ceiling", "cumulative_used", "new_this_invocation")}
    return report


def compact(summary: Dict) -> Dict:
    """The few numbers worth printing: metrics and requests per arm per round."""
    if summary.get("final_d"):
        return {"mode": summary["mode"], "slice": summary["evaluation_slice"]["name"], "final": True, "arm": "D",
                "retriever": summary["retrieval"]["variant"],
                "context": summary["retrieval"]["context_fingerprint"][:12],
                **{k: summary["metrics"][k] for k in ("accuracy", "brier", "ece")}, "unscored": summary["unscored"],
                "comparison": {arm: {k: m[k] for k in ("accuracy", "brier", "ece")}
                               for arm, m in summary["comparison"]["arms"].items()},
                "contrasts": {name: {m: [v["effect"], v["lower"], v["upper"]] for m, v in by_metric.items()}
                              for name, by_metric in summary["comparison"]["contrasts"].items()},
                "embedder": summary["retrieval"]["embedder"],
                "requests_this_invocation": summary["requests"]["new_this_invocation"],
                "upper_bound_requests": summary.get("upper_bound_requests")}
    if summary.get("final"):
        entry = summary["per_round"][0]
        return {"mode": summary["mode"], "slice": summary["evaluation_slice"]["name"], "final": True,
                "arms": {arm: {**{k: v["metrics"][k] for k in ("accuracy", "brier", "ece")},
                               "bundle": v["bundle"]["bundle_hash"][:12], "requests": v["requests"],
                               "unscored": v["unscored"]} for arm, v in entry["arms"].items()},
                "contrasts": {name: {m: [v["effect"], v["lower"], v["upper"]] for m, v in by_metric.items()}
                              for name, by_metric in entry["contrasts"].items()},
                "requests_this_invocation": summary["requests"]["new_this_invocation"],
                "upper_bound_requests": summary.get("upper_bound_requests")}
    rounds = []
    for entry in summary["per_round"]:
        rounds.append({
            "round": entry["round"], "n_labeled": entry["n_labeled"],
            "arms": {arm: {**{k: entry["arms"][arm]["metrics"][k] for k in ("accuracy", "brier", "ece")},
                           "gate": (entry["arms"][arm].get("gate") or entry["arms"][arm].get("workspace_gate") or {}).get("decision"),
                           "steer": (entry["arms"][arm].get("steering") or {}).get("decision"),
                           "features": len(entry["arms"][arm]["head_features"])}
                     for arm in entry["arms"]},
            "contrasts": {name: {m: [v["effect"], v["lower"], v["upper"]] for m, v in by_metric.items()}
                          for name, by_metric in entry["contrasts"].items()},
        })
    requests = summary["requests"]
    return {"mode": summary["mode"], "slice": summary["evaluation_slice"]["name"], "rounds": rounds,
            "requests_by_arm": requests["by_arm"], "requests_this_invocation": requests["new_this_invocation"],
            "upper_bound_requests": summary.get("upper_bound_requests")}


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-clone":
            identity = unified_env.prepare_clone(args.source, args.clone)
            print(json.dumps(identity.__dict__))
            return 0
        if args.command == "check-env":
            report = {"decision_flywheel": unified_env.ensure_decision_flywheel(),
                      "jev_flywheel_clone": unified_env.verify_clone().__dict__,
                      "scikit_learn_and_tactus": unified_env.jev_dependencies_available()}
            print(json.dumps(report, sort_keys=True))
            return 0 if report["scikit_learn_and_tactus"] else 2
        if args.command == "comments":
            print(json.dumps(comments(args), indent=1, sort_keys=True))
            return 0
        summary = run(args)
    except (UsageError, unified_env.EnvironmentProblem) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(compact(summary), indent=1, sort_keys=True))
    return 0
