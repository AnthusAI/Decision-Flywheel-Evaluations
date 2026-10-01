"""Command line for the unified-flywheel harness (``scripts/unified_flywheel.py``).

Offline is the default and the only mode the specs exercise: a deterministic fake Jev
behind the same counting clients, the analyst replaced by fixed replies through
``jev_flywheel.steer``'s ``mock_replies``, and a process-wide guard that blocks every
non-local socket.

Live mode mirrors the repository's collector: it needs ``--live`` *and* ``--confirm``, a
request ceiling no greater than the plan's 9,500, a per-invocation cap at least as large as
the run's precomputed upper bound, a durable ``--spend-ledger``, and an explicit Jev model.
Every one of these is checked before any data is read or any client is constructed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import unified_env
from .unified_spend import request_upper_bound

UNIFIED_ARMS = ("0", "A", "B-local", "B", "A+B")
ARMS = UNIFIED_ARMS + ("F", "F-rand", "A-c", "A-c+F")
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
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--clone", type=Path, default=None)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--arms", default=",".join(UNIFIED_ARMS), help="comma-separated subset of " + ",".join(ARMS))
    run.add_argument("--comments", type=Path, default=None,
                     help="JSONL of {item_id, comment} from an explanation labeler, for arms A-c and A-c+F")
    run.add_argument("--final", action="store_true",
                     help="score paper-600 instead of dev-100 (final comparison only)")
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
    return top


def parse_arms(text: str) -> tuple:
    arms = tuple(a.strip() for a in text.split(",") if a.strip())
    unknown = set(arms) - set(ARMS)
    if not arms or unknown:
        raise UsageError(f"unknown arms {sorted(unknown)}; choose from {','.join(ARMS)}")
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
    bound = request_upper_bound(arms, rounds=args.rounds, per_round=100, eval_n=eval_n)
    if bound["total"] > args.max_new_requests:
        raise UsageError(f"this run may need up to {bound['total']} requests ({bound}); "
                         f"--max-new-requests {args.max_new_requests} is lower. Raise it deliberately or run fewer arms/rounds")
    return bound


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


def run(args: argparse.Namespace) -> Dict:
    arms = parse_arms(args.arms)
    eval_n = 600 if args.final else 100
    bound = request_upper_bound(arms, rounds=args.rounds, per_round=100, eval_n=eval_n)
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
        clone=Path(identity.path), run_dir=Path(run_dir), seed=args.seed, rounds=args.rounds, arms=arms,
        final=args.final, live=args.live, replay=args.replay, request_ceiling=args.request_ceiling,
        max_new_requests=args.max_new_requests, spend_ledger=args.spend_ledger,
        max_concurrency=args.max_concurrency, max_consecutive_failures=args.max_consecutive_failures,
        analyst_provider=args.analyst_provider, analyst_model=args.analyst_model,
        analyst_replies=_offline_replies(args.analyst_replies), comments=_comments(args.comments),
        provider_model=args.provider_model or "fake-jev-0", base_url=args.base_url,
        timeout_seconds=args.timeout_seconds, bootstrap_resamples=args.bootstrap_resamples)
    summary = UnifiedFlywheel(cfg).run()
    summary["upper_bound_requests"] = bound
    return summary


def compact(summary: Dict) -> Dict:
    """The few numbers worth printing: metrics and requests per arm per round."""
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
        summary = run(args)
    except (UsageError, unified_env.EnvironmentProblem) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(compact(summary), indent=1, sort_keys=True))
    return 0
