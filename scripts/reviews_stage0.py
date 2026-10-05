#!/usr/bin/env python3
"""Cached-first Stage 0 report and explicitly confirmed native Jev resume command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from decision_flywheel_evaluations.unified_reviews_manifest import read_reviews_manifest
from decision_flywheel_evaluations.unified_reviews_stage0 import (
    FutureStageCaps, Stage0Caps, build_stage0_plan, cached_agreement_from_sqlite, cached_stage0,
    read_screen_cache, run_live_stage0,
)


def _records(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("cache input must be a JSON array of text-free records")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="cached-first, text-free reviews Stage 0 evidence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True, help="text-free JSON identity")
    parser.add_argument("--screen-cache", type=Path, required=True, help="durable text-free Stage 0 cache")
    agreement = parser.add_mutually_exclusive_group(required=True)
    agreement.add_argument("--agreement-cache", type=Path, help="text-free agreement JSON array")
    agreement.add_argument("--sme-cache", type=Path, help="existing SQLite cache; opened read-only")
    parser.add_argument("--second-sme-model")
    parser.add_argument("--report", type=Path, required=True, help="text-free output JSON")
    parser.add_argument("--merge-attempt", type=int, choices=(0, 1), default=0)
    parser.add_argument("--jev-ceiling", type=int, required=True)
    parser.add_argument("--jev-max-new", type=int, required=True)
    parser.add_argument("--sme-ceiling", type=int, required=True)
    parser.add_argument("--sme-max-new", type=int, required=True)
    parser.add_argument("--stage1-ceiling", type=int, required=True)
    parser.add_argument("--stage1-max-new", type=int, required=True)
    parser.add_argument("--stage2-ceiling", type=int, required=True)
    parser.add_argument("--stage2-max-new", type=int, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm", "--confirm-live", action="store_true")
    parser.add_argument("--jev-ledger", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--s-path", type=Path)
    parser.add_argument("--f-path", type=Path)
    parser.add_argument("--jev-model")
    args = parser.parse_args(argv)
    if args.confirm and not args.live:
        parser.error("--confirm requires --live")
    if args.live and not args.confirm:
        parser.error("--live requires --confirm before credentials or private inputs are considered")
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    manifest = read_reviews_manifest(args.manifest)
    plan = build_stage0_plan(manifest, identity)
    screen_rows = read_screen_cache(args.screen_cache, plan, model=identity["jev_model"])
    agreement_rows = (_records(args.agreement_cache) if args.agreement_cache else
                      cached_agreement_from_sqlite(manifest, identity, cache_path=args.sme_cache,
                                                   second_model=args.second_sme_model))
    caps = Stage0Caps(args.jev_ceiling, args.jev_max_new, args.sme_ceiling, args.sme_max_new)
    future = FutureStageCaps(args.stage1_ceiling, args.stage1_max_new, args.stage2_ceiling, args.stage2_max_new)
    if not args.live:
        outcome = cached_stage0(manifest=manifest, identity=identity, screen_cache=screen_rows,
                                agreement_cache=agreement_rows, caps=caps, future_caps=future,
                                merge_attempt=args.merge_attempt, source_verified=False)
    else:
        required = {"--jev-ledger": args.jev_ledger, "--pool": args.pool, "--source-manifest": args.source_manifest,
                    "--s-path": args.s_path, "--f-path": args.f_path, "--jev-model": args.jev_model}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("live Stage 0 requires " + ", ".join(missing))
        def factory(ledger):
            from decision_flywheel_evaluations.unified_reviews_stage0_live import build_stage0_live_collector
            collector = build_stage0_live_collector(manifest, pool_path=args.pool,
                source_manifest_path=args.source_manifest, s_path=args.s_path, f_path=args.f_path,
                model=args.jev_model, ledger=ledger)
            collector.validate_tasks()
            def collect(condition, item_id):
                return collector.collect(condition, item_id)
            return collect, collector.task_fingerprints()
        outcome = run_live_stage0(manifest=manifest, identity=identity, screen_cache=screen_rows,
                                  agreement_cache=agreement_rows, caps=caps, future_caps=future,
                                  merge_attempt=args.merge_attempt, confirm_live=True, live_factory=factory,
                                  jev_ledger_path=args.jev_ledger, screen_cache_path=args.screen_cache,
                                  model=args.jev_model,
                                  # The collector revalidates the private pool, source manifest, and policy
                                  # before it can make a request; a live artifact is therefore source-verified.
                                  source_verified=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
