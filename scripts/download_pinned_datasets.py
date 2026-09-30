#!/usr/bin/env python3
"""Explicitly acquire exact public dataset revisions into an ignored local cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_flywheel_evaluations.download import acquire_pinned_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download",), help="explicit network-enabled pinned dataset acquisition")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".data" / "huggingface")
    parser.add_argument("--confirm", action="store_true", help="authorize this one public dataset download")
    args = parser.parse_args(argv)
    if args.confirm is not True:
        print(json.dumps({"status": "confirmation_required"}, sort_keys=True))
        return 2
    try:
        acquired = acquire_pinned_datasets(cache_root=args.cache_root, confirmed=args.confirm)
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "reason": type(error).__name__}, sort_keys=True))
        return 1
    for item in acquired:
        print(json.dumps({"status": "cached", "dataset": item.dataset, "config": item.config,
                          "split": item.split, "revision": item.revision, "rows": item.rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
