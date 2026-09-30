#!/usr/bin/env python3
"""Explicitly prepare cached pinned public datasets into text-free study manifests.

Run only after the offline test suite passes:
    PYTHONPATH=src python scripts/prepare_datasets.py prepare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Preparation is strictly cache-only: a missing pinned dataset is a blocker,
# never a reason to contact a provider from this command.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    from datasets.utils import logging as _datasets_logging
except ImportError:  # The loader gives the actionable missing-dependency failure.
    _datasets_logging = None
else:
    _datasets_logging.set_verbosity_error()

from decision_flywheel_evaluations.datasets import AG_NEWS, EMOTION, load_huggingface_split
from decision_flywheel_evaluations.manifests import write_manifest
from decision_flywheel_evaluations.preparation import (AG_NEWS_PLAN, EMOTION_PLAN,
                                                        HISTORICAL_AG_NEWS_COMMIT,
                                                        prepare_pinned_study,
                                                        read_historical_ag_news_exposure)


HISTORICAL_REPOSITORY = ROOT.parent / "Jev-AGNews-Fewshot"


def _historical_exposure(repository: Path):
    """Resolve the reviewed inventory exactly; never downgrade freshness on failure."""
    completed = subprocess.run(["git", "-C", str(repository), "show",
                                f"{HISTORICAL_AG_NEWS_COMMIT}:manifests/scoreboard.jsonl"],
                               check=True, capture_output=True)
    return read_historical_ag_news_exposure(completed.stdout, commit=HISTORICAL_AG_NEWS_COMMIT)


def _emit(*, dataset: str, status: str, **details: object) -> None:
    """Print only counts, digests, and status—never dataset text."""
    print(json.dumps({"dataset": dataset, "status": status, **details}, sort_keys=True))


def _write(destination: Path, manifest) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(destination, manifest)
    _emit(dataset=manifest.dataset, status="prepared",
          manifest_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
          counts=manifest.counts, exposure_status=manifest.exposure_status.value,
          preparation_counts=manifest.preparation.sample_counts if manifest.preparation else {})


def _prepare_ag_news(*, output_dir: Path, raw_cache: Path, historical_repository: Path) -> bool:
    try:
        history = _historical_exposure(historical_repository)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        _emit(dataset=AG_NEWS.name, status="blocked", reason=type(error).__name__)
        return False
    train = load_huggingface_split(AG_NEWS, "train", cache_dir=str(raw_cache))
    official = load_huggingface_split(AG_NEWS, "test", cache_dir=str(raw_cache))
    _write(output_dir / "ag_news.json", prepare_pinned_study(AG_NEWS, train, official, AG_NEWS_PLAN, history=history))
    return True


def _prepare_emotion(*, output_dir: Path, raw_cache: Path) -> bool:
    train = load_huggingface_split(EMOTION, "train", cache_dir=str(raw_cache))
    official = load_huggingface_split(EMOTION, "test", cache_dir=str(raw_cache))
    _write(output_dir / "emotion.json", prepare_pinned_study(EMOTION, train, official, EMOTION_PLAN))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",), help="explicitly prepare pinned cached data")
    parser.add_argument("--dataset", choices=("all", "ag_news", "emotion"), default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "studies" / "manifests")
    parser.add_argument("--raw-cache", type=Path, default=ROOT / ".data" / "huggingface")
    parser.add_argument("--historical-repository", type=Path, default=HISTORICAL_REPOSITORY)
    args = parser.parse_args(argv)
    try:
        succeeded = True
        if args.dataset in {"all", "ag_news"}:
            succeeded = _prepare_ag_news(output_dir=args.output_dir, raw_cache=args.raw_cache,
                                         historical_repository=args.historical_repository) and succeeded
        if args.dataset in {"all", "emotion"}:
            succeeded = _prepare_emotion(output_dir=args.output_dir, raw_cache=args.raw_cache) and succeeded
    except (OSError, RuntimeError, ValueError) as error:
        _emit(dataset=args.dataset, status="failed", reason=type(error).__name__)
        return 1
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
