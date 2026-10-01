"""Splits, slices and the fixed label order for the unified-flywheel study.

Plan section 3, made executable:

* the **pool** (5,280 items) is the only place labels come from;
* **paper-600** is the 600 test items Jev-Flywheel's recorded run scored (the test items
  that carry a recorded ``topic_domain`` answer, which is how ``scripts/finetune_laya.py``
  derives it), and it is reachable only through ``final=True``;
* **dev-100** is 100 test items *outside* paper-600, drawn with a fixed seed, and is the
  only slice iteration runs score;
* the **label order** is a seeded shuffle of the pool, cut into rounds of 100.

Nothing here reads a reference label for a held-out item except to *score* it, and the
assertions at the bottom are the leakage rules every caller runs before spending anything.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

DEV_SLICE_SEED = "unified-flywheel-dev100-v1"
DEV_SLICE_SIZE = 100
PAPER_SLICE_SIZE = 600
LABEL_ORDER_SEED = "unified-flywheel-label-order-v1"
RECORDED_EXTRA_ANSWERS = Path("recordings") / "simulated-labeler" / "extra_answers.jsonl.gz"
LABELS = ("positive", "negative")


class LeakageError(AssertionError):
    """A held-out item reached a place only labeled pool items may be."""


@dataclass(frozen=True)
class CorpusItem:
    id: str
    split: str
    reference_label: str
    text: str


@dataclass(frozen=True)
class Splits:
    items: Mapping[str, CorpusItem]
    pool: Tuple[str, ...]
    test: Tuple[str, ...]
    paper600: Tuple[str, ...]
    dev100: Tuple[str, ...]

    def held_out(self) -> frozenset:
        return frozenset(self.test)

    def evaluation_slice(self, *, final: bool) -> Tuple[str, Tuple[str, ...]]:
        """The slice a run scores. paper-600 only behind an explicit ``final``."""
        if final:
            return "paper-600", self.paper600
        return "dev-100", self.dev100


def fingerprint_ids(ids: Iterable[str]) -> str:
    """A text-free identifier for a set of item IDs."""
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def load_items(fixtures: Path) -> Dict[str, CorpusItem]:
    items: Dict[str, CorpusItem] = {}
    with (Path(fixtures) / "items.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            items[row["id"]] = CorpusItem(row["id"], str(metadata.get("split")),
                                          str(metadata.get("reference_label")), row["text"])
    return items


def paper600_ids(fixtures: Path, items: Mapping[str, CorpusItem]) -> Tuple[str, ...]:
    """The 600 test items of the recorded run, without replaying it."""
    ids = set()
    with gzip.open(Path(fixtures) / RECORDED_EXTRA_ANSWERS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["item_id"])
    paper = tuple(sorted(i for i in ids if items[i].split == "test"))
    if len(paper) != PAPER_SLICE_SIZE:
        raise ValueError(f"expected {PAPER_SLICE_SIZE} paper-600 items, found {len(paper)}")
    return paper


def dev_slice(test: Sequence[str], paper600: Sequence[str], *, size: int = DEV_SLICE_SIZE,
              seed: str = DEV_SLICE_SEED) -> Tuple[str, ...]:
    """``size`` test items outside paper-600, from a fixed seed, in sorted order."""
    excluded = set(paper600)
    candidates = sorted(i for i in test if i not in excluded)
    return tuple(sorted(random.Random(seed).sample(candidates, size)))


def load_splits(fixtures: Path, *, dev_size: int = DEV_SLICE_SIZE) -> Splits:
    items = load_items(fixtures)
    pool = tuple(sorted(i for i, item in items.items() if item.split == "pool"))
    test = tuple(sorted(i for i, item in items.items() if item.split == "test"))
    paper = paper600_ids(fixtures, items)
    splits = Splits(items, pool, test, paper, dev_slice(test, paper, size=dev_size))
    assert_partitions_disjoint(splits)
    return splits


def label_order(pool: Sequence[str], seed: int) -> Tuple[str, ...]:
    """The fixed, seeded order every arm labels in."""
    order = sorted(pool)
    random.Random(f"{LABEL_ORDER_SEED}:{seed}").shuffle(order)
    return tuple(order)


def round_batches(order: Sequence[str], *, rounds: int, per_round: int) -> List[Tuple[str, ...]]:
    if rounds * per_round > len(order):
        raise ValueError("not enough pool items for the requested rounds")
    return [tuple(order[r * per_round:(r + 1) * per_round]) for r in range(rounds)]


# ---- the leakage rules ------------------------------------------------------------------------

def assert_partitions_disjoint(splits: Splits) -> None:
    pool, test = set(splits.pool), set(splits.test)
    if pool & test:
        raise LeakageError("pool and test splits overlap")
    if set(splits.dev100) & set(splits.paper600):
        raise LeakageError("dev-100 overlaps paper-600")
    if not set(splits.dev100) <= test or not set(splits.paper600) <= test:
        raise LeakageError("evaluation slices must come from the test split")


def assert_labels_from_pool(labeled: Iterable[str], splits: Splits) -> None:
    """Rule 1: a labeled item is never a held-out item."""
    stray = set(labeled) - set(splits.pool)
    if stray:
        raise LeakageError(f"{len(stray)} labeled items are not pool items")


def assert_retrieval_pool_clean(pool_ids: Iterable[str], splits: Splits) -> None:
    """Rule 3: retrieval and kNN only ever draw from labeled pool items."""
    stray = set(pool_ids) & splits.held_out()
    if stray:
        raise LeakageError(f"{len(stray)} held-out items reached a retrieval pool")


def assert_target_excluded(target_id: str, context_ids: Iterable[str]) -> None:
    """Rule 2: an item is never its own example or neighbour."""
    if target_id in set(context_ids):
        raise LeakageError("a target was retrieved as its own context")
