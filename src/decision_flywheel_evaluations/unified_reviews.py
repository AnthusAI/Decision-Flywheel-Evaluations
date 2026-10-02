"""The Amazon review moderation corpus for the unified flywheel and the S-vs-F screen (streaming SME plan, step 3).

Gold labels are the simulated SME's (``unified_sme``), read READ-ONLY from its sqlite cache; review text
comes from the private pool file ``var/amazon-reviews/pool.jsonl``. Nothing here calls a model.

* **Labels** are in the fixed order ``approve, abusive, promotional, seller_shipping, price_availability``
  (the last is the few-shot reference label). If the SME-labeled pool holds fewer than
  ``MERGE_MIN_ABUSIVE`` (20) abusive cases, abusive and promotional are scored together as ``remove_other``
  (``should_merge``): the switch is decided from the data, and the two registered corpora (``reviews`` and
  ``reviews-merged``) refuse data that decided the other way.
* **Gold** excludes every SME record without a label (``ambiguous``/UNEXPLAINABLE, or rejected) and every
  pool item the SME has not labeled; the report counts each.
* **Splits.** Held-out 300 (FINAL, the scoreboard) and stream 600 (the labeled side) are drawn together,
  stratified by SME label: every class gets at least ``MINORITY_FLOOR`` (10%) of the 900 where the pool has
  that many, the rest follow the pool's mix, and each class is cut 1:2 between held-out and stream. A
  dev-100 comes from the remaining labeled items (pool mix). Fixed seed; disjoint; the report gives the
  achieved mix of each slice and, separately, the SME labels of the natural-mix sample (if labeled).
* **S** is ``studies/amazon_reviews/S.txt``; **F** (ceiling and stakeholder guideline) is the private
  ``var/policy/amazon_reviews_F.txt``, read only when a ceiling or live stakeholder needs it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .unified_emotion import assert_no_duplicate_text, stratified_label_order
from .unified_sme import PROMPT_VERSION, SmeCache, SmeRecord, cache_key
from .unified_splits import CorpusItem, Splits, assert_partitions_disjoint, fingerprint_ids

REVIEWS_LABELS = ("approve", "abusive", "promotional", "seller_shipping", "price_availability")
MERGED_LABEL = "remove_other"
MERGED_FROM = ("abusive", "promotional")
MERGED_LABELS = ("approve", MERGED_LABEL, "seller_shipping", "price_availability")
MERGE_MIN_ABUSIVE = 20
REVIEWS_TASK_NAME = "reviews"
REVIEWS_FINAL_SIZE = 300
REVIEWS_STREAM_SIZE = 600
REVIEWS_DEV_SIZE = 100
MINORITY_FLOOR = 0.10
SPLIT_SEED = "reviews-splits-v1"
LABEL_ORDER_SEED = "unified-flywheel-reviews-label-order-v1"

ROOT = Path(__file__).resolve().parents[2]
S_PATH = ROOT / "studies" / "amazon_reviews" / "S.txt"
POLICY_PATH = ROOT / "var" / "policy" / "amazon_reviews_F.txt"   # private; never committed
POOL_PATH = ROOT / "var" / "amazon-reviews" / "pool.jsonl"
NATURAL_PATH = ROOT / "var" / "amazon-reviews" / "natural.jsonl"
SME_CACHE_PATH = ROOT / "var" / "amazon-reviews" / "sme-cache.sqlite"
SPLITS_OUT = ROOT / "studies" / "amazon_reviews" / "splits.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# ---- labels, S and F --------------------------------------------------------------------------

def reviews_labels(merged: bool = False) -> Tuple[str, ...]:
    return MERGED_LABELS if merged else REVIEWS_LABELS


def should_merge(label_counts: Mapping[str, int], *, threshold: int = MERGE_MIN_ABUSIVE) -> bool:
    """The plan's class merge: abusive joins promotional as ``remove_other`` when it has too few cases."""
    return label_counts.get("abusive", 0) < threshold


def gold_label(label: str, merged: bool) -> str:
    return MERGED_LABEL if merged and label in MERGED_FROM else label


def _merge_note() -> str:
    return f"Label a review that is {' or '.join(MERGED_FROM)} as {MERGED_LABEL}."


def starting_rubric(merged: bool = False, path: Path = S_PATH) -> str:
    """S, exactly the committed one-liner; the merged corpus names ``remove_other`` in place of the two classes."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not merged:
        return text
    old = ", ".join(MERGED_FROM)
    if old not in text:
        raise ValueError("S does not list abusive, promotional together; cannot word the merged rubric")
    return text.replace(old, f"{MERGED_LABEL} ({' or '.join(MERGED_FROM)})")


def guideline_text(merged: bool = False, path: Optional[Path] = None) -> str:
    """F: the private written policy (ceiling question and stakeholder guideline). Needed only for those."""
    path = Path(path) if path is not None else POLICY_PATH
    if not path.is_file():
        raise FileNotFoundError(f"the private policy F is missing at {path}; only the ceiling and stakeholder need it")
    text = path.read_text(encoding="utf-8").strip()
    return f"{text}\n\n{_merge_note()}" if merged else text


def _holistic_config(labels: Sequence[str], instructions: str) -> dict:
    features = [f"self.holistic.clr.{label}" for label in labels[:-1]]
    weights = {label: {feature: 1.0} for label, feature in zip(labels[:-1], features)}
    weights[labels[-1]] = {feature: -1.0 for feature in features}
    return {"name": REVIEWS_TASK_NAME, "key": REVIEWS_TASK_NAME, "question_type": "choice",
            "instructions": instructions, "criteria": {label: None for label in labels},
            "decision": {"model": "multinomial_logistic", "classes": list(labels), "features": features,
                         "parameters": {"weights": weights}}}


def seed_score_config(merged: bool = False) -> dict:
    return _holistic_config(reviews_labels(merged), starting_rubric(merged))


def ceiling_score_config(merged: bool = False) -> dict:
    return _holistic_config(reviews_labels(merged), guideline_text(merged))


# ---- the SME cache, read-only -----------------------------------------------------------------

class ReadOnlySmeCache(SmeCache):
    """``SmeCache`` opened ``mode=ro``: never creates, locks for writing or alters the live labeling cache."""

    def __init__(self, path):  # noqa: D107 - deliberately does not call SmeCache.__init__ (it creates the table)
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no SME cache at {path}")
        self.db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)

    def identities(self) -> List[Tuple[str, str]]:
        """(model, policy sha256) pairs present for the primary pass at this prompt version."""
        return [tuple(row) for row in self.db.execute(
            "SELECT DISTINCT model, policy_sha256 FROM sme WHERE pass_tag = 'primary' AND prompt_version = ? "
            "ORDER BY model, policy_sha256", (PROMPT_VERSION,))]


def read_sme_records(cache_path, item_ids: Sequence[str], *, model: Optional[str] = None,
                     policy_sha256: Optional[str] = None) -> Tuple[Dict[str, SmeRecord], Dict[str, str]]:
    """Primary-pass SME records for ``item_ids`` (uncached items are absent) and the (model, policy) read."""
    cache = ReadOnlySmeCache(cache_path)
    try:
        found = [(m, s) for m, s in cache.identities()
                 if (model is None or m == model) and (policy_sha256 is None or s == policy_sha256)]
        if len(found) != 1:
            raise ValueError(f"expected one SME (model, policy) in the cache, found {len(found)}; "
                             "name --sme-model / the policy explicitly")
        model, sha = found[0]
        records = {}
        for item_id in item_ids:
            record = cache.get(cache_key(item_id, sha, model))
            if record is not None:
                records[item_id] = record
        return records, {"sme_model": model, "policy_sha256": sha}
    finally:
        cache.db.close()


def check_policy_matches(policy_sha256: str, path: Optional[Path] = None) -> Optional[bool]:
    """True/False when the private F exists (its sha must match the SME's), None when it is absent."""
    path = Path(path) if path is not None else POLICY_PATH
    if not path.is_file():
        return None
    return _sha(path.read_text(encoding="utf-8")) == policy_sha256


# ---- splits -----------------------------------------------------------------------------------

def floor_quotas(available: Mapping[str, int], size: int, labels: Sequence[str], *,
                 floor: float = MINORITY_FLOOR) -> Dict[str, int]:
    """Per-label counts summing to ``size``: the pool's mix, with every label raised to ``floor`` of ``size``
    where available. Items move one at a time from the label with most spare above its own floor."""
    total = sum(available.get(label, 0) for label in labels)
    if total < size:
        raise ValueError(f"only {total} SME-labeled items for {size} held-out + stream places")
    raw = {label: size * available.get(label, 0) / total for label in labels}
    quota = {label: int(raw[label]) for label in labels}
    for label in sorted(labels, key=lambda l: (-(raw[l] - quota[l]), labels.index(l)))[:size - sum(quota.values())]:
        quota[label] += 1
    need = math.ceil(round(floor * size, 6))
    target = {label: min(available.get(label, 0), need) for label in labels}
    while True:
        short = [label for label in labels if quota[label] < target[label]]
        if not short:
            return quota
        donor = max(labels, key=lambda l: (quota[l] - target[l], -labels.index(l)))
        if quota[donor] - target[donor] <= 0:
            return quota
        quota[donor] -= 1
        quota[short[0]] += 1


def _largest_remainder(counts: Mapping[str, int], size: int, labels: Sequence[str]) -> Dict[str, int]:
    total = sum(counts.values())
    if total <= size:
        return {label: counts.get(label, 0) for label in labels}
    raw = {label: size * counts.get(label, 0) / total for label in labels}
    out = {label: int(raw[label]) for label in labels}
    for label in sorted(labels, key=lambda l: (-(raw[l] - out[l]), labels.index(l)))[:size - sum(out.values())]:
        out[label] += 1
    return out


def build_reviews_splits(pool_rows: Sequence[Mapping[str, object]], records: Mapping[str, SmeRecord], *,
                         merge: Optional[bool] = None, final_size: int = REVIEWS_FINAL_SIZE,
                         stream_size: int = REVIEWS_STREAM_SIZE, dev_size: int = REVIEWS_DEV_SIZE,
                         seed: str = SPLIT_SEED, floor: float = MINORITY_FLOOR,
                         natural_records: Optional[Mapping[str, SmeRecord]] = None,
                         natural_size: int = 0) -> Tuple[Splits, dict]:
    """Splits (pool = stream, FINAL = held-out, dev) and a text-free report. ``merge=None`` decides from data."""
    texts = {str(r["id"]): str(r["text"]) for r in pool_rows}
    if len(texts) != len(pool_rows):
        raise ValueError("duplicate review id in the pool")
    status = _count(records[i].status if i in records else "unlabeled" for i in texts)
    gold_raw = {i: records[i].label for i in sorted(texts) if i in records and records[i].label is not None}
    raw_counts = _count(gold_raw.values())
    merged = should_merge(raw_counts) if merge is None else bool(merge)
    labels = reviews_labels(merged)
    gold = {i: gold_label(label, merged) for i, label in gold_raw.items()}
    by_label: Dict[str, List[str]] = {label: [] for label in labels}
    for item_id, label in gold.items():
        by_label[label].append(item_id)
    for label in labels:
        random.Random(f"{seed}:{label}").shuffle(by_label[label])
    combined = floor_quotas({l: len(v) for l, v in by_label.items()}, final_size + stream_size, labels, floor=floor)
    held_quota = _largest_remainder(combined, final_size, labels)
    heldout, stream, rest = [], [], []
    for label in labels:
        ids, h, q = by_label[label], held_quota[label], combined[label]
        heldout += ids[:h]
        stream += ids[h:q]
        rest += ids[q:]
    rest_counts = _count(gold[i] for i in rest)
    dev_quota = _largest_remainder(rest_counts, dev_size, labels)
    dev = [i for label in labels for i in [x for x in rest if gold[x] == label][:dev_quota[label]]]
    items = {i: CorpusItem(i, "pool", gold[i], texts[i]) for i in stream}
    items.update({i: CorpusItem(i, "test", gold[i], texts[i]) for i in (*heldout, *dev)})
    splits = Splits(items, tuple(sorted(stream)), tuple(sorted((*heldout, *dev))), tuple(sorted(heldout)),
                    tuple(sorted(dev)), final_name=f"heldout-{len(heldout)}")
    assert_partitions_disjoint(splits)
    assert_no_duplicate_text(splits)

    def mix(ids):
        counts = _count(gold[i] for i in ids)
        return {"n": len(ids), "by_label": counts,
                "share": {l: round(counts.get(l, 0) / len(ids), 4) for l in labels} if ids else {}}

    natural = None
    if natural_records is not None:
        labeled = [r.label for r in natural_records.values() if r.label is not None]
        natural = {"n": natural_size, "sme_labeled": len(labeled),
                   "by_label": _count(gold_label(l, merged) for l in labeled),
                   "no_label": len(natural_records) - len(labeled)}
    report = {
        "seed": seed, "labels": list(labels), "merged": merged, "merge_rule": f"abusive < {MERGE_MIN_ABUSIVE}",
        "pool_items": len(texts), "sme_status": status,
        "excluded_from_gold": {"ambiguous": status.get("ambiguous", 0), "rejected": status.get("rejected", 0),
                               "unlabeled": status.get("unlabeled", 0)},
        "gold_by_raw_label": raw_counts, "gold_by_label": _count(gold.values()),
        "minority_floor": floor, "floor_met": {l: combined[l] >= floor * (final_size + stream_size) for l in labels},
        "heldout": mix(heldout), "stream": mix(stream), "dev": mix(dev), "natural_mix": natural,
        "fingerprints": {"heldout": fingerprint_ids(heldout), "stream": fingerprint_ids(stream),
                         "dev": fingerprint_ids(dev)},
    }
    return splits, report


def _read_jsonl(path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_reviews_splits(*, merge: Optional[bool] = None, pool_path=None, natural_path=None, cache_path=None,
                        sme_model: Optional[str] = None, dev_size: int = REVIEWS_DEV_SIZE) -> Tuple[Splits, dict]:
    """The real splits: private pool text + the SME cache (read-only). Refuses if F exists and disagrees."""
    pool = _read_jsonl(pool_path or POOL_PATH)
    records, identity = read_sme_records(cache_path or SME_CACHE_PATH, [str(r["id"]) for r in pool], model=sme_model)
    if check_policy_matches(identity["policy_sha256"]) is False:
        raise ValueError("the SME labels were made under a different policy F than var/policy/amazon_reviews_F.txt")
    natural_path = Path(natural_path or NATURAL_PATH)
    natural_records, natural_size = None, 0
    if natural_path.is_file():
        natural_ids = [str(r["id"]) for r in _read_jsonl(natural_path)]
        natural_records, _ = read_sme_records(cache_path or SME_CACHE_PATH, natural_ids, model=identity["sme_model"],
                                              policy_sha256=identity["policy_sha256"])
        natural_size = len(natural_ids)
    splits, report = build_reviews_splits(pool, records, merge=merge, dev_size=dev_size,
                                          natural_records=natural_records, natural_size=natural_size)
    report.update(identity)
    report["policy_path"] = str(POLICY_PATH.relative_to(ROOT))
    return splits, report


def load_reviews_corpus(fixtures: Path, *, dev_size: int = REVIEWS_DEV_SIZE, merged: bool = False) -> Splits:
    """The ``Corpus.load_splits`` hook (``fixtures`` unused). The data's merge decision must match the corpus."""
    splits, report = load_reviews_splits(dev_size=dev_size)
    if report["merged"] != merged:
        want = "reviews-merged" if report["merged"] else "reviews"
        raise ValueError(f"the SME labels decide merged={report['merged']} ({report['merge_rule']}); use --corpus {want}")
    if len(splits.paper600) != REVIEWS_FINAL_SIZE:
        raise ValueError(f"held-out has {len(splits.paper600)} items, not {REVIEWS_FINAL_SIZE}")
    return splits


def reviews_label_order(splits: Splits, seed: int, *, merged: bool = False) -> Tuple[str, ...]:
    return stratified_label_order(splits, seed, reviews_labels(merged), order_seed=LABEL_ORDER_SEED)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin CLI over load_reviews_splits
    parser = argparse.ArgumentParser(description="Build the reviews held-out/stream/dev splits from the SME cache (read-only)")
    parser.add_argument("--sme-model", default=None)
    parser.add_argument("--merge", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--out", type=Path, default=None, help=f"also write the text-free report + ids (e.g. {SPLITS_OUT.relative_to(ROOT)})")
    args = parser.parse_args(argv)
    splits, report = load_reviews_splits(merge={"auto": None, "yes": True, "no": False}[args.merge], sme_model=args.sme_model)
    print(json.dumps(report, indent=1, sort_keys=True))
    if args.out is not None:
        ids = {"heldout": list(splits.paper600), "stream": list(splits.pool), "dev": list(splits.dev100)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**report, "ids": ids}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0
