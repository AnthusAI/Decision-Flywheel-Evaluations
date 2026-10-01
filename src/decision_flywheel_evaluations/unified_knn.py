"""Lever B-local: labeled neighbours as head features (plan section 2, step 4).

For each item, the k=8 nearest labeled neighbours by the *same* lexical overlap that
Decision-Flywheel's ``PerLabelLexicalRetrieval`` ranks with (cosine-normalized overlap of
lowercase alphabetic token sets), **without** balancing by label. The head then gets:

* ``knn.share.positive`` -- the similarity-weighted share of positive labels among the k.
  ``knn.share.negative`` is ``1 - share.positive`` for two labels, so it is omitted: it would
  be exactly collinear and only split the regularized weight.
* ``knn.top4.<label>`` -- the mean similarity of the 4 most similar labeled items carrying
  each label, searched over the whole labeled pool (not only the k).

A label-balanced pick (4 per label) would make the shares constant, which is why the
neighbours are unbalanced. The same firewall as the retrieval policy applies: the target is
excluded by ID and by normalized text, so a labeled item's features are leave-one-out.

The cache key for anything that depends on the pool is a *context fingerprint*: the policy's
fingerprint plus the sorted labeled IDs (plan section 2, "Compatibility").
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from decision_flywheel.context import PerLabelLexicalRetrieval, _tokens

from .unified_splits import LABELS, assert_target_excluded

KNN_POLICY = {"name": "knn-lexical-unbalanced", "version": "1", "k": 8, "top": 4,
              "similarity": "per-label-lexical-retrieval cosine token-set overlap"}
SHARE_FEATURE = "knn.share.positive"
TOP_FEATURES = tuple(f"knn.top4.{label}" for label in LABELS)
KNN_FEATURES = (SHARE_FEATURE, *TOP_FEATURES)


@dataclass(frozen=True)
class PoolEntry:
    id: str
    text: str
    label: str


def normalized_text(text: str) -> str:
    """The firewall's normalization, used only to exclude an exact duplicate of the target."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def similarity(target_text: str, other_text: str) -> float:
    """Exactly ``PerLabelLexicalRetrieval``'s score."""
    target, other = _tokens(target_text), _tokens(other_text)
    return len(target & other) / math.sqrt(max(1, len(target)) * max(1, len(other)))


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def knn_policy_fingerprint(k: int = 8, top: int = 4) -> str:
    return _fingerprint({**KNN_POLICY, "k": k, "top": top})


def retrieval_policy_fingerprint() -> str:
    return PerLabelLexicalRetrieval().fingerprint


def context_fingerprint(policy_fingerprint: str, labeled_ids: Sequence[str], **extra: object) -> str:
    """Policy plus the sorted labeled IDs: what a pool-dependent answer or feature depends on."""
    return _fingerprint({"policy": policy_fingerprint, "labeled_ids": sorted(labeled_ids), **extra})


def ranked_neighbours(target_id: str, target_text: str,
                      pool: Sequence[PoolEntry]) -> List[Tuple[float, PoolEntry]]:
    target_norm = normalized_text(target_text)
    ranked = [(similarity(target_text, entry.text), entry) for entry in pool
              if entry.id != target_id and normalized_text(entry.text) != target_norm]
    ranked.sort(key=lambda pair: (-pair[0], pair[1].id))
    return ranked


def knn_features(target_id: str, target_text: str, pool: Sequence[PoolEntry], *,
                 k: int = 8, top: int = 4) -> Tuple[Dict[str, float], Tuple[str, ...]]:
    """The B-local features for one item, and the IDs of the k neighbours used."""
    ranked = ranked_neighbours(target_id, target_text, pool)
    if len(ranked) < k:
        raise ValueError(f"only {len(ranked)} labeled neighbours; k={k} needs more labels")
    nearest = ranked[:k]
    assert_target_excluded(target_id, [entry.id for _, entry in nearest])
    total = sum(sim for sim, _ in nearest)
    if total > 0:
        share = sum(sim for sim, entry in nearest if entry.label == LABELS[0]) / total
    else:  # no lexical overlap at all: fall back to the unweighted vote of the k
        share = sum(1 for _, entry in nearest if entry.label == LABELS[0]) / k
    features = {SHARE_FEATURE: share}
    for label, name in zip(LABELS, TOP_FEATURES):
        sims = [sim for sim, entry in ranked if entry.label == label][:top]
        features[name] = sum(sims) / len(sims) if sims else 0.0
    return features, tuple(entry.id for _, entry in nearest)


def knn_rows(targets: Mapping[str, str], pool: Sequence[PoolEntry], *, k: int = 8,
             top: int = 4) -> Dict[str, Dict[str, float]]:
    """Features for many targets against one pool (leave-one-out for pool members)."""
    return {target_id: knn_features(target_id, text, pool, k=k, top=top)[0]
            for target_id, text in targets.items()}
