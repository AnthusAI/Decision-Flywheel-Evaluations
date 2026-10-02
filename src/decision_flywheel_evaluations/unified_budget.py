"""The feature-budget cap: keep the final head within what the capability ladder allows.

Jev-Flywheel's ladder lets a head carry at most ``n / 5`` features at 30-199 effective labels and
``n / 10`` from 200 (``jev_flywheel.ladder.Tier.feature_budget``); a fit over that budget is REFUSED.
A multi-class choice element brings N-1 features, so steering can spend the budget at small label
counts. Jev's own check rejects a whole over-budget *proposal*; this cap is the harness's safety
net applied AFTER steering, so the head the arm finally fits never exceeds the budget.

The rule is the simplest one consistent with how steering orders proposals: elements keep the
order they were added in (earlier rounds first, within a proposal the analyst's own order, most
important first), so the cap drops from the END, one whole element at a time (all of its features),
until the head fits. The holistic element, and any feature that belongs to no element in the
scorecard (the few-shot and kNN features the arm adds itself), are never dropped. ``reserved`` counts
features the arm adds after steering. This module is pure: it plans over names and counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DroppedElement:
    key: str
    n_features: int
    new_this_round: bool


@dataclass(frozen=True)
class FeatureCapPlan:
    dropped: Tuple[DroppedElement, ...]
    features_before: int
    features_after: int
    reserved: int
    satisfied: bool

    def record(self, *, n_labeled: int, budget: int) -> Optional[Dict[str, Any]]:
        """The text-free summary entry (keys and counts only); ``None`` when nothing was dropped."""
        if not self.dropped:
            return None
        out: Dict[str, Any] = {
            "n_labeled": n_labeled, "budget": budget, "reserved_features": self.reserved,
            "features_before": self.features_before, "features_after": self.features_after,
            "reason": "over the capability-ladder feature budget",
            "dropped": [{"key": d.key, "features": d.n_features, "new_this_round": d.new_this_round}
                        for d in self.dropped]}
        if not self.satisfied:
            out["still_over_budget"] = True
        return out


def plan_feature_cap(features: Sequence[str], element_keys: Sequence[str], *, budget: int, reserved: int = 0,
                     protected: Collection[str] = (), new_keys: Collection[str] = ()) -> FeatureCapPlan:
    """Which elements to drop so ``len(features) + reserved <= budget``; drops from the end of ``element_keys``."""
    counts = {key: sum(1 for f in features if f.split(".", 1)[0] == key) for key in element_keys}
    droppable = [k for k in element_keys if k not in protected and counts[k] > 0]
    total = len(features) + reserved
    dropped = []
    while total > budget and droppable:
        key = droppable.pop()
        total -= counts[key]
        dropped.append(DroppedElement(key, counts[key], key in set(new_keys)))
    return FeatureCapPlan(tuple(dropped), len(features), total - reserved, reserved, total <= budget)
