"""Accuracy, Brier, ECE and paired item bootstrap intervals for the unified flywheel.

The metric definitions are Jev-Flywheel's (``jev_flywheel.evaluate``): confidence is the
probability of the emitted value, Brier is top-label Brier, and ECE uses 10 equal-width bins
weighted by mass. They are restated here, unweighted, so the bootstrap can recompute them on
every resample without importing Jev-Flywheel; a spec checks parity with ``summarize``.

Intervals are paired percentile bootstraps over items (plan section 4: 1,000 resamples), with
the same index draw applied to both arms, using this repository's percentile convention
(``effects[int(0.025 * (R - 1))]`` and ``effects[int(0.975 * (R - 1))]``). No significance
claims are made from them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

METRICS = ("accuracy", "brier", "ece")
HIGHER_IS_BETTER = {"accuracy": True, "brier": False, "ece": False}


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    confidence: float
    correct: int


def accuracy(confidences: Sequence[float], correct: Sequence[int]) -> float:
    return sum(correct) / len(correct) if correct else 0.0


def brier(confidences: Sequence[float], correct: Sequence[int]) -> float:
    if not confidences:
        return 0.0
    return sum((c - k) ** 2 for c, k in zip(confidences, correct)) / len(confidences)


def ece(confidences: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    members: List[List[int]] = [[] for _ in range(bins)]
    for index, confidence in enumerate(confidences):
        members[min(int(confidence * bins), bins - 1)].append(index)
    total = len(confidences)
    out = 0.0
    for indices in members:
        if indices:
            mean_conf = sum(confidences[i] for i in indices) / len(indices)
            acc = sum(correct[i] for i in indices) / len(indices)
            out += len(indices) / total * abs(mean_conf - acc)
    return out


_FUNCTIONS = {"accuracy": accuracy, "brier": brier, "ece": ece}


def metric(name: str, rows: Sequence[ItemResult]) -> float:
    return _FUNCTIONS[name]([r.confidence for r in rows], [r.correct for r in rows])


def summarize(rows: Sequence[ItemResult]) -> Dict[str, float]:
    return {"n": len(rows), **{name: round(metric(name, rows), 6) for name in METRICS}}


def _aligned(baseline: Sequence[ItemResult], treatment: Sequence[ItemResult]
             ) -> Tuple[List[ItemResult], List[ItemResult]]:
    by_id = {r.item_id: r for r in baseline}
    if set(by_id) != {r.item_id for r in treatment} or len(by_id) != len(baseline):
        raise ValueError("paired arms must score exactly the same items")
    ordered = sorted(by_id)
    other = {r.item_id: r for r in treatment}
    return [by_id[i] for i in ordered], [other[i] for i in ordered]


def paired_interval(baseline: Sequence[ItemResult], treatment: Sequence[ItemResult], name: str, *,
                    resamples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """``treatment - baseline`` for one metric, with a paired item bootstrap interval."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    base, treat = _aligned(baseline, treatment)
    n = len(base)
    point = metric(name, treat) - metric(name, base)
    rng = random.Random(seed)
    effects = []
    for _ in range(resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        effects.append(metric(name, [treat[i] for i in sample]) - metric(name, [base[i] for i in sample]))
    effects.sort()
    return {"effect": round(point, 6), "lower": round(effects[int(0.025 * (resamples - 1))], 6),
            "upper": round(effects[int(0.975 * (resamples - 1))], 6)}


def better_arm(results: Mapping[str, Sequence[ItemResult]], first: str, second: str, name: str) -> str:
    """The better of two arms on one metric's point estimate (ties go to ``first``)."""
    a, b = metric(name, results[first]), metric(name, results[second])
    if HIGHER_IS_BETTER[name]:
        return first if a >= b else second
    return first if a <= b else second


def contrasts(results: Mapping[str, Sequence[ItemResult]], *, resamples: int = 1000,
              seed: int = 0) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Plan section 4's contrasts, for whichever arms are present.

    ``A+B - max(A, B)`` compares against the better of A and B *per metric*, chosen on the
    full-sample point estimate and then held fixed across resamples.
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    pairs = (("A-0", "0", "A"), ("B-0", "0", "B"), ("B-B-local", "B-local", "B"))
    for label, base, treat in pairs:
        if base in results and treat in results:
            out[label] = {name: paired_interval(results[base], results[treat], name,
                                                resamples=resamples, seed=seed) for name in METRICS}
    if all(arm in results for arm in ("A", "B", "A+B")):
        entry: Dict[str, Dict[str, object]] = {}
        for name in METRICS:
            base = better_arm(results, "A", "B", name)
            entry[name] = {**paired_interval(results[base], results["A+B"], name,
                                             resamples=resamples, seed=seed), "baseline_arm": base}
        out["A+B-max(A,B)"] = entry
    return out
