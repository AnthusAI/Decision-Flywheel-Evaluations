"""Accuracy, Brier, ECE and paired item bootstrap intervals for the unified flywheel.

The metric definitions are Jev-Flywheel's (``jev_flywheel.evaluate``): confidence is the
probability of the emitted value, Brier is top-label Brier, and ECE uses 10 equal-width bins
weighted by mass. They are restated here, unweighted, so the bootstrap can recompute them on
every resample without importing Jev-Flywheel; a spec checks parity with ``summarize``.

Brier and ECE here are TOP-LABEL measures: each item contributes the confidence of the label the
classifier predicted and whether that label was correct. That is valid for any number of classes
(Emotion's six), but it is NOT the multiclass Brier score, which sums squared errors over every
class's probability; do not compare these numbers with a multiclass Brier.

Intervals are paired percentile bootstraps over items (plan section 4: 1,000 resamples), with
the same index draw applied to both arms, using this repository's percentile convention
(``effects[int(0.025 * (R - 1))]`` and ``effects[int(0.975 * (R - 1))]``). No significance
claims are made from them.

MULTI-CLASS METRICS (Emotion; the planted corpus never reports them). Macro-F1 is the primary metric on an
imbalanced multi-class test set; accuracy, per-class recall/precision/F1 and a label-ordered confusion
matrix accompany it. They need each row's predicted and gold label (``ItemResult.predicted``/``gold``).
Definitions, stated once:

* per class c: ``precision = tp / (tp + fp)``, ``recall = tp / (tp + fn)``, ``F1 = 2tp / (2tp + fp + fn)``;
  a ratio with a zero denominator is ``None`` (undefined), not 0;
* **absent classes.** A class with no gold item AND no prediction in the slice is *absent*: its precision,
  recall and F1 are all ``None`` and it is left out of the macro average (it can neither help nor hurt).
  A class with no gold items that the classifier nevertheless predicted has ``recall None``, ``precision 0``
  and ``F1 0`` and DOES count (predicting a class that is not there is an error). A class with gold items that
  is never predicted has ``precision None``, ``recall 0`` and ``F1 0`` and counts;
* macro-F1 is the unweighted mean of the F1 of every non-absent class, over the corpus's label list;
* a class with fewer than ``LOW_SUPPORT_THRESHOLD`` (10) gold items in the slice is flagged ``low_support``:
  its recall (and F1) are shown but are too noisy to lean on (dev-100 has about two 'surprise' items);
* the confusion matrix has gold labels on the rows and predicted labels on the columns, both in label order.

The paired bootstrap for macro-F1 is the existing one (same seeded index draw for both arms, 1,000
resamples by default, the same percentile convention); only the statistic recomputed per resample differs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

METRICS = ("accuracy", "brier", "ece")
HIGHER_IS_BETTER = {"accuracy": True, "brier": False, "ece": False, "macro_f1": True}
MULTICLASS_METRICS = ("macro_f1", "accuracy")   # the contrasts reported for multi-class corpora
LOW_SUPPORT_THRESHOLD = 10                      # fewer gold items than this: flag the per-class numbers


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    confidence: float
    correct: int
    predicted: Optional[str] = None   # the label the classifier emitted (set for multi-class corpora only)
    gold: Optional[str] = None        # the reference label


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


def _counts(rows: Sequence[ItemResult], labels: Sequence[str]) -> Dict[str, List[int]]:
    """``{label: [tp, fp, fn]}`` for every label; refuses rows it cannot place."""
    known = set(labels)
    out = {label: [0, 0, 0] for label in labels}
    for row in rows:
        if row.predicted is None or row.gold is None:
            raise ValueError("multi-class metrics need every row's predicted and gold label")
        for value in (row.predicted, row.gold):
            if value not in known:
                raise ValueError(f"label {value!r} is not one of the corpus labels {tuple(labels)}")
        if row.predicted == row.gold:
            out[row.gold][0] += 1
        else:
            out[row.predicted][1] += 1
            out[row.gold][2] += 1
    return out


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> Optional[float]:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def macro_f1(rows: Sequence[ItemResult], labels: Sequence[str]) -> float:
    """Unweighted mean F1 over the labels that are not absent from the slice (see the module docstring)."""
    scores = [_f1(*counts) for counts in _counts(rows, labels).values()]
    present = [score for score in scores if score is not None]
    return sum(present) / len(present) if present else 0.0


def per_class(rows: Sequence[ItemResult], labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Support, number predicted, recall, precision and F1 per label, flagging thin support."""
    out: Dict[str, Dict[str, Any]] = {}
    for label, (tp, fp, fn) in _counts(rows, labels).items():
        support = tp + fn
        out[label] = {"support": support, "predicted": tp + fp, "recall": _round(_ratio(tp, tp + fn)),
                      "precision": _round(_ratio(tp, tp + fp)), "f1": _round(_f1(tp, fp, fn)),
                      "low_support": support < LOW_SUPPORT_THRESHOLD}
    return out


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 6)


def confusion_matrix(rows: Sequence[ItemResult], labels: Sequence[str]) -> Dict[str, Any]:
    """Gold labels on the rows, predicted labels on the columns, both in ``labels`` order."""
    _counts(rows, labels)   # validates every row
    index = {label: n for n, label in enumerate(labels)}
    counts = [[0] * len(labels) for _ in labels]
    for row in rows:
        counts[index[row.gold]][index[row.predicted]] += 1
    return {"labels": list(labels), "rows": "gold", "columns": "predicted", "counts": counts}


def multiclass_summary(rows: Sequence[ItemResult], labels: Sequence[str]) -> Dict[str, Any]:
    """Everything the multi-class readout reports for one arm on one slice (text-free)."""
    scores = per_class(rows, labels)
    return {"n": len(rows), "macro_f1": round(macro_f1(rows, labels), 6), "accuracy": round(accuracy([], [r.correct for r in rows]), 6),
            "per_class": scores, "confusion_matrix": confusion_matrix(rows, labels),
            "absent_classes": [l for l in labels if scores[l]["support"] == 0 and scores[l]["predicted"] == 0],
            "low_support_classes": [l for l in labels if scores[l]["low_support"]],
            "low_support_threshold": LOW_SUPPORT_THRESHOLD}


def metric(name: str, rows: Sequence[ItemResult], labels: Optional[Sequence[str]] = None) -> float:
    if name == "macro_f1":
        if labels is None:
            raise ValueError("macro_f1 needs the corpus labels")
        return macro_f1(rows, labels)
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
                    resamples: int = 1000, seed: int = 0,
                    labels: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """``treatment - baseline`` for one metric, with a paired item bootstrap interval."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    base, treat = _aligned(baseline, treatment)
    n = len(base)
    point = metric(name, treat, labels) - metric(name, base, labels)
    rng = random.Random(seed)
    effects = []
    for _ in range(resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        effects.append(metric(name, [treat[i] for i in sample], labels) - metric(name, [base[i] for i in sample], labels))
    effects.sort()
    return {"effect": round(point, 6), "lower": round(effects[int(0.025 * (resamples - 1))], 6),
            "upper": round(effects[int(0.975 * (resamples - 1))], 6)}


def better_arm(results: Mapping[str, Sequence[ItemResult]], first: str, second: str, name: str,
               labels: Optional[Sequence[str]] = None) -> str:
    """The better of two arms on one metric's point estimate (ties go to ``first``)."""
    a, b = metric(name, results[first], labels), metric(name, results[second], labels)
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
    pairs = (("A-0", "0", "A"), ("B-0", "0", "B"), ("B-B-local", "B-local", "B"),
             ("F-0", "0", "F"), ("F-F-rand", "F-rand", "F"), ("A-c-A", "A", "A-c"))
    for label, base, treat in pairs:
        if base in results and treat in results:
            out[label] = {name: paired_interval(results[base], results[treat], name,
                                                resamples=resamples, seed=seed) for name in METRICS}
    for both, first, second in (("A+B", "A", "B"), ("A-c+F", "A-c", "F")):
        if all(arm in results for arm in (first, second, both)):
            entry: Dict[str, Dict[str, object]] = {}
            for name in METRICS:
                base = better_arm(results, first, second, name)
                entry[name] = {**paired_interval(results[base], results[both], name,
                                                 resamples=resamples, seed=seed), "baseline_arm": base}
            out[f"{both}-max({first},{second})"] = entry
    return out


def multiclass_contrasts(results: Mapping[str, Sequence[ItemResult]], labels: Sequence[str], *,
                         resamples: int = 1000, seed: int = 0) -> Dict[str, Dict[str, Dict[str, object]]]:
    """The same contrasts as ``contrasts``, for macro-F1 and accuracy, for whichever arms are present.

    Arm names are the harness's: ``A-c`` is the labels-only lever-A arm (the plan's "A" when no comments
    are supplied) and ``A-c+F`` the plan's "A+F". ``A-c+F-max(A-c,F)`` compares against the better of the two
    single levers per metric, chosen on the full-sample point estimate and held fixed across resamples.
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    pairs = (("A-0", "0", "A"), ("A-c-0", "0", "A-c"), ("B-0", "0", "B"), ("B-B-local", "B-local", "B"),
             ("F-0", "0", "F"), ("F-rand-0", "0", "F-rand"), ("F-F-rand", "F-rand", "F"),
             ("A-c+F-0", "0", "A-c+F"), ("A-c-A", "A", "A-c"))
    for label, base, treat in pairs:
        if base in results and treat in results:
            out[label] = {name: paired_interval(results[base], results[treat], name, resamples=resamples,
                                                seed=seed, labels=labels) for name in MULTICLASS_METRICS}
    for both, first, second in (("A+B", "A", "B"), ("A-c+F", "A-c", "F")):
        if all(arm in results for arm in (first, second, both)):
            entry: Dict[str, Dict[str, object]] = {}
            for name in MULTICLASS_METRICS:
                base = better_arm(results, first, second, name, labels)
                entry[name] = {**paired_interval(results[base], results[both], name, resamples=resamples,
                                                 seed=seed, labels=labels), "baseline_arm": base}
            out[f"{both}-max({first},{second})"] = entry
    return out
