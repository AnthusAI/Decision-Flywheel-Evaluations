"""Deterministic paired target/draw bootstrap for offline evaluation rows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .metrics import Observation, summarize, validate_rows


@dataclass(frozen=True)
class PairedInterval:
    effect: float
    lower: float
    upper: float
    replicates: tuple[float, ...]


def paired_macro_f1_bootstrap(
    rows: Sequence[Observation], labels: Sequence[str], *, baseline: str,
    treatment: str, seed: int = 0, resamples: int = 1000,
    include_order: bool = False, metric: str = "macro_f1",
    expected_target_ids: Sequence[str] | None = None,
) -> PairedInterval:
    """Resample target clusters and draw clusters as matched condition pairs.

    When both arms have multiple draws, they must use exactly the same draw
    IDs. A single-draw arm may be compared with either a multi-draw baseline or
    treatment (for example retrieval versus five random draws), but its sole
    draw is never resampled as independent evidence.
    Ordering-factor inference is intentionally unsupported rather than silently
    reducing it to a one-order analysis.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or
        isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1):
        raise ValueError("seed and resamples must be valid")
    if not isinstance(baseline, str) or not baseline or not isinstance(treatment, str) or not treatment or baseline == treatment:
        raise ValueError("baseline and treatment must differ")
    if metric not in {"macro_f1", "accuracy"}:
        raise ValueError("metric must be macro_f1 or accuracy")
    if include_order:
        raise ValueError("ordering-factor inference is not implemented")
    validate_rows(rows, labels)
    arms = [row for row in rows if row.condition in {baseline, treatment}]
    if not any(row.condition == baseline for row in arms):
        raise ValueError("missing baseline arm")
    if not any(row.condition == treatment for row in arms):
        raise ValueError("missing treatment arm")
    if any(row.status != "completed" for row in arms):
        raise ValueError("missing cells cannot support confirmatory paired inference")
    orders = tuple(sorted({row.order for row in arms}))
    if len(orders) != 1:
        raise ValueError("ordering factor requires a single order or explicit supported inference")
    index = _index(arms)
    draws = {condition: tuple(sorted({row.draw for row in arms if row.condition == condition})) for condition in (baseline, treatment)}
    baseline_multi = len(draws[baseline]) > 1
    treatment_multi = len(draws[treatment]) > 1
    if baseline_multi and treatment_multi and draws[baseline] != draws[treatment]:
        raise ValueError("multi-draw baseline and treatment must share draw set")
    targets = _targets(arms, baseline, treatment, draws, orders, expected_target_ids)

    def score(condition: str, draw_sample: Sequence[int], target_sample: Sequence[str]) -> float:
        summaries = [summarize([index[(condition, draw, orders[0], target)] for target in target_sample], labels) for draw in draw_sample]
        values = [summary.macro_f1 if metric == "macro_f1" else summary.accuracy for summary in summaries]
        # Every source cell was completed and label-validated, so this is defensive.
        if any(value is None for value in values):
            raise ValueError("paired inference has no scoreable observations")
        return sum(values) / len(values)

    shared_draws = draws[baseline] if baseline_multi else draws[treatment]
    point = score(treatment, draws[treatment], targets) - score(baseline, draws[baseline], targets)
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(resamples):
        target_sample = tuple(rng.choice(targets) for _ in targets)
        # One joint sampled draw vector is re-used by both multi-draw arms.
        draw_sample = tuple(rng.choice(shared_draws) for _ in shared_draws)
        baseline_draw_sample = draw_sample if baseline_multi else draws[baseline]
        treatment_draw_sample = draw_sample if treatment_multi else draws[treatment]
        effects.append(
            score(treatment, treatment_draw_sample, target_sample)
            - score(baseline, baseline_draw_sample, target_sample)
        )
    effects.sort()
    return PairedInterval(point, effects[int(0.025 * (resamples - 1))], effects[int(0.975 * (resamples - 1))], tuple(effects))


def _index(rows: Sequence[Observation]) -> dict[tuple[str, int, str, str], Observation]:
    index: dict[tuple[str, int, str, str], Observation] = {}
    request_ids: set[str] = set()
    for row in rows:
        if row.request_id in request_ids:
            raise ValueError("duplicate logical observation ID")
        request_ids.add(row.request_id)
        key = (row.condition, row.draw, row.order, row.target_id)
        if key in index:
            raise ValueError("duplicate paired cell")
        index[key] = row
    return index


def _targets(
    rows: Sequence[Observation], baseline: str, treatment: str,
    draws: dict[str, tuple[int, ...]], orders: Sequence[str],
    expected_target_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    groups: list[set[str]] = []
    for condition in (baseline, treatment):
        for draw in draws[condition]:
            for order in orders:
                group = {row.target_id for row in rows if (row.condition, row.draw, row.order) == (condition, draw, order)}
                if not group:
                    raise ValueError("missing condition/draw/order cell")
                groups.append(group)
    actual = groups[0]
    if any(group != actual for group in groups):
        raise ValueError("paired conditions and draws must share equal target sets")
    if expected_target_ids is not None:
        expected = tuple(expected_target_ids)
        if (not expected or any(not isinstance(target, str) or not target for target in expected)
            or len(set(expected)) != len(expected) or set(expected) != actual):
            raise ValueError("expected target manifest does not exactly cover paired targets")
    for target in actual:
        if len({row.true_label for row in rows if row.target_id == target}) != 1:
            raise ValueError("shared targets must have identical true labels")
    return tuple(sorted(actual))


def paired_order_bootstrap(
    rows: Sequence[Observation], labels: Sequence[str], *, condition: str,
    baseline_order: str, treatment_order: str, seed: int = 0, resamples: int = 1000,
    metric: str = "accuracy", expected_target_ids: Sequence[str] | None = None,
) -> PairedInterval:
    """Matched order contrast, resampling target and draw clusters independently.

    Orders are compared within each same condition/draw/target cell; they are
    never treated as independent repeat rows.
    """
    if (not isinstance(condition, str) or not condition or not isinstance(baseline_order, str)
        or not baseline_order or not isinstance(treatment_order, str) or not treatment_order
        or baseline_order == treatment_order or metric not in {"accuracy", "macro_f1"}):
        raise ValueError("orders must differ and metric must be accuracy or macro_f1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("seed and resamples must be valid")
    validate_rows(rows, labels)
    selected = [row for row in rows if row.condition == condition and row.order in {baseline_order, treatment_order}]
    if not selected or any(row.status != "completed" for row in selected):
        raise ValueError("missing cells cannot support paired order inference")
    index = _index(selected)
    draws = tuple(sorted({row.draw for row in selected}))
    targets = {row.target_id for row in selected}
    if not draws or not targets or any((condition, draw, order, target) not in index for draw in draws for order in (baseline_order, treatment_order) for target in targets):
        raise ValueError("orders and draws must share exact target cells")
    targets = tuple(sorted(targets))
    if expected_target_ids is not None and set(expected_target_ids) != set(targets):
        raise ValueError("expected target manifest does not exactly cover paired targets")
    for target in targets:
        if len({row.true_label for row in selected if row.target_id == target}) != 1:
            raise ValueError("shared targets must have identical true labels")
    def score(order, draw_sample, target_sample):
        values=[]
        for draw in draw_sample:
            summary=summarize([index[(condition, draw, order, target)] for target in target_sample], labels)
            values.append(summary.accuracy if metric == "accuracy" else summary.macro_f1)
        return sum(values)/len(values)
    point=score(treatment_order,draws,targets)-score(baseline_order,draws,targets)
    rng=random.Random(seed); effects=[]
    for _ in range(resamples):
        target_sample=tuple(rng.choice(targets) for _ in targets)
        draw_sample=tuple(rng.choice(draws) for _ in draws)
        effects.append(score(treatment_order,draw_sample,target_sample)-score(baseline_order,draw_sample,target_sample))
    effects.sort()
    return PairedInterval(point,effects[int(.025*(resamples-1))],effects[int(.975*(resamples-1))],tuple(effects))


def paired_permutation_family_bootstrap(
    rows: Sequence[Observation], labels: Sequence[str], *, condition: str,
    reference_order: str, exchangeable_orders: Sequence[str], seed: int = 0,
    resamples: int = 1000, metric: str = "accuracy",
    expected_target_ids: Sequence[str] | None = None,
) -> PairedInterval:
    """Compare a fixed reference with declared shuffled permutations only.

    Named presentation orders are not exchangeable: callers must pass only the
    predeclared shuffled-permutation family in ``exchangeable_orders``.
    """
    family = tuple(exchangeable_orders)
    fixed_orders = {"canonical", "interleaved", "reversed"}
    if (not isinstance(condition, str) or not condition
        or not isinstance(reference_order, str) or not reference_order
        or not family or len(set(family)) != len(family)
        or reference_order in family or set(family) & fixed_orders
        or any(not isinstance(order, str) or not order for order in family)
        or metric not in {"accuracy", "macro_f1"}):
        raise ValueError("declare only unique shuffled permutations distinct from reference")
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        or isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1):
        raise ValueError("seed and resamples must be valid")
    validate_rows(rows, labels)
    relevant_orders = {reference_order, *family}
    selected = [row for row in rows if row.condition == condition and row.order in relevant_orders]
    if not selected or any(row.status != "completed" for row in selected):
        raise ValueError("missing cells cannot support paired permutation inference")
    index = _index(selected)
    draws = tuple(sorted({row.draw for row in selected}))
    targets = tuple(sorted({row.target_id for row in selected}))
    if any((condition, draw, order, target) not in index
           for draw in draws for order in relevant_orders for target in targets):
        raise ValueError("reference and permutations must share exact target/draw cells")
    if expected_target_ids is not None:
        expected = tuple(expected_target_ids)
        if (not expected or any(not isinstance(target, str) or not target for target in expected)
            or len(set(expected)) != len(expected) or set(expected) != set(targets)):
            raise ValueError("expected target manifest does not exactly cover paired targets")
    for target in targets:
        if len({row.true_label for row in selected if row.target_id == target}) != 1:
            raise ValueError("shared targets must have identical true labels")

    def score(order: str, draw_sample: Sequence[int], target_sample: Sequence[str]) -> float:
        scores = []
        for draw in draw_sample:
            summary = summarize([index[(condition, draw, order, target)] for target in target_sample], labels)
            value = summary.accuracy if metric == "accuracy" else summary.macro_f1
            if value is None:
                raise ValueError("paired permutation inference has no scoreable observations")
            scores.append(value)
        return sum(scores) / len(scores)

    point = sum(score(order, draws, targets) - score(reference_order, draws, targets) for order in family) / len(family)
    rng = random.Random(seed)
    effects = []
    for _ in range(resamples):
        target_sample = tuple(rng.choice(targets) for _ in targets)
        draw_sample = tuple(rng.choice(draws) for _ in draws)
        order_sample = tuple(rng.choice(family) for _ in family)
        effects.append(sum(score(order, draw_sample, target_sample) - score(reference_order, draw_sample, target_sample)
                           for order in order_sample) / len(order_sample))
    effects.sort()
    return PairedInterval(point, effects[int(.025 * (resamples - 1))],
                          effects[int(.975 * (resamples - 1))], tuple(effects))
