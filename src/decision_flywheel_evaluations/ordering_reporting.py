"""Descriptive, integrity-bound reporting for a frozen ordering follow-up."""
from __future__ import annotations

from typing import Callable, Sequence

from .bootstrap import paired_order_bootstrap, paired_permutation_family_bootstrap
from .cli import _validate_observations_against_plan
from .manifests import DatasetManifest
from .metrics import Observation, summarize
from .ordering import OrderingPlan
from .preflight import PreflightResult, manifest_sha256
from .protocol import FrozenProtocol
from .reporting import percentage_effect, report_study
from .serialization import jsonable


_DESCRIPTIVE_INTERVAL = "nominal per-contrast, not multiplicity adjusted"
_DESCRIPTIVE_CORRECTION = "none; descriptive paired 95% intervals"


def report_ordering(
    protocol: FrozenProtocol,
    manifest: DatasetManifest,
    plan: OrderingPlan,
    observations: Sequence[Observation],
    *,
    seed: int = 0,
    resamples: int = 1_000,
) -> dict[str, object]:
    """Report every frozen ordering cell without selecting a preferred order.

    The returned values are descriptive aggregate metrics and nominal paired
    intervals only.  It contains neither raw text nor individual predictions.
    """
    _validate_arguments(protocol, manifest, plan, observations, seed, resamples)
    rows = tuple(observations)
    _validate_observations(protocol, manifest, plan, rows)
    study = report_study(rows, protocol.task.labels,
                         expected_logical_ids=tuple(cell.request.id for cell in plan.cells))
    return {
        "schema": "decision-flywheel-evaluations/ordering-report/v1",
        "scope": "descriptive ordering follow-up; not a selection or significance result",
        "provenance": {
            "initial_protocol_identity": plan.initial_protocol_identity,
            "manifest_sha256": plan.manifest_sha256,
            "initial_preflight_checksum": plan.initial_preflight_checksum,
            "initial_observations_sha256": plan.initial_observations_sha256,
            "initial_result_artifact": plan.initial_result_artifact,
            "ordering_plan_checksum": plan.checksum,
        },
        "study": jsonable(study),
        "paired_effects": _paired_effects(protocol, plan, rows, seed=seed, resamples=resamples),
        "inference": {
            "family_correction": _DESCRIPTIVE_CORRECTION,
            "significance": {
                "available": False,
                "reason": "formal significance testing was not requested",
            },
        },
    }


def _validate_arguments(protocol: FrozenProtocol, manifest: DatasetManifest,
                        plan: OrderingPlan, observations: Sequence[Observation],
                        seed: int, resamples: int) -> None:
    protocol.validate()
    manifest.validate()
    if not isinstance(plan, OrderingPlan):
        raise ValueError("ordering report requires a typed frozen ordering plan")
    plan.validate()
    if manifest_sha256(manifest) != protocol.dataset_manifest_sha256:
        raise ValueError("ordering report manifest does not match the frozen protocol")
    if (plan.initial_protocol_identity != protocol.identity
            or plan.manifest_sha256 != manifest_sha256(manifest)):
        raise ValueError("ordering plan does not bind the supplied frozen protocol and manifest")
    jev = tuple(model for model in protocol.models if model.engine.value == "jev")
    if len(jev) != 1:
        raise ValueError("ordering report requires one frozen JEV model")
    scoreboard_ids = {record.id for record in manifest.scoreboard}
    candidate_ids = {record.id for record in manifest.candidate}
    if {cell.request.target_id for cell in plan.cells} != scoreboard_ids:
        raise ValueError("ordering plan targets do not exactly cover the committed scoreboard")
    for cell in plan.cells:
        request = cell.request
        if (request.phase != "scoreboard" or not request.id.startswith("ordering:")
                or request.model != jev[0].semantic_identity
                or request.task_fingerprint != protocol.task.fingerprint
                or request.dataset_revision != manifest.revision
                or len(request.example_ids) != request.per_label * len(protocol.task.labels)
                or not set(request.example_ids) <= candidate_ids):
            raise ValueError("ordering plan request metadata does not bind the initial frozen study")
    if (not isinstance(observations, Sequence)
            or isinstance(observations, (str, bytes))
            or any(not isinstance(row, Observation) for row in observations)):
        raise ValueError("ordering observations must be a typed sequence")
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1):
        raise ValueError("ordering report seed and resamples must be valid")


def _validate_observations(protocol: FrozenProtocol, manifest: DatasetManifest,
                           plan: OrderingPlan, rows: tuple[Observation, ...]) -> None:
    cells = {cell.request.id: cell.request for cell in plan.cells}
    # Reuse the established scoreboard observation firewall. This temporary
    # wrapper intentionally has no checksum: an ordering-plan checksum is not
    # and must not be presented as its initial preflight checksum.
    _validate_observations_against_plan(
        rows, manifest,
        PreflightResult(tuple(cells.values()), (), (), 0, 0, 0, len(cells), stage="scoreboard"),
    )


def _paired_effects(protocol: FrozenProtocol, plan: OrderingPlan,
                    rows: tuple[Observation, ...], *, seed: int,
                    resamples: int) -> dict[str, dict[str, object]]:
    """Return the declared, non-selective order contrasts for each positive arm."""
    treatments = {treatment.label for treatment in plan.treatments}
    shuffled = tuple(sorted(label for label in treatments if label.startswith("shuffled-")))
    conditions = tuple(sorted({
        f"scoreboard:{cell.request.model}:{cell.request.selector}:{cell.request.per_label}"
        for cell in plan.cells if cell.request.per_label > 0
    }))
    output: dict[str, dict[str, object]] = {}
    for condition in conditions:
        expected_targets = tuple(sorted({
            cell.request.target_id for cell in plan.cells
            if cell.request.per_label > 0
            and f"scoreboard:{cell.request.model}:{cell.request.selector}:{cell.request.per_label}" == condition
        }))
        condition_rows = tuple(row for row in rows if row.condition == condition)
        complete = all(row.status == "completed" for row in condition_rows)
        for order in ("interleaved", "reversed"):
            name = f"{condition}:canonical-vs-{order}"
            output[name] = _interval_or_unavailable(
                complete,
                lambda order=order: paired_order_bootstrap(
                    condition_rows, protocol.task.labels, condition=condition,
                    baseline_order="canonical", treatment_order=order,
                    seed=seed, resamples=resamples, metric=protocol.metric,
                    expected_target_ids=expected_targets,
                ),
                _mean_draw_metric(condition_rows, protocol.task.labels, order="canonical",
                                  metric=protocol.metric),
                _mean_draw_metric(condition_rows, protocol.task.labels, order=order,
                                  metric=protocol.metric),
            )
        name = f"{condition}:canonical-vs-mean-shuffled-family"
        output[name] = _interval_or_unavailable(
            complete,
            lambda: paired_permutation_family_bootstrap(
                condition_rows, protocol.task.labels, condition=condition,
                reference_order="canonical", exchangeable_orders=shuffled,
                seed=seed, resamples=resamples, metric=protocol.metric,
                expected_target_ids=expected_targets,
            ),
            _mean_draw_metric(condition_rows, protocol.task.labels, order="canonical",
                              metric=protocol.metric),
            _mean_draw_metric(condition_rows, protocol.task.labels, orders=shuffled,
                              metric=protocol.metric),
        )
    return output


def _mean_draw_metric(rows: Sequence[Observation], labels: Sequence[str], *, metric: str,
                      order: str | None = None, orders: Sequence[str] = ()) -> float | None:
    declared_orders = (order,) if order is not None else tuple(orders)
    values: list[float] = []
    for value in declared_orders:
        for draw in sorted({row.draw for row in rows if row.order == value}):
            summary = summarize([row for row in rows if row.order == value and row.draw == draw], labels)
            score = getattr(summary, metric)
            if score is None:
                return None
            values.append(score)
    return sum(values) / len(values) if values else None


def _interval_or_unavailable(complete: bool, compute: Callable[[], object],
                             baseline: float | None, treatment: float | None) -> dict[str, object]:
    if not complete:
        return {"available": False, "reason": "declared ordering cells are incomplete"}
    if baseline is None or treatment is None:
        return {"available": False, "reason": "declared ordering cells have no scoreable mean-draw metric"}
    try:
        interval = compute()
    except ValueError:
        # This is a descriptive report boundary: do not embed raw values from
        # any caller-provided row in an error artifact.
        return {"available": False, "reason": "declared ordering cells are not pairable"}
    return {
        "available": True,
        "effect": interval.effect,
        "lower": interval.lower,
        "upper": interval.upper,
        "confidence_level": 0.95,
        "interval_scope": _DESCRIPTIVE_INTERVAL,
        "baseline_mean_draw_metric": baseline,
        "treatment_mean_draw_metric": treatment,
        "percentage_effect": percentage_effect(treatment, baseline),
    }
