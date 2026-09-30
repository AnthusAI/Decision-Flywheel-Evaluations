"""Strict text-free JSON boundary for frozen ordering follow-up plans."""
from __future__ import annotations

from pathlib import Path

from .ordering import OrderingCell, OrderingPlan
from .protocol import OrderTreatment
from .serialization import _cell, _cell_payload, _keys, _mapping, _read, _sequence, _write


_SCHEMA = "decision-flywheel-evaluations/ordering-plan/v1"
_PLAN_FIELDS = {
    "initial_protocol_identity", "manifest_sha256", "initial_preflight_checksum",
    "initial_observations_sha256", "initial_result_artifact", "treatments", "cells", "checksum",
}
_TREATMENT_FIELDS = {"display_order", "seed"}
_CELL_FIELDS = {"source_request_id", "treatment", "request"}


def write_ordering(path: str | Path, plan: OrderingPlan) -> None:
    """Write a validated ordering plan without source text or provider payloads."""
    if not isinstance(plan, OrderingPlan):
        raise ValueError("ordering plan must be typed")
    plan.validate()
    _write(path, {"schema": _SCHEMA, "ordering": _payload(plan)})


def read_ordering(path: str | Path) -> OrderingPlan:
    """Read one exact ordering-plan document and revalidate all provenance."""
    document = _read(path, _SCHEMA, {"schema", "ordering"})
    payload = _mapping(document["ordering"], "ordering plan")
    _keys(payload, _PLAN_FIELDS, "ordering plan")
    treatments = tuple(_treatment(_mapping(item, "order treatment"))
                       for item in _sequence(payload["treatments"], "order treatments"))
    cells = tuple(_ordering_cell(_mapping(item, "ordering cell"))
                  for item in _sequence(payload["cells"], "ordering cells"))
    values = (
        payload["initial_protocol_identity"], payload["manifest_sha256"],
        payload["initial_preflight_checksum"], payload["initial_observations_sha256"],
        payload["initial_result_artifact"], payload["checksum"],
    )
    if any(not isinstance(value, str) for value in values):
        raise ValueError("ordering plan scalar metadata must be strings")
    plan = OrderingPlan(payload["initial_protocol_identity"], payload["manifest_sha256"],
                        payload["initial_preflight_checksum"], payload["initial_observations_sha256"],
                        payload["initial_result_artifact"], treatments, cells, payload["checksum"])
    plan.validate()
    return plan


def _payload(plan: OrderingPlan) -> dict[str, object]:
    return {
        "initial_protocol_identity": plan.initial_protocol_identity,
        "manifest_sha256": plan.manifest_sha256,
        "initial_preflight_checksum": plan.initial_preflight_checksum,
        "initial_observations_sha256": plan.initial_observations_sha256,
        "initial_result_artifact": plan.initial_result_artifact,
        "treatments": [_treatment_payload(item) for item in plan.treatments],
        "cells": [_ordering_cell_payload(item) for item in plan.cells],
        "checksum": plan.checksum,
    }


def _treatment_payload(treatment: OrderTreatment) -> dict[str, object]:
    return {"display_order": treatment.display_order, "seed": treatment.seed}


def _treatment(payload: dict[str, object]) -> OrderTreatment:
    _keys(payload, _TREATMENT_FIELDS, "order treatment")
    if (not isinstance(payload["display_order"], str) or not isinstance(payload["seed"], int)
            or isinstance(payload["seed"], bool)):
        raise ValueError("order treatment has invalid immutable field types")
    treatment = OrderTreatment(payload["display_order"], payload["seed"])
    treatment.validate()
    return treatment


def _ordering_cell_payload(cell: OrderingCell) -> dict[str, object]:
    return {"source_request_id": cell.source_request_id,
            "treatment": _treatment_payload(cell.treatment),
            "request": _cell_payload(cell.request)}


def _ordering_cell(payload: dict[str, object]) -> OrderingCell:
    _keys(payload, _CELL_FIELDS, "ordering cell")
    source_request_id = payload["source_request_id"]
    if not isinstance(source_request_id, str):
        raise ValueError("ordering source request ID must be a string")
    treatment = _treatment(_mapping(payload["treatment"], "order treatment"))
    request = _cell(_mapping(payload["request"], "request cell"))
    return OrderingCell(source_request_id, treatment, request)
